from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

PUNCT_RE = re.compile(r"^[\s\W_]+$", re.UNICODE)

QUOTE_LIKE = {
    "'",
    "’",
    "‘",
    '"',
    "“",
    "”",
    "`",
    "``",
    "''",
}

SPACE_LIKE = {
    " ",
    "\n",
    "\t",
    "",
}


def is_punctuation_like(token: str) -> bool:
    """Return True if a token looks mostly like punctuation/space/symbols."""
    return bool(PUNCT_RE.match(str(token)))


def is_quote_like(token: str) -> bool:
    """Return True for apostrophe/quote-like tokens."""
    return str(token).strip() in QUOTE_LIKE


def is_space_like(token: str) -> bool:
    """Return True for whitespace-like tokens."""
    token = str(token)
    return token in SPACE_LIKE or token.isspace()


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _source_is_informative_for_run(
    acts: pd.DataFrame,
    min_sources: int = 3,
    max_top_source_share: float = 0.85,
) -> bool:
    """Decide whether source concentration is meaningful for this run.

    If almost all activations come from one source, source concentration is not
    useful. This happens in runs like `pile-10k` plus a few manual examples:
    technically there are multiple source values, but the run is effectively
    single-source.

    We only treat source as informative when:
    - there are at least `min_sources` sources;
    - the dominant source is not overwhelmingly dominant.
    """
    if "source" not in acts.columns or acts.empty:
        return False

    shares = acts["source"].astype(str).value_counts(normalize=True)

    if len(shares) < min_sources:
        return False

    return float(shares.iloc[0]) <= max_top_source_share


def _first_top_examples(top_examples: pd.DataFrame, feature_id: int, n: int = 5) -> list[dict]:
    if top_examples.empty:
        return []

    rows = top_examples[top_examples["feature_id"] == feature_id].copy()
    if rows.empty:
        return []

    cols = [
        "rank",
        "activation",
        "source",
        "text_id",
        "token_pos",
        "left_context",
        "center_token",
        "right_context"]
    cols = [col for col in cols if col in rows.columns]

    return rows.sort_values("activation", ascending=False).head(n)[cols].to_dict("records")


def _manual_priority_from_scores(
    artifact_score: float,
    semantic_score: float,
    n_activations: int,
    high_intensity: bool,
) -> str:
    """Local inspection priority before global feature-card labels are assigned.

    This should be conservative. A feature should not become high priority just
    because it is not an obvious formatting artifact.
    """
    if artifact_score >= 0.35:
        return "low"

    if high_intensity and artifact_score < 0.25:
        return "high"

    if semantic_score >= 0.95 and n_activations >= 100 and artifact_score < 0.10:
        return "high"

    if semantic_score >= 0.85 and n_activations >= 50 and artifact_score < 0.18:
        return "medium"

    return "unreviewed"


def _feature_labels_from_scores(
    quote_share: float,
    punctuation_share: float,
    space_share: float,
    boundary_share: float,
    token_concentration: float,
    source_concentration: float,
    source_is_informative: bool,
    position_concentration: float,
    artifact_score: float,
    semantic_score: float,
    n_activations: int,
) -> list[str]:
    """Assign local inspection labels.

    These labels are heuristic diagnostics. They are not final explanations.
    """
    labels: list[str] = []

    if quote_share >= 0.25:
        labels.append("quote_like")

    if punctuation_share >= 0.50:
        labels.append("punctuation_like")

    if space_share >= 0.25:
        labels.append("space_like")

    if boundary_share >= 0.30:
        labels.append("boundary_heavy")

    if token_concentration >= 0.50:
        labels.append("token_concentrated")

    # Important:
    # In a single-corpus run, source_concentration is usually meaningless because
    # almost every feature is concentrated in the only available source.
    # We only label source concentration when the run actually contains multiple
    # source categories.
    if source_is_informative and source_concentration >= 0.70:
        labels.append("source_concentrated")

    if position_concentration >= 0.40:
        labels.append("position_concentrated")

    if artifact_score >= 0.35:
        labels.append("likely_artifact")

    # This is deliberately weaker than "semantic feature".
    # It means "worth manually inspecting as a potentially interpretable feature".
    if (
        artifact_score < 0.10
        and semantic_score >= 0.95
        and n_activations >= 100
        and token_concentration < 0.20
        and position_concentration < 0.25
        and punctuation_share < 0.20
        and quote_share < 0.05
        and space_share < 0.10
        and boundary_share < 0.15
    ):
        labels.append("manual_review")
    
    return labels


def summarize_feature(
    acts: pd.DataFrame,
    top_examples: pd.DataFrame,
    feature_id: int,
    max_seq_len: int = 256,
    boundary_margin: int = 16,
    source_is_informative: bool | None = None,
) -> dict:
    """Summarize one feature.

    The pipeline should usually prefer `summarize_features_batch`, which is much
    faster for all features.
    """
    f = acts[acts["feature_id"] == feature_id].copy()

    if f.empty:
        return {"feature_id": int(feature_id), "status": "missing"}

    if source_is_informative is None:
        source_is_informative = _source_is_informative_for_run(acts)

    return _summarize_feature_group(
        feature_id=feature_id,
        group=f,
        top_examples=top_examples,
        max_seq_len=max_seq_len,
        boundary_margin=boundary_margin,
        source_is_informative=source_is_informative,
    )


def _summarize_feature_group(
    feature_id: int,
    group: pd.DataFrame,
    top_examples: pd.DataFrame,
    max_seq_len: int,
    boundary_margin: int,
    source_is_informative: bool,
) -> dict:
    token_str = group["token_str"].astype(str)
    token_pos = pd.to_numeric(group["token_pos"], errors="coerce").fillna(-1)

    n = len(group)
    n_texts = int(group["text_id"].nunique()) if "text_id" in group.columns else 0

    quote_share = float(token_str.map(is_quote_like).mean())
    punctuation_share = float(token_str.map(is_punctuation_like).mean())
    space_share = float(token_str.map(is_space_like).mean())
    boundary_share = float((token_pos >= max_seq_len - boundary_margin).mean())

    token_counts = token_str.value_counts()
    source_counts = group["source"].astype(str).value_counts() if "source" in group.columns else pd.Series()
    position_counts = token_pos.value_counts()

    token_concentration = _safe_ratio(float(token_counts.iloc[0]), n) if not token_counts.empty else 0.0
    raw_source_concentration = (
        _safe_ratio(float(source_counts.iloc[0]), n) if not source_counts.empty else 0.0
    )
    source_concentration = raw_source_concentration if source_is_informative else 0.0
    position_concentration = (
        _safe_ratio(float(position_counts.iloc[0]), n) if not position_counts.empty else 0.0
    )

    # Artifact score: conservative and focused on token/position artifacts.
    # Source concentration is included only if source is informative.
    artifact_score = (
        0.24 * quote_share
        + 0.20 * punctuation_share
        + 0.12 * space_share
        + 0.20 * boundary_share
        + 0.18 * token_concentration
        + 0.04 * source_concentration
        + 0.02 * position_concentration
    )

    # Semantic score is only a triage score.
    # It rewards variety and penalizes obvious artifact indicators.
    diversity_score = 1.0 - max(token_concentration, position_concentration)
    non_artifact_score = 1.0 - min(1.0, artifact_score)
    token_quality_score = 1.0 - min(1.0, punctuation_share + quote_share + space_share)
    semantic_score = max(
        0.0,
        0.40 * non_artifact_score + 0.35 * diversity_score + 0.25 * token_quality_score,
    )

    activation = pd.to_numeric(group["activation"], errors="coerce")
    p99_activation = float(activation.quantile(0.99)) if activation.notna().any() else 0.0
    high_intensity = p99_activation > 0 and p99_activation >= 500.0

    labels = _feature_labels_from_scores(
        quote_share=quote_share,
        punctuation_share=punctuation_share,
        space_share=space_share,
        boundary_share=boundary_share,
        token_concentration=token_concentration,
        source_concentration=source_concentration,
        source_is_informative=source_is_informative,
        position_concentration=position_concentration,
        artifact_score=artifact_score,
        semantic_score=semantic_score,
        n_activations=n,
    )

    manual_priority = _manual_priority_from_scores(
        artifact_score=artifact_score,
        semantic_score=semantic_score,
        n_activations=n,
        high_intensity=high_intensity,
    )

    return {
        "feature_id": int(feature_id),
        "n_activations": int(n),
        "n_texts": int(n_texts),
        "token_pos_min": int(token_pos.min()) if len(token_pos) else -1,
        "token_pos_median": float(token_pos.median()) if len(token_pos) else math.nan,
        "token_pos_max": int(token_pos.max()) if len(token_pos) else -1,
        "quote_share": float(quote_share),
        "punctuation_share": float(punctuation_share),
        "space_share": float(space_share),
        "boundary_share": float(boundary_share),
        "token_concentration": float(token_concentration),
        "source_concentration": float(source_concentration),
        "raw_source_concentration": float(raw_source_concentration),
        "source_is_informative": bool(source_is_informative),
        "position_concentration": float(position_concentration),
        "artifact_score": float(artifact_score),
        "semantic_score": float(semantic_score),
        "manual_priority": manual_priority,
        "inspection_labels": labels,
        "top_tokens": token_counts.head(10).to_dict(),
        "top_positions": {int(k): int(v) for k, v in position_counts.head(10).to_dict().items()},
        "top_sources": source_counts.head(10).to_dict() if not source_counts.empty else {},
        "top_examples": _first_top_examples(top_examples, int(feature_id), n=5),
    }


def summarize_features_batch(
    acts: pd.DataFrame,
    top_examples: pd.DataFrame,
    feature_ids: list[int],
    max_seq_len: int = 256,
    boundary_margin: int = 16,
) -> list[dict]:
    """Summarize many features efficiently."""
    if not feature_ids:
        return []

    wanted = set(int(fid) for fid in feature_ids)
    subset = acts[acts["feature_id"].isin(wanted)].copy()

    # Source concentration is only useful when the run has a genuinely mixed corpus
    source_is_informative = _source_is_informative_for_run(acts)

    summaries: list[dict] = []
    seen: set[int] = set()

    for feature_id, group in subset.groupby("feature_id"):
        seen.add(int(feature_id))
        summaries.append(
            _summarize_feature_group(
                feature_id=int(feature_id),
                group=group,
                top_examples=top_examples,
                max_seq_len=max_seq_len,
                boundary_margin=boundary_margin,
                source_is_informative=source_is_informative,
            )
        )

    missing = sorted(wanted - seen)
    for feature_id in missing:
        summaries.append({"feature_id": int(feature_id), "status": "missing"})

    return summaries


def summarize_pair(
    acts: pd.DataFrame,
    top_examples: pd.DataFrame,
    feature_i: int,
    feature_j: int,
    pair_row: dict | None = None,
    max_seq_len: int = 256,
    boundary_margin: int = 16,
    feature_summary_by_id: dict[int, dict] | None = None,
) -> dict:
    """Summarize a pair of features.

    Pair-level inspection is useful because artifact behavior can be visible in
    a coactivation clique even when individual features are below artifact
    thresholds.
    """
    if feature_summary_by_id is None:
        feature_summary_by_id = {}

    fi = feature_summary_by_id.get(feature_i)
    if fi is None:
        fi = summarize_feature(
            acts,
            top_examples,
            feature_i,
            max_seq_len=max_seq_len,
            boundary_margin=boundary_margin,
        )

    fj = feature_summary_by_id.get(feature_j)
    if fj is None:
        fj = summarize_feature(
            acts,
            top_examples,
            feature_j,
            max_seq_len=max_seq_len,
            boundary_margin=boundary_margin,
        )

    shared = acts[acts["feature_id"].isin([feature_i, feature_j])].copy()
    counts = (
        shared.groupby(["text_id", "token_pos"])["feature_id"]
        .nunique()
        .reset_index(name="n_features")
    )
    shared_positions = counts[counts["n_features"] == 2][["text_id", "token_pos"]]
    shared_rows = shared.merge(shared_positions, on=["text_id", "token_pos"], how="inner")

    shared_top_tokens = shared_rows["token_str"].astype(str).value_counts().head(10).to_dict()

    pair_artifact_score = float(
        max(
            fi.get("artifact_score", 0.0),
            fj.get("artifact_score", 0.0),
        )
    )

    shared_token_concentration = 0.0
    if len(shared_rows) > 0:
        shared_token_counts = shared_rows["token_str"].astype(str).value_counts()
        if not shared_token_counts.empty:
            shared_token_concentration = float(shared_token_counts.iloc[0] / len(shared_rows))

    pair_suspicion_score = max(
        pair_artifact_score,
        shared_token_concentration,
    )

    labels: list[str] = []

    if pair_suspicion_score >= 0.50:
        labels.append("likely_artifact_pair")

    if "quote_like" in fi.get("inspection_labels", []) or "quote_like" in fj.get(
        "inspection_labels",
        [],
    ):
        labels.append("quote_related_pair")

    if "boundary_heavy" in fi.get("inspection_labels", []) or "boundary_heavy" in fj.get(
        "inspection_labels",
        [],
    ):
        labels.append("boundary_related_pair")

    if pair_suspicion_score < 0.35 and len(shared_positions) > 0:
        labels.append("manual_priority_pair")

    manual_priority = "low" if pair_suspicion_score >= 0.50 else "medium"

    return {
        "feature_i": int(feature_i),
        "feature_j": int(feature_j),
        "pair_metrics": pair_row or {},
        "pair_artifact_score": float(pair_artifact_score),
        "pair_suspicion_score": float(pair_suspicion_score),
        "shared_token_concentration": float(shared_token_concentration),
        "manual_priority": manual_priority,
        "inspection_labels": labels,
        "shared_activation_count_estimate": int(len(shared_positions)),
        "shared_top_tokens": shared_top_tokens,
        "feature_i_summary": fi,
        "feature_j_summary": fj,
    }


def feature_summaries_to_frame(items: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []

    for item in items:
        rows.append(
            {
                "feature_id": item["feature_id"],
                "n_activations_inspected": item.get("n_activations", 0),
                "quote_share": item.get("quote_share", math.nan),
                "punctuation_share": item.get("punctuation_share", math.nan),
                "space_share": item.get("space_share", math.nan),
                "boundary_share": item.get("boundary_share", math.nan),
                "token_concentration": item.get("token_concentration", math.nan),
                "source_concentration": item.get("source_concentration", math.nan),
                "raw_source_concentration": item.get("raw_source_concentration", math.nan),
                "source_is_informative": item.get("source_is_informative", False),
                "position_concentration": item.get("position_concentration", math.nan),
                "artifact_score": item.get("artifact_score", math.nan),
                "semantic_score": item.get("semantic_score", math.nan),
                "manual_priority": item.get("manual_priority", "unreviewed"),
                "inspection_labels": ",".join(item.get("inspection_labels", [])),
                "top_tokens_json": _json(item.get("top_tokens", {})),
                "top_positions_json": _json(item.get("top_positions", {})),
                "top_sources_json": _json(item.get("top_sources", {})),
            }
        )

    return pd.DataFrame(rows)


def pair_summaries_to_frame(items: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []

    for item in items:
        metrics = item.get("pair_metrics", {})

        rows.append(
            {
                "feature_i": item["feature_i"],
                "feature_j": item["feature_j"],
                "pair_artifact_score": item.get("pair_artifact_score", math.nan),
                "pair_suspicion_score": item.get("pair_suspicion_score", math.nan),
                "shared_token_concentration": item.get("shared_token_concentration", math.nan),
                "manual_priority": item.get("manual_priority", "unreviewed"),
                "inspection_labels": ",".join(item.get("inspection_labels", [])),
                "shared_activation_count_estimate": item.get("shared_activation_count_estimate", 0),
                "jaccard": metrics.get("jaccard", None),
                "pmi": metrics.get("pmi", None),
                "coactivation_count": metrics.get("coactivation_count", None),
                "shared_top_tokens_json": _json(item.get("shared_top_tokens", {})),
            }
        )

    return pd.DataFrame(rows)


def write_inspection_reports(
    run_name: str,
    report_dir: Path,
    feature_summaries: list[dict],
    pair_summaries: list[dict],
    bimodal_summaries: list[dict],
) -> None:
    """Write machine-readable and human-readable inspection reports."""
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_name": run_name,
        "feature_summaries": feature_summaries,
        "pair_summaries": pair_summaries,
        "bimodal_feature_summaries": bimodal_summaries,
    }

    (report_dir / "inspection_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    def first_example(item: dict) -> str:
        examples = item.get("top_examples") or []
        if not examples:
            return "No example available."

        ex = examples[0]
        return (
            f'activation={ex["activation"]:.2f}, source={ex["source"]}, '
            f'text_id={ex["text_id"]}, token_pos={ex["token_pos"]}\n'
            f'{ex["left_context"]}[{ex["center_token"]}]{ex["right_context"]}'
        )

    likely_artifacts = sorted(
        feature_summaries,
        key=lambda x: x.get("artifact_score", -1),
        reverse=True,
    )[:20]

    manual_reviews = sorted(
        [
            x for x in feature_summaries
            if "manual_review" in x.get("inspection_labels", [])
        ],
        key=lambda x: x.get("semantic_score", 0),
        reverse=True,
    )[:20]

    suspicious_pairs = sorted(
        pair_summaries,
        key=lambda x: x.get("pair_suspicion_score", -1),
        reverse=True,
    )[:20]

    lines: list[str] = []

    lines.append(f"# Automated inspection report: `{run_name}`")
    lines.append("")
    lines.append(
        "This is a heuristic triage report. Labels are not final semantic interpretations."
    )
    lines.append("")
    lines.append("## What this report tries to catch")
    lines.append("")
    lines.append("- formatting/tokenization artifacts")
    lines.append("- quote/punctuation-heavy features")
    lines.append("- boundary-position-heavy features")
    lines.append("- token-concentrated features")
    lines.append("- suspicious coactivation cliques")
    lines.append("- features that look worth manual semantic inspection")
    lines.append("")
    lines.append("Note: `source_concentrated` is only used when the run contains multiple source categories.")
    lines.append("")

    lines.append("## Top likely artifact features")
    lines.append("")
    for item in likely_artifacts:
        lines.append(f"### Feature {item['feature_id']}")
        lines.append(f"- Labels: `{', '.join(item.get('inspection_labels', [])) or 'none'}`")
        lines.append(f"- Artifact score: `{item.get('artifact_score', math.nan):.3f}`")
        lines.append(f"- Semantic score: `{item.get('semantic_score', math.nan):.3f}`")
        lines.append(
            f"- Quote / punctuation / boundary share: "
            f"`{item.get('quote_share', math.nan):.2f}` / "
            f"`{item.get('punctuation_share', math.nan):.2f}` / "
            f"`{item.get('boundary_share', math.nan):.2f}`"
        )
        lines.append("")
        lines.append("Top tokens:")
        lines.append("```text")
        lines.append(json.dumps(item.get("top_tokens", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("Example:")
        lines.append("```text")
        lines.append(first_example(item))
        lines.append("```")
        lines.append("")

    lines.append("## Top manual-review features")
    lines.append("")
    for item in manual_reviews:
        lines.append(f"### Feature {item['feature_id']}")
        lines.append(f"- Semantic score: `{item.get('semantic_score', math.nan):.3f}`")
        lines.append(f"- Artifact score: `{item.get('artifact_score', math.nan):.3f}`")
        lines.append(f"- Labels: `{', '.join(item.get('inspection_labels', [])) or 'none'}`")
        lines.append("")
        lines.append("Example:")
        lines.append("```text")
        lines.append(first_example(item))
        lines.append("```")
        lines.append("")

    lines.append("## Suspicious coactivation pairs")
    lines.append("")
    for item in suspicious_pairs:
        metrics = item.get("pair_metrics", {})
        lines.append(f"### Pair {item['feature_i']} / {item['feature_j']}")
        lines.append(f"- Labels: `{', '.join(item.get('inspection_labels', [])) or 'none'}`")
        lines.append(f"- Pair suspicion score: `{item.get('pair_suspicion_score', math.nan):.3f}`")
        lines.append(f"- Shared token concentration: `{item.get('shared_token_concentration', math.nan):.3f}`")
        lines.append(f"- Jaccard: `{metrics.get('jaccard', 'n/a')}`")
        lines.append(f"- PMI: `{metrics.get('pmi', 'n/a')}`")
        lines.append(f"- Coactivation count: `{metrics.get('coactivation_count', 'n/a')}`")
        lines.append("")
        lines.append("Shared top tokens:")
        lines.append("```text")
        lines.append(json.dumps(item.get("shared_top_tokens", {}), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## Bimodality-ranked feature summaries")
    lines.append("")
    for item in bimodal_summaries[:30]:
        lines.append(f"### Feature {item['feature_id']}")
        lines.append(f"- Labels: `{', '.join(item.get('inspection_labels', [])) or 'none'}`")
        lines.append(f"- Artifact score: `{item.get('artifact_score', math.nan):.3f}`")
        lines.append(f"- Semantic score: `{item.get('semantic_score', math.nan):.3f}`")
        lines.append("")

    (report_dir / "inspection_report.md").write_text("\n".join(lines), encoding="utf-8")