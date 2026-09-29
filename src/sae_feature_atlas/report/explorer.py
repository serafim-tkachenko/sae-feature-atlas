"""Export saved evidence into a portable, offline browser report.

This is a presentation schema, independent of the collection artifact schema.
No model, tokenizer, or analysis policy is loaded or inferred here.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from html import escape
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sae_feature_atlas.analysis.token_quality import token_quality_label

REPORT_SCHEMA_VERSION = 1
EXAMPLES_PER_GROUP = 8
NEIGHBORS_PER_FEATURE = 10
HISTOGRAM_BINS = 24

TABLES = {
    "feature_cards.parquet": {"feature_id"},
    "analysis_features.parquet": {"feature_id"},
    "feature_stats.parquet": {"feature_id"},
    "top_feature_examples.parquet": {"feature_id", "activation", "text_id", "token_pos"},
    "bimodal_peak_examples.parquet": {"feature_id", "activation", "peak_label"},
    "decoder_neighbors.parquet": {"feature_i", "feature_j", "decoder_cosine"},
    "coactivation_pairs.parquet": {"feature_i", "feature_j", "coactivation_count", "jaccard"},
    "token_metadata.parquet": {"text_id", "token_pos"},
    "sae_activations_topk.parquet": {"feature_id", "activation"},
    "sae_activations_positive.parquet": {"feature_id", "activation"},
}


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if value is pd.NA or value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _Artifacts:
    def __init__(self, directory: Path):
        self.directory = directory
        self.status: dict[str, dict] = {}
        for name, required in TABLES.items():
            item = {"name": name, "status": "missing", "rows": None}
            self.status[name] = item
            if not (directory / name).exists():
                continue
            try:
                table = pq.ParquetFile(directory / name)
                missing = required - set(table.schema_arrow.names)
                if missing and table.metadata.num_rows:
                    raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
                item.update(
                    status="ready" if table.metadata.num_rows else "empty",
                    rows=table.metadata.num_rows,
                )
            except Exception as error:
                self.failed(name, error)

    def failed(self, name: str, error: Exception) -> None:
        self.status[name].update(status="unreadable", detail=str(error)[:300])

    def read(self, name: str, columns: list[str] | None = None) -> pd.DataFrame:
        if self.status[name]["status"] != "ready":
            return pd.DataFrame()
        try:
            return pd.read_parquet(self.directory / name, columns=columns)
        except Exception as error:
            self.failed(name, error)
            return pd.DataFrame()

    def json(self, name: str) -> dict:
        self.status[name] = {"name": name, "status": "missing", "rows": None}
        path = self.directory / name
        if not path.exists():
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Expected a JSON object")
            self.status[name]["status"] = "ready" if value else "empty"
            return value
        except Exception as error:
            self.failed(name, error)
            return {}


def _embedded(row: dict, key: str) -> list[dict]:
    try:
        value = json.loads(row.get(key) or "[]")
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    except (ValueError, TypeError):
        return []


def _groups(table: pd.DataFrame, column: str = "feature_id") -> dict[int, list[dict]]:
    if table.empty:
        return {}
    return {int(key): group.to_dict("records") for key, group in table.groupby(column)}


def _examples(rows: list[dict], feature_id: int, name: str) -> list[dict]:
    result = []
    for row in sorted(rows, key=lambda row: row.get("activation") or 0, reverse=True):
        if len(result) == EXAMPLES_PER_GROUP:
            break
        result.append(
            {
                "evidence_id": f"f{feature_id}:t{row.get('text_id')}:p{row.get('token_pos')}",
                "artifact": name,
                **{
                    key: row.get(key)
                    for key in (
                        "activation",
                        "text_id",
                        "token_pos",
                        "source",
                        "left_context",
                        "center_token",
                        "right_context",
                        "display_context",
                        "activation_population",
                    )
                },
            }
        )
    return result


def _histograms(artifacts: _Artifacts, name: str, feature_ids: set[int]) -> dict:
    """Two streaming passes; never send raw sparse activation rows to the browser."""
    if artifacts.status[name]["status"] != "ready":
        return {}
    limits: dict[int, tuple[float, float]] = {}
    try:
        table = pq.ParquetFile(artifacts.directory / name)

        def batches():
            for batch in table.iter_batches(
                batch_size=131_072, columns=["feature_id", "activation"]
            ):
                frame = batch.to_pandas()
                yield frame[frame.feature_id.isin(feature_ids) & np.isfinite(frame.activation)]

        for frame in batches():
            for fid, group in frame.groupby("feature_id"):
                low, high = limits.get(int(fid), (float("inf"), -float("inf")))
                limits[int(fid)] = (
                    min(low, group.activation.min()),
                    max(high, group.activation.max()),
                )
        edges = {
            fid: np.histogram_bin_edges([low, high], bins=HISTOGRAM_BINS)
            for fid, (low, high) in limits.items()
        }
        counts = {fid: np.zeros(HISTOGRAM_BINS, dtype=np.int64) for fid in limits}
        for frame in batches():
            for fid, group in frame.groupby("feature_id"):
                counts[int(fid)] += np.histogram(group.activation, bins=edges[int(fid)])[0]
        return {
            fid: {
                "edges": edges[fid].tolist(),
                "counts": count.tolist(),
                "population": "all_stored_activations",
                "count": int(count.sum()),
            }
            for fid, count in counts.items()
        }
    except Exception as error:
        artifacts.failed(name, error)
        return {}


def build_explorer_payload(run_dir: str | Path) -> dict:
    directory = Path(run_dir).expanduser()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    artifacts = _Artifacts(directory)
    warnings: list[str] = []
    lineage = artifacts.json("lineage.json")
    try:
        for key in ("fingerprints", "fingerprint_payloads", "git"):
            if key in lineage and not isinstance(lineage[key], dict):
                raise ValueError(f"Invalid lineage field: {key}")
        recorded_collection = lineage.get("fingerprint_payloads", {}).get("collection", {})
        if not isinstance(recorded_collection, dict) or any(
            not isinstance(recorded_collection.get(key, {}), dict)
            for key in ("model", "collection")
        ):
            raise ValueError("Invalid collection payload in lineage")
        if lineage.get("artifact_schema_version") == 2 and (
            not recorded_collection
            or not all(
                isinstance(lineage.get("fingerprints", {}).get(key), str)
                and lineage["fingerprints"][key]
                for key in ("collection", "analysis")
            )
        ):
            raise ValueError("Lineage schema 2 is missing collection identity or fingerprints")
    except ValueError as error:
        artifacts.failed("lineage.json", error)
        lineage = {}
    coactivation_metadata = artifacts.json("coactivation_metadata.json")
    reanalysis = artifacts.json("reanalysis.json")
    if reanalysis:
        if (reanalysis.get("schema_version") != 1
                or reanalysis.get("kind") != "saved_evidence_reanalysis"
                or not isinstance(reanalysis.get("source_run"), str)
                or not isinstance(reanalysis.get("notes"), list)
                or not all(isinstance(note, str) for note in reanalysis["notes"])
                or not isinstance(reanalysis.get("activation_filter"), dict)):
            artifacts.failed("reanalysis.json", ValueError("Unsupported reanalysis record"))
            reanalysis = {}
        else:
            warnings.extend(reanalysis["notes"])
    schema = lineage.get("artifact_schema_version")
    provenance = "recorded" if schema == 2 else "legacy" if not lineage else "unsupported"
    if artifacts.status["lineage.json"]["status"] == "unreadable":
        provenance = "unreadable"
    if provenance != "recorded":
        warnings.append(
            "Collection provenance is unavailable or uses an unsupported schema. "
            "Treat the original collection as unverified. Viewing a report does not re-run inference."
        )

    cards = pd.DataFrame()
    feature_source = "feature_cards.parquet"
    for name in ("feature_cards.parquet", "analysis_features.parquet", "feature_stats.parquet"):
        cards = artifacts.read(name)
        if artifacts.status[name]["status"] in {"ready", "empty"}:
            feature_source = name
            break
    if cards.empty and artifacts.status[feature_source]["status"] == "empty":
        warnings.append("No features are present in the saved selection. Check feature-selection "
                        "thresholds and sample size; rejected features are not substituted.")
    if feature_source != "feature_cards.parquet":
        warnings.append(
            f"Feature cards are unavailable. Showing {feature_source}; "
            + ("some features may not have passed analysis selection."
               if feature_source == "feature_stats.parquet" else "showing the saved analysis selection.")
        )
    if not cards.empty:
        ids = pd.to_numeric(cards.feature_id, errors="coerce")
        if ids.isna().any() or (ids < 0).any() or (ids % 1 != 0).any() or ids.duplicated().any():
            raise ValueError(
                f"{feature_source} must contain unique non-negative integer feature IDs"
            )
        cards = cards.copy()
        cards["feature_id"] = ids.astype(int)

    def unique(column: str):
        values = cards[column].dropna().unique() if column in cards else []
        if len(values) > 1:
            raise ValueError(
                f"Conflicting {column} values in {feature_source}; cannot combine runs"
            )
        return values[0] if len(values) else None

    collection = lineage.get("fingerprint_payloads", {}).get("collection", {})
    model = collection.get("model", {})
    config = collection.get("collection", {})
    metadata = {
        "name": directory.name,
        "provenance": provenance,
        "artifact_schema": schema,
        "fingerprints": lineage.get("fingerprints", {}),
        "git": lineage.get("git", {}),
        "reanalysis_source": reanalysis.get("source_run"),
    }
    for key in (
        "model_name",
        "sae_release",
        "sae_id",
        "layer",
        "hook_name",
        "corpus",
        "activation_mode",
        "top_k",
    ):
        recorded = model.get(key, config.get("top_k_features_per_token" if key == "top_k" else key))
        saved = unique(key)
        if recorded is not None and saved is not None and recorded != saved:
            raise ValueError(f"{key} in {feature_source} conflicts with lineage.json")
        metadata[key] = recorded if recorded is not None else saved
    metadata["stored_tokens"] = unique("stored_token_denominator")
    metadata["analysis_tokens"] = unique("analysis_token_denominator")
    metadata["feature_source"] = feature_source
    metadata["collected_token_rows"] = artifacts.status["token_metadata.parquet"]["rows"]
    token_ids = artifacts.read("token_metadata.parquet", columns=["text_id"])
    metadata["collected_texts"] = int(token_ids.text_id.nunique()) if not token_ids.empty else None

    top = _groups(artifacts.read("top_feature_examples.parquet"))
    regimes = _groups(artifacts.read("bimodal_peak_examples.parquet"))
    decoder = _groups(artifacts.read("decoder_neighbors.parquet"), "feature_i")
    pairs = artifacts.read("coactivation_pairs.parquet")
    coactivation: dict[int, list[dict]] = {}
    for pair in pairs.to_dict("records"):
        for source, target in (("feature_i", "feature_j"), ("feature_j", "feature_i")):
            coactivation.setdefault(int(pair[source]), []).append(
                {**pair, "neighbor_feature_id": int(pair[target])}
            )

    mode = metadata["activation_mode"]
    candidates = [
        name
        for name in ("sae_activations_topk.parquet", "sae_activations_positive.parquet")
        if artifacts.status[name]["status"] in {"ready", "empty"}
    ]
    activation_file = f"sae_activations_{mode}.parquet" if mode in {"topk", "positive"} else None
    if activation_file is None and len(candidates) == 1:
        activation_file = candidates[0]
        metadata["activation_mode"] = "topk" if "topk" in activation_file else "positive"
    if len(candidates) > 1:
        warnings.append(
            "Both activation storage formats exist. Histograms use the recorded collection "
            "mode only; without it, the distribution is unavailable."
        )
    histograms = (
        _histograms(artifacts, activation_file, set(cards.feature_id))
        if activation_file and not cards.empty
        else {}
    )

    features = []
    for row in cards.to_dict("records"):
        row = _clean(row)
        fid = int(row["feature_id"])
        explicit = row.get("analysis_token_frequency") is not None and not pd.isna(
            row.get("analysis_token_frequency")
        )
        examples = {}
        for group in ("top", "low", "high"):
            saved_rows = (
                top.get(fid, [])
                if group == "top"
                else [item for item in regimes.get(fid, []) if item.get("peak_label") == group]
            )
            source = (
                "top_feature_examples.parquet"
                if group == "top"
                else "bimodal_peak_examples.parquet"
            )
            if not saved_rows and artifacts.status[source]["status"] not in {"ready", "empty"}:
                saved_rows = _embedded(
                    row, "top_examples_json" if group == "top" else f"bimodal_{group}_examples_json"
                )
                source = feature_source
            examples[group] = _examples(saved_rows, fid, source)
        neighbors = {}
        for kind in ("decoder", "coactivation"):
            source = decoder if kind == "decoder" else coactivation
            rows = source.get(fid, [])
            artifact = "decoder_neighbors.parquet" if kind == "decoder" else "coactivation_pairs.parquet"
            if not rows and artifacts.status[artifact]["status"] not in {"ready", "empty"}:
                rows = _embedded(row, f"top_{kind}_neighbors_json")
            metric = "decoder_cosine" if kind == "decoder" else "jaccard"
            rows = sorted(rows, key=lambda item: item.get(metric) or 0, reverse=True)[
                :NEIGHBORS_PER_FEATURE
            ]
            neighbors[kind] = [
                {
                    "id": int(item.get("neighbor_feature_id", item.get("feature_j", -1))),
                    "value": item.get(metric),
                    "support": item.get("coactivation_count"),
                    "pmi": item.get("pmi"),
                }
                for item in rows
            ]
        features.append(
            {
                "id": fid,
                "label": row.get("primary_label") or "unlabeled",
                "priority": row.get("manual_priority"),
                "frequency": row.get("analysis_token_frequency" if explicit else "token_frequency"),
                "population": "analysis_activations" if explicit else "legacy_unspecified",
                "support": row.get(
                    "analysis_activation_count" if explicit else "n_token_activations"
                ),
                "text_count": row.get("analysis_text_count" if explicit else "n_texts"),
                "denominator": row.get("analysis_token_denominator") if explicit else None,
                "stored_frequency": row.get("stored_token_frequency"),
                "stored_support": row.get("stored_activation_count"),
                "p99": row.get("p99_activation"),
                "median": row.get("p50_activation"),
                "artifact_score": row.get("artifact_score"),
                "triage_score": row.get("interpretability_triage_score"),
                "examples": examples,
                "neighbors": neighbors,
                "histogram": histograms.get(fid),
                "diagnostics": {
                    key: row.get(key)
                    for key in (
                        "bimodality_score",
                        "pc_mass_observed",
                        "effective_pc_dim",
                        "gca_at_10",
                    )
                },
            }
        )
    target_quality_counts = Counter(
        token_quality_label(example["center_token"])
        for feature in features
        for examples in feature["examples"].values()
        for example in examples
        if example["center_token"] is not None
    )
    formatting_targets = sum(count for label, count in target_quality_counts.items() if label != "clean")
    if formatting_targets:
        warnings.append(
            f"Saved examples contain {formatting_targets:,} formatting targets "
            "(quotes, punctuation, whitespace, symbols or other token artifacts). "
            "This export preserves saved evidence; it does not apply a new token filter. "
            "Reanalyze examples and statistics together to change their eligibility policy."
        )
    return _clean(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run": metadata,
            "warnings": warnings,
            "artifacts": list(artifacts.status.values()),
            "features": features,
            "coactivation_metadata": coactivation_metadata,
            "export_policy": {
                "examples_per_group": EXAMPLES_PER_GROUP,
                "neighbors_per_feature": NEIGHBORS_PER_FEATURE,
                "histogram_population": "all_stored_activations",
                "example_target_quality_counts": dict(target_quality_counts),
            },
        }
    )


def write_explorer(run_dir: str | Path, output: str | Path, *, links: dict | None = None) -> Path:
    """Write a self-contained report; the original run is never modified."""
    path = Path(output).expanduser()
    if path.suffix.lower() != ".html":
        raise ValueError("Explorer output must be an .html file")
    payload = build_explorer_payload(run_dir)
    payload["links"] = links or {}
    resources = files("sae_feature_atlas.report")
    javascript = resources.joinpath("static/explorer.js").read_text(encoding="utf-8")
    javascript = javascript.replace("</script", "<\\/script")
    stylesheet = resources.joinpath("static/explorer.css").read_text(encoding="utf-8")
    notices = resources.joinpath("THIRD_PARTY_NOTICES.txt").read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Corpus text and provenance are untrusted, including inside a script data block.
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; font-src 'none'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>Feature Atlas · {escape(payload["run"]["name"])}</title>
<!-- {notices} -->
<style>{stylesheet}</style></head><body>
<div id="root"></div><noscript>Enable JavaScript to explore this report. The Markdown summary remains available separately.</noscript>
<script id="atlas-data" type="application/json">{data}</script>
<script>{javascript}</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
