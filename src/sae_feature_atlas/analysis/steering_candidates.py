from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-12


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _text(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].fillna("").astype(str)


def _has_label(series: pd.Series, label: str) -> pd.Series:
    return series.fillna("").astype(str).str.split(",").map(
        lambda xs: label in {x.strip() for x in xs if x.strip()}
    )


def _rank01(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Robust rank-normalize a score to [0, 1].

    Rank normalization is deliberate here: metrics such as activation magnitude,
    observed PCA mass, and graph agreement can live on very different scales
    across layers/models/corpora. Ranking gives a stable candidate selector for
    a first-pass atlas report without pretending that the weights are universal.
    """
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() == 0:
        return pd.Series(0.0, index=series.index)
    ranks = s.rank(method="average", pct=True)
    ranks = ranks.fillna(0.0)
    if not higher_is_better:
        ranks = 1.0 - ranks
    return ranks.clip(0.0, 1.0)


def _safe_log_activation(p99: pd.Series) -> pd.Series:
    values = np.log1p(np.clip(pd.to_numeric(p99, errors="coerce").fillna(0.0), 0.0, None))
    return _rank01(pd.Series(values, index=p99.index), higher_is_better=True)


def _coverage_coherence_score(cards: pd.DataFrame) -> pd.Series:
    """Score useful residual coverage without rewarding trivial top-20 mass.

    Important distinction:
    - `pc_norm_mass_top_20` can be 1.0 simply because we only computed 20 PCA
      components. It is not useful as a discriminative signal.
    - `pc_mass_observed` is the absolute squared decoder-vector mass captured by
      the observed residual PCA subspace. This must be high before we call a
      feature's residual coverage coherent.

    A feature with compact normalized PCA mass but very low observed PCA mass
    should not be treated as strongly coverage-coherent.
    """
    observed = _num(cards, "pc_mass_observed", default=np.nan)
    if observed.isna().all():
        observed = _num(cards, "pc_observed_mass", default=0.0)

    observed = observed.clip(0.0, 1.0)

    top1 = _num(cards, "pc_norm_mass_top_1", default=0.0).clip(0.0, 1.0)
    top5 = _num(cards, "pc_norm_mass_top_5", default=0.0).clip(0.0, 1.0)

    eff_dim = _num(cards, "effective_pc_dim", default=np.nan)
    entropy = _num(cards, "pc_entropy", default=np.nan)

    compact_rank = _rank01(eff_dim, higher_is_better=False)
    entropy_rank = _rank01(entropy, higher_is_better=False)

    # Coherence inside the observed PCA subspace.
    internal_coherence = (
        0.30 * top1
        + 0.25 * top5
        + 0.25 * compact_rank
        + 0.20 * entropy_rank
    ).clip(0.0, 1.0)

    # Absolute observed-mass gate.
    # If only 10% of the decoder vector is captured by the observed PCA basis,
    # we should not call the feature strongly coverage-coherent even if the
    # captured 10% is concentrated.
    observed_gate = (observed / 0.35).clip(0.0, 1.0)

    return (internal_coherence * observed_gate).clip(0.0, 1.0)


def _graph_agreement_score(cards: pd.DataFrame) -> pd.Series:
    g5 = _num(cards, "gca_at_5", 0.0).clip(0, 1)
    g10 = _num(cards, "gca_at_10", 0.0).clip(0, 1)
    g20 = _num(cards, "gca_at_20", 0.0).clip(0, 1)
    return (0.25 * g5 + 0.45 * g10 + 0.30 * g20).clip(0, 1)


def score_steering_candidates(feature_cards: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Rank SAE features as hypotheses for later causal steering validation.

    This function does *not* prove steering effectiveness. It creates an
    atlas-wide ranked table of candidate hypotheses for the next experimental
    stage.

    Important design decision:
    we return one row per feature card, not only the top-N candidates.
    The top-N selection is represented by:
    - steering_candidate_rank
    - is_steering_candidate

    This keeps feature_cards.parquet complete: every feature gets comparable
    steering / risk / graph / coverage scores, while reports can still display
    only the selected top-N rows.
    """
    if feature_cards.empty:
        return pd.DataFrame()

    cards = feature_cards.copy()

    artifact = _num(cards, "artifact_score", 0.0).clip(lower=0.0)
    token_frequency = _num(cards, "token_frequency", 0.0).clip(lower=0.0)
    p99 = _num(cards, "p99_activation", 0.0)
    max_coact = _num(cards, "max_coactivation_jaccard", 0.0).clip(0, 1)
    max_cos = _num(cards, "max_decoder_cosine", 0.0).clip(0, 1)

    labels = _text(cards, "inspection_labels")
    primary = _text(cards, "primary_label")

    is_artifact = (
        _has_label(labels, "likely_artifact")
        | primary.eq("likely_artifact")
        | (artifact >= 0.35)
    )
    is_inspection_candidate = (
        _has_label(labels, "inspection_candidate")
        | primary.eq("inspection_candidate")
    )
    is_coactivation_hub = primary.eq("coactivation_hub") | (max_coact > 0)

    artifact_penalty = (artifact / 0.35).clip(0, 1)
    generic_penalty = ((token_frequency - 0.05) / 0.15).clip(0, 1)
    activation_score = _safe_log_activation(p99)
    graph_score = _graph_agreement_score(cards)
    coverage_score = _coverage_coherence_score(cards)
    neighborhood_score = np.maximum(max_coact, 0.5 * max_cos).clip(0, 1)

    disagreement_penalty = _text(cards, "graph_alignment_bucket").eq(
        "geometry_coactivation_disagreement"
    ).astype(float)

    cards["coverage_coherence_score"] = coverage_score
    cards["graph_agreement_score"] = graph_score

    cards["atlas_steering_score"] = (
        0.26 * (1.0 - artifact_penalty)
        + 0.18 * graph_score
        + 0.16 * coverage_score
        + 0.14 * is_inspection_candidate.astype(float)
        + 0.10 * neighborhood_score
        + 0.08 * activation_score
        + 0.08 * is_coactivation_hub.astype(float)
        - 0.14 * generic_penalty
        - 0.08 * disagreement_penalty
    ).clip(-1, 1)

    cards["steering_risk_score"] = (
        0.50 * artifact_penalty
        + 0.25 * generic_penalty
        + 0.15 * disagreement_penalty
        + 0.10 * (1.0 - graph_score)
    ).clip(0, 1)

    reasons: list[str] = []
    for _, row in cards.iterrows():
        parts: list[str] = []

        if float(row.get("artifact_score", 0.0) or 0.0) < 0.10:
            parts.append("low_artifact")

        if float(row.get("graph_agreement_score", 0.0) or 0.0) >= 0.15:
            parts.append("geometry_coactivation_agreement")

        if (
            float(row.get("coverage_coherence_score", 0.0) or 0.0) >= 0.55
            and float(row.get("pc_mass_observed", 0.0) or 0.0) >= 0.35
        ):
            parts.append("coherent_residual_coverage")

        if "inspection_candidate" in str(row.get("inspection_labels", "")):
            parts.append("inspection_candidate")

        if float(row.get("max_coactivation_jaccard", 0.0) or 0.0) > 0:
            parts.append("coactivation_signal")

        if float(row.get("token_frequency", 0.0) or 0.0) >= 0.05:
            parts.append("generic_frequency_risk")

        if is_artifact.loc[row.name]:
            parts.append("artifact_risk")

        if not parts:
            parts.append("baseline_candidate")

        reasons.append(",".join(parts))

    cards["steering_candidate_reason"] = reasons

    # Rank all features, but artifacts should not become selected candidates.
    # We keep artifacts in the output with scores/risk for transparency.
    rankable = cards.loc[~is_artifact].copy()
    rankable = rankable.sort_values(
        ["atlas_steering_score", "steering_risk_score"],
        ascending=[False, True],
    )

    rank_by_feature_id = {
        int(feature_id): rank
        for rank, feature_id in enumerate(rankable["feature_id"].astype(int).tolist(), start=1)
    }

    cards["steering_candidate_rank"] = (
        cards["feature_id"]
        .astype(int)
        .map(rank_by_feature_id)
        .astype("Int64")
    )

    cards["is_steering_candidate"] = (
        cards["steering_candidate_rank"].notna()
        & (cards["steering_candidate_rank"] <= int(top_n))
    )

    # Sort output for readability
    cards = cards.sort_values(
        [
            "is_steering_candidate",
            "atlas_steering_score",
            "steering_risk_score",
        ],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return cards
