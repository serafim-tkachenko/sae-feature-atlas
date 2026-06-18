from __future__ import annotations

import math
from collections.abc import Iterable

import pandas as pd


def _ensure_column(df: pd.DataFrame, column: str, default) -> None:
    """Create a column if it is missing."""
    if column not in df.columns:
        df[column] = default


def _safe_numeric(series: pd.Series | None, default: float = 0.0, index=None) -> pd.Series:
    """Convert a Series to numeric and replace missing values.

    If the input series is missing, return a default-valued Series aligned to
    the provided index.
    """
    if series is None:
        return pd.Series(default, index=index)

    return pd.to_numeric(series, errors="coerce").fillna(default)


def _available_numeric(series: pd.Series | None, index) -> tuple[pd.Series, pd.Series]:
    """Return numeric values and availability mask.

    This is different from `_safe_numeric`: missing values are tracked explicitly.
    We need this because missing artifact_score must not be interpreted as
    artifact_score = 0.0.
    """
    if series is None:
        return pd.Series(0.0, index=index), pd.Series(False, index=index)

    numeric = pd.to_numeric(series, errors="coerce")
    available = numeric.notna()
    return numeric.fillna(0.0), available


def _split_labels(value) -> set[str]:
    """Parse comma-separated labels or iterable labels into a set."""
    if value is None:
        return set()

    if isinstance(value, float) and math.isnan(value):
        return set()

    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}

    if isinstance(value, Iterable):
        return {str(part).strip() for part in value if str(part).strip()}

    return set()


def _missing_or_empty(series: pd.Series) -> pd.Series:
    """Return True for missing values or empty strings."""
    return series.isna() | (series.astype(str).str.strip() == "")


def _quantile_threshold(series: pd.Series, q: float, default: float = float("inf")) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.quantile(q))


def _positive_quantile_threshold(
    series: pd.Series,
    q: float,
    default: float = float("inf"),
) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return default
    return float(values.quantile(q))


def primary_label(label_string: str) -> str:
    """Return the most useful primary label from a comma-separated label string.

    These are triage labels, not final semantic interpretations.
    """
    labels = _split_labels(label_string)

    priority = [
        "likely_artifact",
        "rare_high_intensity",
        "coactivation_hub",
        "decoder_geometry_dense",
        "bimodal_candidate",
        "high_intensity",
        "high_frequency",
        "manual_review",
        "rare_feature"]

    for label in priority:
        if label in labels:
            return label

    return "unlabeled"


def assign_feature_labels(cards: pd.DataFrame) -> pd.DataFrame:
    """Assign conservative heuristic labels to feature cards.

    Important rule:
    we should not call a feature semantic just because it is not an obvious
    artifact. `manual_review` only means "worth manual inspection",
    not "semantic feature".
    """
    out = cards.copy()

    _ensure_column(out, "inspection_labels", "")
    _ensure_column(out, "manual_priority", pd.NA)

    artifact_score, artifact_available = _available_numeric(
        out.get("artifact_score"),
        index=out.index,
    )

    token_frequency = _safe_numeric(out.get("token_frequency"), 0.0, index=out.index)
    p99_activation = _safe_numeric(out.get("p99_activation"), 0.0, index=out.index)
    n_token_activations = _safe_numeric(out.get("n_token_activations"), 0.0, index=out.index)
    bimodality_score = _safe_numeric(out.get("bimodality_score"), 0.0, index=out.index)
    max_decoder_cosine = _safe_numeric(out.get("max_decoder_cosine"), 0.0, index=out.index)
    max_coactivation_jaccard = _safe_numeric(
        out.get("max_coactivation_jaccard"),
        0.0,
        index=out.index,
    )
    semantic_score = _safe_numeric(out.get("semantic_score"), 0.0, index=out.index)

    existing_labels = [_split_labels(v) for v in out["inspection_labels"].tolist()]

    def add_label(mask: pd.Series, label: str) -> None:
        mask = mask.reindex(out.index, fill_value=False).fillna(False).astype(bool)
        for idx, active in enumerate(mask.tolist()):
            if active:
                existing_labels[idx].add(label)

    likely_artifact = artifact_available & (artifact_score >= 0.35)
    add_label(likely_artifact, "likely_artifact")

    high_frequency_threshold = max(
        0.03,
        _quantile_threshold(token_frequency, 0.90, default=0.03),
    )
    high_frequency = token_frequency >= high_frequency_threshold
    add_label(high_frequency, "high_frequency")

    positive_frequency = token_frequency[token_frequency > 0]
    if positive_frequency.empty:
        rare_feature = pd.Series(False, index=out.index)
    else:
        rare_threshold = positive_frequency.quantile(0.20)
        rare_feature = (token_frequency > 0) & (token_frequency <= rare_threshold)
    add_label(rare_feature, "rare_feature")

    high_intensity_threshold = _quantile_threshold(
        p99_activation,
        0.90,
        default=float("inf"),
    )
    high_intensity = p99_activation >= high_intensity_threshold
    add_label(high_intensity, "high_intensity")

    rare_high_intensity = rare_feature & high_intensity
    add_label(rare_high_intensity, "rare_high_intensity")

    bimodality_threshold = max(
        500.0,
        _positive_quantile_threshold(bimodality_score, 0.95, default=float("inf")),
    )
    bimodal_candidate = bimodality_score >= bimodality_threshold
    add_label(bimodal_candidate, "bimodal_candidate")

    positive_decoder = max_decoder_cosine[max_decoder_cosine > 0]
    if positive_decoder.empty:
        decoder_dense = pd.Series(False, index=out.index)
    else:
        decoder_threshold = positive_decoder.quantile(0.90)
        decoder_dense = max_decoder_cosine >= decoder_threshold
    add_label(decoder_dense, "decoder_geometry_dense")

    positive_coactivation = max_coactivation_jaccard[max_coactivation_jaccard > 0]
    if positive_coactivation.empty:
        coactivation_hub = pd.Series(False, index=out.index)
    else:
        coactivation_threshold = positive_coactivation.quantile(0.90)
        coactivation_hub = max_coactivation_jaccard >= coactivation_threshold
    add_label(coactivation_hub, "coactivation_hub")

    # Strict global manual-review label. This intentionally duplicates the
    # inspection-layer logic so old broad labels from previous runs do not leak.
    strict_manual_review = (
        artifact_available
        & ~likely_artifact
        & (n_token_activations >= 100)
        & (artifact_score < 0.10)
        & (semantic_score >= 0.95)
    )
    add_label(strict_manual_review, "manual_review")

    cleaned_labels = []
    for labels in existing_labels:
        labels.discard("semantic_candidate")
        labels.discard("source_concentrated")

        # If manual_review was added by an older, looser inspection run,
        # keep it only if the strict global condition still holds.
        labels.discard("manual_review")

        cleaned_labels.append(labels)

    out["inspection_labels"] = [",".join(sorted(labels)) for labels in cleaned_labels]

    # Re-add strict manual-review features after cleanup.
    existing_labels = [_split_labels(v) for v in out["inspection_labels"].tolist()]
    for idx, active in enumerate(strict_manual_review.tolist()):
        if active:
            existing_labels[idx].add("manual_review")

    out["inspection_labels"] = [",".join(sorted(labels)) for labels in existing_labels]
    out["primary_label"] = out["inspection_labels"].map(primary_label)

    missing_priority = _missing_or_empty(out["manual_priority"])

    high_priority = (
        out["primary_label"].isin(
            [
                "rare_high_intensity",
                "bimodal_candidate",
                "coactivation_hub",
                "decoder_geometry_dense"]
        )
        & ~likely_artifact
    )

    # manual_review is useful, but not automatically "high" unless it is
    # also selected by another scientific signal.
    medium_priority = (
        out["primary_label"].isin(
            [
                "manual_review",
                "high_frequency",
                "high_intensity",
                "rare_feature"]
        )
        & ~likely_artifact
    )

    low_priority = likely_artifact
    unreviewed = out["primary_label"].eq("unlabeled")

    out.loc[missing_priority & high_priority, "manual_priority"] = "high"
    out.loc[missing_priority & medium_priority, "manual_priority"] = "medium"
    out.loc[missing_priority & low_priority, "manual_priority"] = "low"
    out.loc[missing_priority & unreviewed, "manual_priority"] = "unreviewed"

    # Recompute stale priorities from previous broader runs
    out.loc[out["primary_label"].eq("manual_review"), "manual_priority"] = "medium"
    out.loc[likely_artifact, "manual_priority"] = "low"

    out["manual_priority"] = out["manual_priority"].fillna("unreviewed")

    return out