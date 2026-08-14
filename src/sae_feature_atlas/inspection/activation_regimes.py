
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def _fit_two_component_log_gmm(
    values: np.ndarray,
    *,
    random_seed: int = 0,
    n_init: int = 5,
) -> GaussianMixture | None:
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) < 2:
        return None
    x = np.log1p(values).reshape(-1, 1)
    try:
        return GaussianMixture(
            n_components=2,
            random_state=random_seed,
            n_init=n_init,
        ).fit(x)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return None


def _context_rows(
    feature_rows: pd.DataFrame,
    renderer,
    context_window: int,
) -> pd.DataFrame:
    """Attach shared raw evidence and decoded display context."""
    rows: list[dict] = []
    for row in feature_rows.to_dict("records"):
        context = renderer.render(
            text_id=int(row["text_id"]),
            token_pos=int(row["token_pos"]),
            context_window=context_window,
        )
        rows.append({**row, **context, "activation_population": "analysis_activations"})
    return pd.DataFrame(rows)


def build_bimodal_peak_examples(
    acts: pd.DataFrame,
    renderer,
    candidates: pd.DataFrame,
    *,
    top_features: int = 50,
    examples_per_peak: int = 8,
    context_window: int = 20,
    random_seed: int = 0,
    n_init: int = 5,
) -> pd.DataFrame:
    """Build low/high activation-regime examples for ranked bimodal features.

    The output supplies representative, posterior-confident context evidence
    for qualified statistical candidates. It does not establish two semantic
    concepts or final regime membership.
    """
    if acts.empty or candidates.empty:
        return pd.DataFrame()

    needed = {"feature_id", "text_id", "token_pos", "activation"}
    missing = needed - set(acts.columns)
    if missing:
        raise ValueError(f"Activation table is missing columns needed for regime examples: {sorted(missing)}")

    ranked_ids = candidates.head(top_features)["feature_id"].astype(int).tolist()
    bimodality_score_by_feature = dict(
        zip(
            candidates["feature_id"].astype(int),
            candidates.get("bimodality_score", pd.Series([np.nan] * len(candidates))),
        )
    )
    rows: list[pd.DataFrame] = []

    for feature_id in tqdm(ranked_ids, desc="Bimodal low/high examples"):
        feature_rows = acts[acts["feature_id"].astype(int).eq(feature_id)].copy()
        if feature_rows.empty:
            continue
        values = feature_rows["activation"].to_numpy(dtype=np.float64)
        gmm = _fit_two_component_log_gmm(values, random_seed=random_seed, n_init=n_init)
        if gmm is None:
            continue

        x = np.log1p(feature_rows["activation"].to_numpy(dtype=np.float64)).reshape(-1, 1)
        probabilities = gmm.predict_proba(x)
        labels = probabilities.argmax(axis=1)
        means = gmm.means_.reshape(-1)
        low_component = int(np.argmin(means))
        high_component = int(np.argmax(means))
        feature_rows["log_activation"] = x.reshape(-1)
        feature_rows["peak_label"] = np.where(labels == low_component, "low", "high")
        feature_rows["component_posterior"] = probabilities.max(axis=1)
        assigned_means = np.where(labels == low_component, means[low_component], means[high_component])
        feature_rows["distance_to_component_log_mean"] = np.abs(x.reshape(-1) - assigned_means)
        feature_rows["peak_log_mean_low"] = float(means[low_component])
        feature_rows["peak_log_mean_high"] = float(means[high_component])
        feature_rows["bimodality_score"] = float(bimodality_score_by_feature.get(feature_id, np.nan))

        example_sort = ["component_posterior", "distance_to_component_log_mean"]
        low = (
            feature_rows[feature_rows["peak_label"].eq("low")]
            .sort_values(example_sort, ascending=[False, True])
            .head(examples_per_peak)
        )
        high = (
            feature_rows[feature_rows["peak_label"].eq("high")]
            .sort_values(example_sort, ascending=[False, True])
            .head(examples_per_peak)
        )
        examples = pd.concat([low, high], ignore_index=True)
        if examples.empty:
            continue
        rows.append(_context_rows(examples, renderer, context_window=context_window))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = [
        "feature_id",
        "peak_label",
        "activation",
        "log_activation",
        "component_posterior",
        "distance_to_component_log_mean",
        "bimodality_score",
        "peak_log_mean_low",
        "peak_log_mean_high",
        "text_id",
        "token_pos",
        "source",
        "context_start_pos",
        "context_end_pos",
        "context_token_ids_json",
        "raw_token_strings_json",
        "target_token_id",
        "target_token_str",
        "target_quality",
        "display_quality",
        "display_context",
        "left_context",
        "center_token",
        "right_context",
        "activation_population",
    ]
    keep = [col for col in keep if col in out.columns]
    return out[keep].sort_values(["bimodality_score", "feature_id", "peak_label", "activation"], ascending=[False, True, True, False]).reset_index(drop=True)


def peak_examples_json(examples: pd.DataFrame, feature_id: int, peak_label: str, n: int = 5) -> str:
    if examples.empty:
        return "[]"
    subset = examples[
        examples["feature_id"].astype(int).eq(int(feature_id)) & examples["peak_label"].astype(str).eq(peak_label)
    ]
    cols = [
        "activation",
        "source",
        "text_id",
        "token_pos",
        "left_context",
        "center_token",
        "right_context"]
    cols = [col for col in cols if col in subset.columns]
    records = subset.sort_values("activation", ascending=False).head(n)[cols].to_dict("records")
    return json.dumps(records, ensure_ascii=False)
