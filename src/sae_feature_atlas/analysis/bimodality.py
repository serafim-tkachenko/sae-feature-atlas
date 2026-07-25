from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


@dataclass(frozen=True)
class BimodalityResult:
    evaluated_features: pd.DataFrame
    candidates: pd.DataFrame


def _ordered_component_diagnostics(gmm: GaussianMixture) -> dict:
    means = gmm.means_.reshape(-1)
    variances = gmm.covariances_.reshape(-1)
    weights = gmm.weights_.reshape(-1)
    order = np.argsort(means)
    low, high = int(order[0]), int(order[1])
    pooled_scale = np.sqrt(0.5 * (variances[low] + variances[high]))
    separation = (
        float((means[high] - means[low]) / pooled_scale)
        if pooled_scale > 0
        else float("inf")
    )
    return {
        "log_mean_low": float(means[low]),
        "log_mean_high": float(means[high]),
        "log_variance_low": float(variances[low]),
        "log_variance_high": float(variances[high]),
        "log_std_low": float(np.sqrt(variances[low])),
        "log_std_high": float(np.sqrt(variances[high])),
        "component_weight_low": float(weights[low]),
        "component_weight_high": float(weights[high]),
        "minimum_component_weight": float(weights.min()),
        "mode_separation": separation,
    }


def compute_bimodality(
    analysis_activations: pd.DataFrame,
    analysis_feature_ids: set[int],
    *,
    min_points: int = 100,
    delta_bic_threshold: float = 10.0,
    min_component_weight: float = 0.10,
    min_separation: float = 2.0,
    random_seed: int = 0,
    n_init: int = 5,
    storage_mode: str = "topk",
) -> BimodalityResult:
    """Evaluate qualified analysis features for log-activation bimodality.

    Under top-k storage these are rank-censored activation observations. A
    candidate is a triage result satisfying explicit evidence and component
    quality thresholds, not proof of two semantic concepts.
    """
    rows: list[dict] = []
    selected = analysis_activations[
        analysis_activations["feature_id"].isin(analysis_feature_ids)
    ]
    grouped = {int(fid): group for fid, group in selected.groupby("feature_id")}

    for feature_id in tqdm(sorted(analysis_feature_ids), desc="Bimodality"):
        group = grouped.get(feature_id)
        values = (
            np.array([], dtype=np.float64)
            if group is None
            else group["activation"].to_numpy(dtype=np.float64)
        )
        values = values[np.isfinite(values) & (values > 0)]
        base = {
            "feature_id": int(feature_id),
            "n_points": int(len(values)),
            "fit_status": "insufficient_points",
            "converged": False,
            "is_bimodal_candidate": False,
            "gmm_random_seed": int(random_seed),
            "gmm_n_init": int(n_init),
            "activation_population": "analysis_activations",
            "activation_storage_mode": storage_mode,
            "rank_censored": storage_mode == "topk",
        }
        if len(values) < min_points:
            rows.append(base)
            continue

        x = np.log1p(values).reshape(-1, 1)
        try:
            gmm1 = GaussianMixture(
                n_components=1,
                random_state=random_seed,
                n_init=n_init,
            ).fit(x)
            gmm2 = GaussianMixture(
                n_components=2,
                random_state=random_seed,
                n_init=n_init,
            ).fit(x)
            bic1 = float(gmm1.bic(x))
            bic2 = float(gmm2.bic(x))
            delta_bic = bic1 - bic2
            diagnostics = _ordered_component_diagnostics(gmm2)
            converged = bool(gmm1.converged_ and gmm2.converged_)
            candidate = bool(
                converged
                and delta_bic >= delta_bic_threshold
                and diagnostics["minimum_component_weight"] >= min_component_weight
                and diagnostics["mode_separation"] >= min_separation
            )
            rows.append(
                {
                    **base,
                    **diagnostics,
                    "fit_status": "ok" if converged else "not_converged",
                    "converged": converged,
                    "n_iter_1": int(gmm1.n_iter_),
                    "n_iter_2": int(gmm2.n_iter_),
                    "bic_1": bic1,
                    "bic_2": bic2,
                    "delta_bic": delta_bic,
                    "bimodality_score": delta_bic,
                    "is_bimodal_candidate": candidate,
                    "activation_min": float(values.min()),
                    "activation_p50": float(np.quantile(values, 0.50)),
                    "activation_p95": float(np.quantile(values, 0.95)),
                    "activation_max": float(values.max()),
                    "candidate_delta_bic_threshold": float(delta_bic_threshold),
                    "candidate_min_component_weight": float(min_component_weight),
                    "candidate_min_separation": float(min_separation),
                }
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rows.append({**base, "fit_status": f"fit_error:{type(exc).__name__}"})

    evaluated = pd.DataFrame(rows)
    if evaluated.empty:
        return BimodalityResult(evaluated, evaluated.copy())
    candidates = evaluated[evaluated["is_bimodal_candidate"].fillna(False)].copy()
    candidates = candidates.sort_values("delta_bic", ascending=False).reset_index(drop=True)
    return BimodalityResult(evaluated.reset_index(drop=True), candidates)


def compute_bimodality_candidates(
    acts: pd.DataFrame,
    min_points: int = 100,
) -> pd.DataFrame:
    """Compatibility wrapper returning only threshold-qualified candidates."""
    ids = set(acts["feature_id"].astype(int).unique())
    return compute_bimodality(acts, ids, min_points=min_points).candidates
