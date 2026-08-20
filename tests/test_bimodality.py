from __future__ import annotations

import numpy as np
import pandas as pd

from sae_feature_atlas.analysis.bimodality import compute_bimodality


def _mixture_rows() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    log_values = np.concatenate(
        [rng.normal(1.0, 0.08, 80), rng.normal(3.0, 0.10, 80)]
    )
    values = np.expm1(log_values)
    return pd.DataFrame(
        {
            "feature_id": 12,
            "text_id": np.arange(len(values)),
            "token_pos": 1,
            "activation": values,
        }
    )


def test_bimodality_records_diagnostics_and_candidate_status() -> None:
    result = compute_bimodality(
        _mixture_rows(),
        {12},
        min_points=40,
        random_seed=3,
        n_init=4,
        storage_mode="positive",
    )
    row = result.evaluated_features.iloc[0]

    assert row["fit_status"] == "ok"
    assert bool(row["converged"])
    assert row["log_mean_low"] < row["log_mean_high"]
    assert row["component_weight_low"] > 0.1
    assert row["component_weight_high"] > 0.1
    assert row["mode_separation"] > 2
    assert bool(row["is_bimodal_candidate"])
    assert len(result.candidates) == 1
    assert not bool(row["rank_censored"])


def test_bimodality_is_deterministic_and_separates_evaluated_from_candidates() -> None:
    first = compute_bimodality(_mixture_rows(), {12, 99}, min_points=40, n_init=3)
    second = compute_bimodality(_mixture_rows(), {12, 99}, min_points=40, n_init=3)

    columns = ["feature_id", "fit_status", "delta_bic", "log_mean_low", "log_mean_high"]
    pd.testing.assert_frame_equal(
        first.evaluated_features[columns],
        second.evaluated_features[columns],
    )
    insufficient = first.evaluated_features.set_index("feature_id").loc[99]
    assert insufficient["fit_status"] == "insufficient_points"
    assert 99 not in set(first.candidates["feature_id"])
