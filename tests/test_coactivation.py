from __future__ import annotations

import math

import pandas as pd

from sae_feature_atlas.analysis.coactivation import compute_same_token_coactivation


def _eligible_tokens() -> pd.DataFrame:
    return pd.DataFrame(
        [{"text_id": 0, "token_pos": pos} for pos in range(4)]
    )


def _activations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"text_id": 0, "token_pos": 0, "feature_id": 1},
            {"text_id": 0, "token_pos": 0, "feature_id": 2},
            {"text_id": 0, "token_pos": 1, "feature_id": 1},
            {"text_id": 0, "token_pos": 2, "feature_id": 2},
        ]
    ).assign(activation=1.0)


def test_coactivation_metrics_use_complete_eligible_token_universe() -> None:
    result = compute_same_token_coactivation(
        _activations(),
        {1, 2},
        _eligible_tokens(),
        min_pair_support=1,
        neighbors_per_feature=5,
    )
    row = result.pairs.iloc[0]

    assert row["coactivation_count"] == 1
    assert row["jaccard"] == 1 / 3
    assert row["p_j_given_i"] == 0.5
    assert row["p_i_given_j"] == 0.5
    assert row["pmi"] == math.log((1 / 4) / ((2 / 4) * (2 / 4)))
    assert row["eligible_token_count"] == 4


def test_duplicate_rows_are_canonicalized_consistently() -> None:
    duplicated = pd.concat([_activations(), _activations().iloc[[0]]], ignore_index=True)
    result = compute_same_token_coactivation(
        duplicated,
        {1, 2},
        _eligible_tokens(),
        min_pair_support=1,
        neighbors_per_feature=5,
    )

    assert result.metadata["duplicate_rows_removed"] == 1
    assert result.pairs.iloc[0]["feature_i_count"] == 2
    assert result.pairs.iloc[0]["coactivation_count"] == 1


def test_minimum_support_filters_rare_pairs() -> None:
    result = compute_same_token_coactivation(
        _activations(),
        {1, 2},
        _eligible_tokens(),
        min_pair_support=2,
        neighbors_per_feature=5,
    )

    assert result.pairs.empty
    assert result.metadata["minimum_pair_support"] == 2
