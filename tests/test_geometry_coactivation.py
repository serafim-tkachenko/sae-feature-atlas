from __future__ import annotations

import pandas as pd
import pytest
import torch
from types import SimpleNamespace

from sae_feature_atlas.analysis.geometry import compute_decoder_neighbors, merge_geometry_with_coactivation


def test_geometry_coactivation_match_is_orientation_invariant() -> None:
    geometry = pd.DataFrame(
        [
            {"feature_i": 8, "feature_j": 3, "rank": 1, "decoder_cosine": 0.8},
            {"feature_i": 3, "feature_j": 8, "rank": 1, "decoder_cosine": 0.8},
        ]
    )
    coactivation = pd.DataFrame(
        [
            {
                "feature_i": 3,
                "feature_j": 8,
                "coactivation_count": 12,
                "jaccard": 0.25,
                "pmi": 1.2,
                "coactivation_status": "observed_supported",
            }
        ]
    )

    merged = merge_geometry_with_coactivation(geometry, coactivation, {3, 8})

    assert merged["coactivation_count"].tolist() == [12, 12]
    assert merged["coactivation_match_status"].tolist() == [
        "observed_supported",
        "observed_supported",
    ]
    assert merged["pair_key"].nunique() == 1


def test_unmatched_edge_is_missing_not_observed_zero() -> None:
    geometry = pd.DataFrame(
        [{"feature_i": 3, "feature_j": 9, "rank": 1, "decoder_cosine": 0.4}]
    )
    merged = merge_geometry_with_coactivation(geometry, pd.DataFrame(), {3, 9})

    assert merged.loc[0, "coactivation_match_status"] == "unsupported_or_not_retained"
    assert merged.loc[0, "quadrant"] == "coactivation_unavailable"
    assert "jaccard" not in merged.columns or pd.isna(merged.loc[0, "jaccard"])


def test_neighbor_count_does_not_reserve_a_self_slot_outside_candidates():
    sae = SimpleNamespace(W_dec=torch.tensor([[1., 0.], [-1., 0.], [0., 1.]]))
    result = compute_decoder_neighbors(sae, [0], top_k=2, candidate_feature_ids=[1, 2])
    assert result.feature_j.tolist() == [2, 1]


def test_self_is_excluded_even_when_every_other_direction_is_opposite():
    sae = SimpleNamespace(W_dec=torch.tensor([[1., 0.], [-1., 0.]]))
    result = compute_decoder_neighbors(sae, [0, 1], top_k=20)
    assert result.feature_i.ne(result.feature_j).all()
    assert result.feature_j.tolist() == [1, 0]


@pytest.mark.parametrize("reverse_empirical", [False, True])
def test_directed_geometry_swaps_asymmetric_conditional_probabilities(reverse_empirical):
    geometry = pd.DataFrame([
        {"feature_i": 3, "feature_j": 8, "decoder_cosine": .5},
        {"feature_i": 8, "feature_j": 3, "decoder_cosine": .5},
    ])
    empirical = pd.DataFrame([{
        "feature_i": 8 if reverse_empirical else 3,
        "feature_j": 3 if reverse_empirical else 8,
        "p_j_given_i": .5 if reverse_empirical else .25,
        "p_i_given_j": .25 if reverse_empirical else .5,
        "jaccard": .2, "coactivation_status": "observed_supported",
    }])
    result = merge_geometry_with_coactivation(geometry, empirical, {3, 8})
    assert result.p_j_given_i.tolist() == [.25, .5]
    assert result.p_i_given_j.tolist() == [.5, .25]


def test_empty_neighbor_search_can_continue_to_geometry_coactivation_merge():
    result = compute_decoder_neighbors(SimpleNamespace(W_dec=torch.eye(2)), [], top_k=1)
    assert {"feature_i", "feature_j", "decoder_cosine"} <= set(result.columns)
    assert merge_geometry_with_coactivation(result, pd.DataFrame(), set()).empty
