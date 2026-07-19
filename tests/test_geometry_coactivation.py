from __future__ import annotations

import pandas as pd

from sae_feature_atlas.analysis.geometry import merge_geometry_with_coactivation


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
