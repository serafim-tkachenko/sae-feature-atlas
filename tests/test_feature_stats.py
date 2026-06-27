from __future__ import annotations

import pandas as pd

from sae_feature_atlas.analysis.feature_stats import compute_feature_stats
from sae_feature_atlas.analysis.populations import build_activation_populations
from sae_feature_atlas.config.schema import ActivationRowFilterConfig


def test_feature_stats_use_explicit_token_denominators() -> None:
    tokens = pd.DataFrame(
        [
            {"text_id": 0, "token_pos": 0, "source": "x", "token_id": 0, "token_str": "<bos>"},
            {"text_id": 0, "token_pos": 1, "source": "x", "token_id": 1, "token_str": " alpha"},
            {"text_id": 0, "token_pos": 2, "source": "x", "token_id": 2, "token_str": " beta"},
            {"text_id": 0, "token_pos": 3, "source": "x", "token_id": 3, "token_str": " gamma"},
        ]
    )
    activations = pd.DataFrame(
        [
            {"text_id": 0, "token_pos": 1, "source": "x", "token_str": " alpha", "feature_id": 7, "activation": 1.0},
            {"text_id": 0, "token_pos": 2, "source": "x", "token_str": " beta", "feature_id": 7, "activation": 2.0},
        ]
    )
    populations = build_activation_populations(
        activations, tokens, ActivationRowFilterConfig()
    )
    row = compute_feature_stats(populations, activation_mode="topk").iloc[0]

    assert row["stored_token_denominator"] == 4
    assert row["analysis_token_denominator"] == 3
    assert row["stored_token_frequency"] == 0.5
    assert row["analysis_token_frequency"] == 2 / 3
    assert row["stored_frequency_semantics"] == "retained_topk_membership_per_collected_token"
