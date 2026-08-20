from __future__ import annotations

import pandas as pd
import pytest

from sae_feature_atlas.analysis.populations import (
    build_activation_populations,
    validate_sparse_activation_rows,
)
from sae_feature_atlas.config.schema import ActivationRowFilterConfig


def _tokens() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"text_id": 0, "token_pos": 0, "source": "x", "token_id": 1, "token_str": "<bos>"},
            {"text_id": 0, "token_pos": 1, "source": "x", "token_id": 2, "token_str": " word"},
            {"text_id": 0, "token_pos": 2, "source": "x", "token_id": 3, "token_str": "'"},
            {"text_id": 0, "token_pos": 3, "source": "x", "token_id": 4, "token_str": "."},
        ]
    )


def _activations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"text_id": 0, "token_pos": 0, "source": "x", "token_str": "<bos>", "feature_id": 1, "activation": 1.0},
            {"text_id": 0, "token_pos": 1, "source": "x", "token_str": " word", "feature_id": 1, "activation": 2.0},
            {"text_id": 0, "token_pos": 2, "source": "x", "token_str": "'", "feature_id": 2, "activation": 3.0},
            {"text_id": 0, "token_pos": 3, "source": "x", "token_str": ".", "feature_id": 3, "activation": 4.0},
        ]
    )


def test_analysis_activations_are_deterministic_subset_and_stored_rows_remain() -> None:
    populations = build_activation_populations(
        _activations(), _tokens(), ActivationRowFilterConfig()
    )

    assert len(populations.all_stored_activations) == 4
    assert populations.analysis_activations[["token_pos", "feature_id"]].to_dict("records") == [
        {"token_pos": 1, "feature_id": 1}
    ]
    assert populations.analysis_tokens["token_quality_label"].tolist() == ["clean"]


def test_sparse_activation_key_must_be_unique() -> None:
    duplicated = pd.concat([_activations(), _activations().iloc[[1]]], ignore_index=True)
    with pytest.raises(ValueError, match="must be unique"):
        validate_sparse_activation_rows(duplicated)
