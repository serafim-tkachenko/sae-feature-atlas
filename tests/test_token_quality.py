from __future__ import annotations

import pandas as pd

from sae_feature_atlas.analysis.feature_filters import apply_activation_row_filters
from sae_feature_atlas.analysis.token_quality import token_quality_label
from sae_feature_atlas.config.schema import ActivationRowFilterConfig


def test_token_quality_labels_common_artifacts() -> None:
    assert token_quality_label(" ") == "space"
    assert token_quality_label("\n") == "space"
    assert token_quality_label("'") == "quote"
    assert token_quality_label("...") == "punctuation"
    assert token_quality_label("★") == "symbol"
    assert token_quality_label("<bos>") == "special"
    assert token_quality_label(" hello") == "clean"
    assert token_quality_label("▁hello") == "clean"


def test_activation_row_filter_drops_token_artifacts() -> None:
    acts = pd.DataFrame(
        [
            {"feature_id": 1, "text_id": 0, "token_pos": 0, "source": "x", "token_str": "<bos>", "activation": 1.0},
            {"feature_id": 1, "text_id": 0, "token_pos": 1, "source": "x", "token_str": " hello", "activation": 2.0},
            {"feature_id": 2, "text_id": 0, "token_pos": 2, "source": "x", "token_str": ".", "activation": 3.0},
            {"feature_id": 3, "text_id": 0, "token_pos": 3, "source": "x", "token_str": "★", "activation": 4.0},
        ]
    )

    filtered = apply_activation_row_filters(acts, ActivationRowFilterConfig())

    assert filtered["token_str"].tolist() == [" hello"]
    assert filtered["token_quality_label"].tolist() == ["clean"]
