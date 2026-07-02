from __future__ import annotations

import json

import pandas as pd

from sae_feature_atlas.analysis.feature_stats import build_top_examples
from sae_feature_atlas.inspection.context import ContextRenderer


TOKEN_TEXT = {1: "<bos>", 2: "He", 3: " said", 4: ",", 5: " hello", 6: "!", 7: "'"}


def _decode(ids: list[int]) -> str:
    return "".join(TOKEN_TEXT[token_id] for token_id in ids)


def _metadata() -> pd.DataFrame:
    strings = ["<bos>", "He", " said", ",", " hello", "!", "'"]
    return pd.DataFrame(
        [
            {
                "text_id": 0,
                "token_pos": pos,
                "source": "x",
                "token_id": pos + 1,
                "token_str": token,
            }
            for pos, token in enumerate(strings)
        ]
    )


def test_renderer_decodes_contiguous_ids_and_retains_raw_evidence() -> None:
    rendered = ContextRenderer(_metadata(), _decode).render(0, 4, context_window=2)

    assert rendered["display_context"] == " said, hello!'"
    assert rendered["left_context"] == " said,"
    assert rendered["center_token"] == " hello"
    assert rendered["right_context"] == "!'"
    assert json.loads(rendered["context_token_ids_json"]) == [3, 4, 5, 6, 7]
    assert rendered["target_quality"] == "clean"
    assert rendered["display_quality"] == "readable"


def test_clean_target_survives_punctuation_rich_context() -> None:
    acts = pd.DataFrame(
        [
            {
                "text_id": 0,
                "token_pos": 4,
                "source": "x",
                "token_str": " hello",
                "feature_id": 9,
                "activation": 4.0,
            }
        ]
    )
    examples = build_top_examples(acts, ContextRenderer(_metadata(), _decode), context_window=2)

    assert examples.loc[0, "target_quality"] == "clean"
    assert examples.loc[0, "center_token"] == " hello"
    assert "," in examples.loc[0, "display_context"]
    assert "!" in examples.loc[0, "display_context"]
