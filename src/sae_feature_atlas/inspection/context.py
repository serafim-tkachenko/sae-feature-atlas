from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable

import pandas as pd

from sae_feature_atlas.analysis.token_quality import is_mojibake_like_token, token_quality_label


class ContextRenderer:
    """Render reproducible raw evidence and decoded human-facing context."""

    def __init__(
        self,
        token_metadata: pd.DataFrame,
        decode_token_ids: Callable[[list[int]], str],
    ) -> None:
        required = {"text_id", "token_pos", "token_id", "token_str"}
        missing = required - set(token_metadata.columns)
        if missing:
            raise ValueError(f"Token metadata is missing context columns: {sorted(missing)}")
        self._decode = decode_token_ids
        self._tokens_by_text = {
            int(text_id): group.sort_values("token_pos").reset_index(drop=True)
            for text_id, group in token_metadata.groupby("text_id")
        }

    @staticmethod
    def display_quality(text: str) -> str:
        if is_mojibake_like_token(text):
            return "mojibake"
        if any(unicodedata.category(char).startswith("C") and not char.isspace() for char in text):
            return "control"
        return "readable"

    def render(self, text_id: int, token_pos: int, context_window: int) -> dict:
        text_id = int(text_id)
        token_pos = int(token_pos)
        tokens = self._tokens_by_text.get(text_id)
        if tokens is None:
            raise KeyError(f"Missing token metadata for text_id={text_id}")
        positions = tokens["token_pos"].astype(int)
        context = tokens[
            (positions >= token_pos - context_window)
            & (positions <= token_pos + context_window)
        ]
        target = tokens[positions.eq(token_pos)]
        if target.empty:
            raise KeyError(f"Missing target token text_id={text_id}, token_pos={token_pos}")

        target_row = target.iloc[0]
        left = context[context["token_pos"].astype(int) < token_pos]
        right = context[context["token_pos"].astype(int) > token_pos]
        context_ids = context["token_id"].astype(int).tolist()
        left_ids = left["token_id"].astype(int).tolist()
        right_ids = right["token_id"].astype(int).tolist()
        target_id = int(target_row["token_id"])
        display = self._decode(context_ids)
        return {
            "context_start_pos": int(context["token_pos"].min()),
            "context_end_pos": int(context["token_pos"].max()) + 1,
            "context_token_ids_json": json.dumps(context_ids),
            "raw_token_strings_json": json.dumps(
                context["token_str"].astype(str).tolist(), ensure_ascii=False
            ),
            "target_token_id": target_id,
            "target_token_str": str(target_row["token_str"]),
            "target_quality": token_quality_label(target_row["token_str"]),
            "display_quality": self.display_quality(display),
            "display_context": display,
            "left_context": self._decode(left_ids) if left_ids else "",
            "center_token": self._decode([target_id]),
            "right_context": self._decode(right_ids) if right_ids else "",
        }
