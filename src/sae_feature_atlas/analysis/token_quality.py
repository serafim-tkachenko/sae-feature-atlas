from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

import pandas as pd

TOKEN_QUALITY_PREFIX = "token_quality_"

QUOTE_LIKE = {
    "'",
    "’",
    "‘",
    '"',
    "“",
    "”",
    "`",
    "``",
    "''",
}

MOJIBAKE_SUBSTRINGS = (
    "�",
    "\ufffd",
    "â",
    "€",
    "™",
    "Ã",
    "Â",
)

SPECIAL_TOKEN_RE = re.compile(r"^<[^>]+>$")


def _as_token(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _surface_for_classification(token: str) -> str:
    """Return a display token without common whitespace/subword markers.

    We keep leading whitespace as meaningful context in examples, but it should
    not make a normal word look like a formatting artifact.
    """
    surface = _as_token(token).strip()
    if len(surface) > 1 and surface[0] in {"▁", "Ġ"}:
        surface = surface[1:]
    return surface.strip()


def is_space_like_token(token: object) -> bool:
    raw = _as_token(token)
    return raw == "" or raw.isspace()


def is_quote_like_token(token: object) -> bool:
    return _surface_for_classification(_as_token(token)) in QUOTE_LIKE


def is_special_like_token(token: object) -> bool:
    surface = _surface_for_classification(_as_token(token))
    return bool(SPECIAL_TOKEN_RE.match(surface))


def is_mojibake_like_token(token: object) -> bool:
    raw = _as_token(token)
    return any(part in raw for part in MOJIBAKE_SUBSTRINGS)


def is_control_like_token(token: object) -> bool:
    raw = _as_token(token)
    if raw == "":
        return False
    return any(unicodedata.category(ch).startswith("C") for ch in raw)


def is_punctuation_or_symbol_like_token(token: object) -> bool:
    surface = _surface_for_classification(_as_token(token))
    if surface == "":
        return False
    categories = [unicodedata.category(ch) for ch in surface]
    return all(cat[0] in {"P", "S"} for cat in categories)


def token_quality_label(token: object) -> str:
    """Return the first matching artifact label for a token-like value."""
    if is_space_like_token(token):
        return "space"
    if is_special_like_token(token):
        return "special"
    if is_mojibake_like_token(token):
        return "mojibake"
    if is_control_like_token(token):
        return "control"
    if is_quote_like_token(token):
        return "quote"
    if is_punctuation_or_symbol_like_token(token):
        surface = _surface_for_classification(_as_token(token))
        if surface and all(unicodedata.category(ch).startswith("P") for ch in surface):
            return "punctuation"
        return "symbol"
    return "clean"


def attach_token_quality_flags(acts: pd.DataFrame) -> pd.DataFrame:
    """Attach token-quality flags to an SAE activation table.

    The function is intentionally row-local. Feature-level artifact decisions
    are built later from the cleaned rows and from inspection summaries.
    """
    out = acts.copy()
    if "token_str" not in out.columns:
        raise ValueError("Activation rows must contain a 'token_str' column for token-quality filtering.")

    labels = out["token_str"].map(token_quality_label)
    out["token_quality_label"] = labels
    for label in ["space", "special", "quote", "punctuation", "symbol", "mojibake", "control"]:
        out[f"{TOKEN_QUALITY_PREFIX}is_{label}"] = labels.eq(label)
    out["token_quality_is_artifact"] = labels.ne("clean")
    return out


def token_quality_keep_mask(
    acts: pd.DataFrame,
    exclude_kinds: Iterable[str] = ("space", "special", "quote", "punctuation", "symbol", "mojibake", "control"),
) -> pd.Series:
    """Return a mask keeping rows whose token-quality label is not excluded."""
    if "token_quality_label" not in acts.columns:
        acts = attach_token_quality_flags(acts)
    excluded = {str(kind) for kind in exclude_kinds}
    return ~acts["token_quality_label"].isin(excluded)


def token_quality_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col == "token_quality_label"
        or col == "token_quality_is_artifact"
        or col.startswith(TOKEN_QUALITY_PREFIX)
    ]
