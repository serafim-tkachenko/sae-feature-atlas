from __future__ import annotations

import numpy as np
import pandas as pd

from sae_feature_atlas.analysis.token_quality import (
    attach_token_quality_flags,
    token_quality_columns,
    token_quality_keep_mask,
)
from sae_feature_atlas.config.schema import ActivationRowFilterConfig, FeatureFilterConfig


def apply_activation_row_filters(acts: pd.DataFrame, cfg: ActivationRowFilterConfig) -> pd.DataFrame:
    """Apply row-level filters before feature statistics and downstream analysis"""
    filtered = acts.copy()

    if cfg.require_finite_activation:
        filtered = filtered[np.isfinite(filtered["activation"])]
    if cfg.include_sources is not None:
        filtered = filtered[filtered["source"].isin(cfg.include_sources)]
    if cfg.exclude_sources:
        filtered = filtered[~filtered["source"].isin(cfg.exclude_sources)]
    if cfg.exclude_token_positions:
        filtered = filtered[~filtered["token_pos"].isin(cfg.exclude_token_positions)]
    if cfg.exclude_token_positions_ge is not None:
        filtered = filtered[filtered["token_pos"] < cfg.exclude_token_positions_ge]
    if cfg.exclude_token_strings:
        filtered = filtered[~filtered["token_str"].isin(cfg.exclude_token_strings)]
    if cfg.exclude_token_substrings:
        mask = pd.Series(False, index=filtered.index)
        for substring in cfg.exclude_token_substrings:
            mask = mask | filtered["token_str"].astype(str).str.contains(substring, regex=False, na=False)
        filtered = filtered[~mask]

    if getattr(cfg, "exclude_token_quality_kinds", ()):
        filtered = attach_token_quality_flags(filtered)
        filtered = filtered[token_quality_keep_mask(filtered, cfg.exclude_token_quality_kinds)]
        if not getattr(cfg, "keep_token_quality_columns", True):
            filtered = filtered.drop(columns=token_quality_columns(filtered), errors="ignore")

    if cfg.min_activation is not None:
        filtered = filtered[filtered["activation"] >= cfg.min_activation]

    return filtered.copy()


def apply_feature_filters(feature_stats: pd.DataFrame, cfg: FeatureFilterConfig) -> pd.DataFrame:
    return feature_stats[
        (feature_stats["n_token_activations"] >= cfg.min_feature_token_count)
        & (feature_stats["n_texts"] >= cfg.min_feature_text_count)
        & (feature_stats["token_frequency"] <= cfg.max_feature_token_frequency)
    ].copy()
