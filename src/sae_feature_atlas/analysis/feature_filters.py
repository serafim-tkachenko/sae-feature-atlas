from __future__ import annotations

import numpy as np
import pandas as pd

from sae_feature_atlas.config.schema import ActivationRowFilterConfig, FeatureFilterConfig


def apply_activation_row_filters(acts: pd.DataFrame, cfg: ActivationRowFilterConfig) -> pd.DataFrame:
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
    if cfg.min_activation is not None:
        filtered = filtered[filtered["activation"] >= cfg.min_activation]
    return filtered.copy()


def apply_feature_filters(feature_stats: pd.DataFrame, cfg: FeatureFilterConfig) -> pd.DataFrame:
    return feature_stats[
        (feature_stats["n_token_activations"] >= cfg.min_feature_token_count)
        & (feature_stats["n_texts"] >= cfg.min_feature_text_count)
        & (feature_stats["token_frequency"] <= cfg.max_feature_token_frequency)
    ].copy()
