from __future__ import annotations

import pandas as pd

from sae_feature_atlas.analysis.populations import build_activation_populations
from sae_feature_atlas.config.schema import ActivationRowFilterConfig, FeatureFilterConfig


def apply_activation_row_filters(acts: pd.DataFrame, cfg: ActivationRowFilterConfig) -> pd.DataFrame:
    """Compatibility wrapper returning the explicit analysis population."""
    token_columns = ["text_id", "token_pos", "source", "token_str"]
    if "token_id" in acts.columns:
        token_columns.append("token_id")
    token_metadata = acts[token_columns].drop_duplicates(["text_id", "token_pos"])
    return build_activation_populations(acts, token_metadata, cfg).analysis_activations


def apply_feature_filters(feature_stats: pd.DataFrame, cfg: FeatureFilterConfig) -> pd.DataFrame:
    return feature_stats[
        (feature_stats["analysis_activation_count"] >= cfg.min_feature_token_count)
        & (feature_stats["analysis_text_count"] >= cfg.min_feature_text_count)
        & (feature_stats["analysis_token_frequency"] <= cfg.max_feature_token_frequency)
    ].copy()
