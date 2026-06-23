from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sae_feature_atlas.analysis.token_quality import attach_token_quality_flags, token_quality_keep_mask
from sae_feature_atlas.config.schema import ActivationRowFilterConfig


TOKEN_KEY = ["text_id", "token_pos"]
SPARSE_ACTIVATION_KEY = ["text_id", "token_pos", "feature_id"]


@dataclass(frozen=True)
class ActivationPopulations:
    """Named populations used by scientific analyses.

    all_stored_activations contains every sparse row persisted by collection.
    In top-k mode it is rank-censored stored activity, not all positive SAE activity.
    analysis_activations is the deterministic subset whose target token passes
    the configured analysis eligibility policy.
    """

    all_stored_activations: pd.DataFrame
    analysis_activations: pd.DataFrame
    stored_tokens: pd.DataFrame
    analysis_tokens: pd.DataFrame


def validate_sparse_activation_rows(acts: pd.DataFrame) -> None:
    missing = set(SPARSE_ACTIVATION_KEY + ["activation"]) - set(acts.columns)
    if missing:
        raise ValueError(f"Sparse activation table is missing columns: {sorted(missing)}")
    duplicates = acts.duplicated(SPARSE_ACTIVATION_KEY, keep=False)
    if duplicates.any():
        examples = acts.loc[duplicates, SPARSE_ACTIVATION_KEY].head(5).to_dict("records")
        raise ValueError(
            "Sparse activation rows must be unique by "
            f"{tuple(SPARSE_ACTIVATION_KEY)}; examples={examples}"
        )


def build_analysis_token_population(
    token_metadata: pd.DataFrame,
    cfg: ActivationRowFilterConfig,
) -> pd.DataFrame:
    """Return the explicit token universe eligible for semantic analysis."""
    missing = set(TOKEN_KEY + ["source", "token_str"]) - set(token_metadata.columns)
    if missing:
        raise ValueError(f"Token metadata is missing columns: {sorted(missing)}")
    if token_metadata.duplicated(TOKEN_KEY).any():
        raise ValueError(f"Token metadata must be unique by {tuple(TOKEN_KEY)}")

    eligible = token_metadata.copy()
    if cfg.include_sources is not None:
        eligible = eligible[eligible["source"].isin(cfg.include_sources)]
    if cfg.exclude_sources:
        eligible = eligible[~eligible["source"].isin(cfg.exclude_sources)]
    if cfg.exclude_token_positions:
        eligible = eligible[~eligible["token_pos"].isin(cfg.exclude_token_positions)]
    if cfg.exclude_token_positions_ge is not None:
        eligible = eligible[eligible["token_pos"] < cfg.exclude_token_positions_ge]
    if cfg.exclude_token_strings:
        eligible = eligible[~eligible["token_str"].isin(cfg.exclude_token_strings)]
    if cfg.exclude_token_substrings:
        excluded = pd.Series(False, index=eligible.index)
        for substring in cfg.exclude_token_substrings:
            excluded |= eligible["token_str"].astype(str).str.contains(
                substring, regex=False, na=False
            )
        eligible = eligible[~excluded]

    eligible = attach_token_quality_flags(eligible)
    if cfg.exclude_token_quality_kinds:
        eligible = eligible[token_quality_keep_mask(eligible, cfg.exclude_token_quality_kinds)]
    return eligible.copy()


def build_activation_populations(
    all_stored_activations: pd.DataFrame,
    token_metadata: pd.DataFrame,
    cfg: ActivationRowFilterConfig,
) -> ActivationPopulations:
    """Construct stored and analysis activation/token populations explicitly."""
    validate_sparse_activation_rows(all_stored_activations)
    stored_tokens = token_metadata.drop_duplicates(TOKEN_KEY).copy()
    analysis_tokens = build_analysis_token_population(stored_tokens, cfg)

    quality_columns = [
        column
        for column in analysis_tokens.columns
        if column == "token_quality_label" or column.startswith("token_quality_")
    ]
    analysis = all_stored_activations.merge(
        analysis_tokens[TOKEN_KEY + quality_columns],
        on=TOKEN_KEY,
        how="inner",
        validate="many_to_one",
    )
    if cfg.require_finite_activation:
        analysis = analysis[np.isfinite(analysis["activation"])]
    if cfg.min_activation is not None:
        analysis = analysis[analysis["activation"] >= cfg.min_activation]
    if not cfg.keep_token_quality_columns:
        analysis = analysis.drop(columns=quality_columns, errors="ignore")

    return ActivationPopulations(
        all_stored_activations=all_stored_activations.copy(),
        analysis_activations=analysis.copy(),
        stored_tokens=stored_tokens,
        analysis_tokens=analysis_tokens.copy(),
    )
