from __future__ import annotations

import json

import pandas as pd

from sae_feature_atlas.config import ExperimentConfig
from sae_feature_atlas.feature_stats import build_top_examples, compute_feature_stats
from sae_feature_atlas.filters import apply_activation_row_filters, apply_feature_filters


def _top_examples_json(df: pd.DataFrame, n: int = 5) -> str:
    cols = [
        "rank",
        "activation",
        "source",
        "text_id",
        "token_pos",
        "left_context",
        "center_token",
        "right_context",
    ]
    examples = df.sort_values("activation", ascending=False).head(n)[cols].to_dict("records")
    return json.dumps(examples, ensure_ascii=False)


def build_feature_cards(
    filtered_features: pd.DataFrame,
    top_examples: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    rows: list[dict] = []
    for feature_id, group in top_examples.groupby("feature_id"):
        rows.append({"feature_id": int(feature_id), "top_examples_json": _top_examples_json(group)})

    top_examples_small = pd.DataFrame(rows)
    feature_cards = filtered_features.merge(top_examples_small, on="feature_id", how="left")

    feature_cards["model_name"] = cfg.model.model_name
    feature_cards["sae_release"] = cfg.model.sae_release
    feature_cards["sae_id"] = cfg.model.sae_id
    feature_cards["layer"] = cfg.model.layer
    feature_cards["hook_name"] = cfg.model.hook_name
    feature_cards["site"] = cfg.model.site
    feature_cards["width"] = cfg.model.width
    feature_cards["l0"] = cfg.model.l0
    feature_cards["run_name"] = cfg.collection.run_name
    feature_cards["corpus"] = cfg.collection.corpus
    feature_cards["activation_mode"] = cfg.collection.activation_mode
    feature_cards["top_k"] = cfg.collection.top_k_features_per_token

    feature_cards["coactivation_neighbors_json"] = None
    feature_cards["decoder_neighbors_json"] = None
    feature_cards["bimodality_json"] = None
    feature_cards["pca_alignment_json"] = None
    feature_cards["natural_language_explanation"] = None
    feature_cards["explanation_confidence"] = None
    feature_cards["human_notes"] = None
    feature_cards["tags"] = None

    return feature_cards


def build_and_save_feature_outputs(
    acts: pd.DataFrame,
    token_meta: pd.DataFrame,
    cfg: ExperimentConfig,
) -> dict:
    acts_filtered = apply_activation_row_filters(acts, cfg.activation_filter)

    feature_stats = compute_feature_stats(acts_filtered, token_meta)
    filtered_features = apply_feature_filters(feature_stats, cfg.feature_filter)
    top_examples = build_top_examples(
        acts_filtered,
        token_meta,
        top_n=cfg.analysis.top_examples_per_feature,
        context_window=cfg.analysis.context_window,
    )
    feature_cards = build_feature_cards(filtered_features, top_examples, cfg)

    cfg.run_data_dir.mkdir(parents=True, exist_ok=True)
    feature_stats.to_parquet(cfg.feature_stats_path, index=False)
    filtered_features.to_parquet(cfg.filtered_features_path, index=False)
    top_examples.to_parquet(cfg.top_examples_path, index=False)
    feature_cards.to_parquet(cfg.feature_cards_path, index=False)

    return {
        "input_sparse_activation_rows": int(len(acts)),
        "input_unique_active_features": int(acts["feature_id"].nunique()),
        "filtered_sparse_activation_rows": int(len(acts_filtered)),
        "filtered_unique_active_features": int(acts_filtered["feature_id"].nunique()),
        "feature_stats_rows": int(len(feature_stats)),
        "filtered_features_rows": int(len(filtered_features)),
        "top_examples_rows": int(len(top_examples)),
        "feature_cards_rows": int(len(feature_cards)),
    }
