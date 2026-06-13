from __future__ import annotations

import json

import pandas as pd
from tqdm import tqdm

from sae_nla_rnd.config import ExperimentConfig
from sae_nla_rnd.filters import apply_activation_row_filters, apply_feature_filters


def compute_feature_stats(acts: pd.DataFrame, token_meta: pd.DataFrame) -> pd.DataFrame:
    relevant_positions = acts[["text_id", "token_pos"]].drop_duplicates()
    token_meta_relevant = token_meta.merge(relevant_positions, on=["text_id", "token_pos"], how="inner")

    n_total_tokens = token_meta_relevant[["text_id", "token_pos"]].drop_duplicates().shape[0]
    n_total_texts = token_meta_relevant["text_id"].nunique()

    if n_total_tokens == 0 or n_total_texts == 0:
        raise ValueError("No tokens left after activation-row filtering")

    feature_stats = (
        acts.groupby("feature_id")
        .agg(
            n_token_activations=("activation", "size"),
            n_texts=("text_id", "nunique"),
            mean_activation=("activation", "mean"),
            max_activation=("activation", "max"),
            p95_activation=("activation", lambda x: x.quantile(0.95)),
            p99_activation=("activation", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )

    feature_stats["token_frequency"] = feature_stats["n_token_activations"] / n_total_tokens
    feature_stats["text_frequency"] = feature_stats["n_texts"] / n_total_texts

    return feature_stats.sort_values("n_token_activations", ascending=False)


def build_top_examples(
    acts: pd.DataFrame,
    token_meta: pd.DataFrame,
    top_n: int = 20,
    context_window: int = 20,
) -> pd.DataFrame:
    top_rows: list[dict] = []

    token_groups = {
        text_id: group.sort_values("token_pos")["token_str"].tolist()
        for text_id, group in token_meta.groupby("text_id")
    }

    for feature_id, group in tqdm(acts.groupby("feature_id"), desc="Building top examples"):
        group = group.sort_values("activation", ascending=False).head(top_n)

        for rank, row in enumerate(group.itertuples(index=False), start=1):
            toks = token_groups[row.text_id]
            pos = int(row.token_pos)

            left = "".join(toks[max(0, pos - context_window):pos])
            center = toks[pos] if pos < len(toks) else row.token_str
            right = "".join(toks[pos + 1 : pos + 1 + context_window])

            top_rows.append(
                {
                    "feature_id": int(feature_id),
                    "rank": int(rank),
                    "activation": float(row.activation),
                    "text_id": int(row.text_id),
                    "source": row.source,
                    "token_pos": int(pos),
                    "token_str": row.token_str,
                    "left_context": left,
                    "center_token": center,
                    "right_context": right,
                }
            )

    return pd.DataFrame(top_rows)


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

    feature_cards = filtered_features.merge(
        top_examples_small,
        on="feature_id",
        how="left",
    )

    feature_cards["model_name"] = cfg.model.model_name
    feature_cards["sae_release"] = cfg.model.sae_release
    feature_cards["sae_id"] = cfg.model.sae_id
    feature_cards["layer"] = cfg.model.layer
    feature_cards["hook_name"] = cfg.model.hook_name
    feature_cards["run_name"] = cfg.collection.run_name

    # Empty placeholders for future analysis stages
    feature_cards["coactivation_neighbors_json"] = None
    feature_cards["decoder_neighbors_json"] = None
    feature_cards["bimodality_json"] = None
    feature_cards["pca_alignment_json"] = None
    feature_cards["natural_language_explanation"] = None
    feature_cards["steering_result_json"] = None

    return feature_cards


def build_and_save_feature_outputs(
    acts: pd.DataFrame,
    token_meta: pd.DataFrame,
    cfg: ExperimentConfig,
) -> dict:
    acts_filtered = apply_activation_row_filters(acts, cfg.activation_filter)

    feature_stats = compute_feature_stats(acts_filtered, token_meta)
    filtered_features = apply_feature_filters(feature_stats, cfg.feature_filter)
    top_examples = build_top_examples(acts_filtered, token_meta)
    feature_cards = build_feature_cards(filtered_features, top_examples, cfg)

    cfg.run_data_dir.mkdir(parents=True, exist_ok=True)
    feature_stats.to_parquet(cfg.feature_stats_path, index=False)
    filtered_features.to_parquet(cfg.filtered_features_path, index=False)
    top_examples.to_parquet(cfg.top_examples_path, index=False)
    feature_cards.to_parquet(cfg.feature_cards_path, index=False)

    return {
        "input_sparse_activation_rows": int(len(acts)),
        "input_unique_active_features_topk": int(acts["feature_id"].nunique()),
        "filtered_sparse_activation_rows": int(len(acts_filtered)),
        "filtered_unique_active_features_topk": int(acts_filtered["feature_id"].nunique()),
        "feature_stats_rows": int(len(feature_stats)),
        "filtered_features_rows": int(len(filtered_features)),
        "top_examples_rows": int(len(top_examples)),
        "feature_cards_rows": int(len(feature_cards)),
    }
