from __future__ import annotations

import json

import pandas as pd

from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.analysis.feature_stats import build_top_examples, compute_feature_stats
from sae_feature_atlas.analysis.feature_filters import apply_activation_row_filters, apply_feature_filters
from sae_feature_atlas.analysis.labels import assign_feature_labels


def _json(records) -> str:
    return json.dumps(records, ensure_ascii=False)


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
    existing_cols = [col for col in cols if col in df.columns]
    records = df.sort_values("activation", ascending=False).head(n)[existing_cols].to_dict("records")
    return _json(records)


def _drop_columns_if_present(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return df
    return df.drop(columns=existing)


def _merge_replace(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    columns_to_replace: list[str],
) -> pd.DataFrame:
    """Merge enrichment columns in an idempotent way."""
    if right is None or right.empty or on not in right.columns:
        return left

    right_cols = [on] + [col for col in columns_to_replace if col in right.columns]
    right_small = right[right_cols].drop_duplicates(subset=[on])
    left = _drop_columns_if_present(left, [col for col in right_cols if col != on])
    return left.merge(right_small, on=on, how="left")


def build_basic_feature_cards(
    filtered_features: pd.DataFrame,
    top_examples: pd.DataFrame,
    cfg: ExperimentConfig,
) -> pd.DataFrame:
    """Build the base feature-card table.

    Later steps add independent evidence channels: inspection, coactivation,
    decoder geometry, activation regimes, residual coverage, graph alignment,
    and steering-candidate scores.
    """
    rows: list[dict] = []
    for feature_id, group in top_examples.groupby("feature_id"):
        rows.append({"feature_id": int(feature_id), "top_examples_json": _top_examples_json(group)})

    examples_df = pd.DataFrame(rows)
    cards = filtered_features.merge(examples_df, on="feature_id", how="left")

    cards["model_name"] = cfg.model.model_name
    cards["sae_release"] = cfg.model.sae_release
    cards["sae_id"] = cfg.model.sae_id
    cards["layer"] = cfg.model.layer
    cards["hook_name"] = cfg.model.hook_name
    cards["site"] = cfg.model.site
    cards["width"] = cfg.model.width
    cards["l0"] = cfg.model.l0
    cards["run_name"] = cfg.collection.run_name
    cards["corpus"] = cfg.collection.corpus
    cards["activation_mode"] = cfg.collection.activation_mode
    cards["top_k"] = cfg.collection.top_k_features_per_token
    return cards


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
    cards = build_basic_feature_cards(filtered_features, top_examples, cfg)

    cfg.run_data_dir.mkdir(parents=True, exist_ok=True)
    feature_stats.to_parquet(cfg.feature_stats_path, index=False)
    filtered_features.to_parquet(cfg.filtered_features_path, index=False)
    top_examples.to_parquet(cfg.top_examples_path, index=False)
    cards.to_parquet(cfg.feature_cards_path, index=False)

    return {
        "input_sparse_activation_rows": int(len(acts)),
        "input_unique_active_features": int(acts["feature_id"].nunique()),
        "filtered_sparse_activation_rows": int(len(acts_filtered)),
        "filtered_unique_active_features": int(acts_filtered["feature_id"].nunique()),
        "feature_stats_rows": int(len(feature_stats)),
        "filtered_features_rows": int(len(filtered_features)),
        "top_examples_rows": int(len(top_examples)),
        "feature_cards_rows": int(len(cards)),
    }


def _load_feature_cards_base(cfg: ExperimentConfig) -> pd.DataFrame:
    if cfg.feature_cards_path.exists():
        return pd.read_parquet(cfg.feature_cards_path)
    if not cfg.filtered_features_path.exists():
        raise FileNotFoundError(
            f"Missing both {cfg.feature_cards_path} and {cfg.filtered_features_path}. "
            "Run the `features` step first."
        )
    cards = pd.read_parquet(cfg.filtered_features_path)
    cards["model_name"] = cfg.model.model_name
    cards["sae_release"] = cfg.model.sae_release
    cards["sae_id"] = cfg.model.sae_id
    cards["layer"] = cfg.model.layer
    cards["hook_name"] = cfg.model.hook_name
    cards["site"] = cfg.model.site
    cards["width"] = cfg.model.width
    cards["l0"] = cfg.model.l0
    cards["run_name"] = cfg.collection.run_name
    cards["corpus"] = cfg.collection.corpus
    cards["activation_mode"] = cfg.collection.activation_mode
    cards["top_k"] = cfg.collection.top_k_features_per_token
    return cards


def _merge_inspection(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.inspection_feature_summaries_path.exists():
        return cards
    inspection = pd.read_parquet(cfg.inspection_feature_summaries_path)
    return _merge_replace(
        cards,
        inspection,
        on="feature_id",
        columns_to_replace=[
            "n_activations_inspected",
            "quote_share",
            "punctuation_share",
            "space_share",
            "boundary_share",
            "token_concentration",
            "source_concentration",
            "raw_source_concentration",
            "source_is_informative",
            "position_concentration",
            "artifact_score",
            "semantic_score",
            "manual_priority",
            "inspection_labels",
            "top_tokens_json",
            "top_positions_json",
            "top_sources_json",
        ],
    )


def _merge_bimodality(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.bimodal_candidates_path.exists():
        return cards
    bimodal = pd.read_parquet(cfg.bimodal_candidates_path)
    keep = [
        "feature_id",
        "bimodality_score",
        "log_mean_low",
        "log_mean_high",
        "activation_min",
        "activation_p50",
        "activation_p95",
        "activation_max",
        "n_points",
    ]
    keep = [col for col in keep if col in bimodal.columns]
    return _merge_replace(cards, bimodal[keep], on="feature_id", columns_to_replace=[c for c in keep if c != "feature_id"])


def _merge_decoder_neighbors(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.decoder_neighbors_path.exists():
        return cards
    neighbors = pd.read_parquet(cfg.decoder_neighbors_path)
    if neighbors.empty:
        return cards

    max_cos = (
        neighbors.groupby("feature_i")["decoder_cosine"]
        .max()
        .reset_index()
        .rename(columns={"feature_i": "feature_id", "decoder_cosine": "max_decoder_cosine"})
    )
    cards = _merge_replace(cards, max_cos, on="feature_id", columns_to_replace=["max_decoder_cosine"])

    rows: list[dict] = []
    for feature_id, group in neighbors.groupby("feature_i"):
        rows.append(
            {
                "feature_id": int(feature_id),
                "top_decoder_neighbors_json": _json(group.sort_values("rank").head(10).to_dict("records")),
            }
        )
    return _merge_replace(cards, pd.DataFrame(rows), on="feature_id", columns_to_replace=["top_decoder_neighbors_json"])


def _merge_coactivation(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.coactivation_pairs_path.exists():
        return cards
    coactivation = pd.read_parquet(cfg.coactivation_pairs_path)
    if coactivation.empty:
        return cards

    left = coactivation.rename(columns={"feature_i": "feature_id", "feature_j": "neighbor_feature_id"})
    right = coactivation.rename(columns={"feature_j": "feature_id", "feature_i": "neighbor_feature_id"})
    both = pd.concat([left, right], ignore_index=True)

    aggregate = (
        both.groupby("feature_id")
        .agg(max_coactivation_jaccard=("jaccard", "max"), max_coactivation_pmi=("pmi", "max"))
        .reset_index()
    )
    cards = _merge_replace(
        cards,
        aggregate,
        on="feature_id",
        columns_to_replace=["max_coactivation_jaccard", "max_coactivation_pmi"],
    )

    rows: list[dict] = []
    for feature_id, group in both.groupby("feature_id"):
        top = group.sort_values(["jaccard", "pmi", "coactivation_count"], ascending=False).head(10)
        rows.append({"feature_id": int(feature_id), "top_coactivation_neighbors_json": _json(top.to_dict("records"))})
    return _merge_replace(cards, pd.DataFrame(rows), on="feature_id", columns_to_replace=["top_coactivation_neighbors_json"])


def _merge_decoder_pca(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.decoder_feature_pca_path.exists():
        return cards
    pca = pd.read_parquet(cfg.decoder_feature_pca_path)
    keep = [col for col in ["feature_id", "decoder_pc1", "decoder_pc2"] if col in pca.columns]
    if len(keep) <= 1:
        return cards
    return _merge_replace(cards, pca[keep], on="feature_id", columns_to_replace=[c for c in keep if c != "feature_id"])


def _merge_decoder_umap(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.decoder_feature_umap_path.exists():
        return cards
    umap_df = pd.read_parquet(cfg.decoder_feature_umap_path)
    keep = [col for col in ["feature_id", "decoder_umap_x", "decoder_umap_y"] if col in umap_df.columns]
    if len(keep) <= 1:
        return cards
    return _merge_replace(cards, umap_df[keep], on="feature_id", columns_to_replace=[c for c in keep if c != "feature_id"])


def _merge_decoder_lda(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.decoder_feature_lda_path.exists():
        return cards
    lda = pd.read_parquet(cfg.decoder_feature_lda_path)
    keep = [col for col in ["feature_id", "decoder_lda1", "decoder_lda2"] if col in lda.columns]
    if len(keep) <= 1:
        return cards
    return _merge_replace(cards, lda[keep], on="feature_id", columns_to_replace=[c for c in keep if c != "feature_id"])


def _merge_coverage(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.feature_coverage_profiles_path.exists():
        return cards
    coverage = pd.read_parquet(cfg.feature_coverage_profiles_path)
    columns = [
        "pc_mass_observed",
        "pc_mass_unobserved_tail",
        "effective_pc_dim",
        "pc_entropy",
        "pc_center_of_mass",
        "pc_mass_top_1",
        "pc_mass_top_5",
        "pc_mass_top_20",
        "pc_norm_mass_top_1",
        "pc_norm_mass_top_5",
        "pc_norm_mass_top_20",
        "coverage_bucket",
    ]
    return _merge_replace(cards, coverage, on="feature_id", columns_to_replace=columns)


def _merge_alignment(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.graph_alignment_path.exists():
        return cards
    alignment = pd.read_parquet(cfg.graph_alignment_path)
    columns = [
        "n_geometry_neighbors",
        "n_coactivation_neighbors",
        "gca_at_5",
        "gca_at_10",
        "gca_at_20",
        "graph_alignment_bucket",
    ]
    return _merge_replace(cards, alignment, on="feature_id", columns_to_replace=columns)


def _merge_candidates(cards: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    if not cfg.feature_steering_scores_path.exists():
        return cards

    candidates = pd.read_parquet(cfg.feature_steering_scores_path)

    columns = [
        "graph_agreement_score",
        "coverage_coherence_score",
        "atlas_steering_score",
        "steering_risk_score",
        "steering_candidate_rank",
        "is_steering_candidate",
        "steering_candidate_reason",
    ]

    return _merge_replace(cards, candidates, on="feature_id", columns_to_replace=columns)


def enrich_feature_cards(cfg: ExperimentConfig) -> pd.DataFrame:
    """Build the canonical feature-card table from all available evidence channels."""
    cards = _load_feature_cards_base(cfg)

    cards = _merge_inspection(cards, cfg)
    cards = _merge_bimodality(cards, cfg)
    cards = _merge_decoder_neighbors(cards, cfg)
    cards = _merge_coactivation(cards, cfg)
    cards = _merge_decoder_pca(cards, cfg)
    cards = _merge_decoder_umap(cards, cfg)
    cards = _merge_decoder_lda(cards, cfg)
    cards = _merge_coverage(cards, cfg)
    cards = _merge_alignment(cards, cfg)
    cards = _merge_candidates(cards, cfg)

    cards = assign_feature_labels(cards)

    preferred_order = [
        "feature_id",
        "primary_label",
        "manual_priority",
        "inspection_labels",
        "artifact_score",
        "semantic_score",
        "atlas_steering_score",
        "steering_risk_score",
        "graph_agreement_score",
        "coverage_coherence_score",
        "steering_candidate_rank",
        "is_steering_candidate"
        "steering_candidate_reason",
        "n_token_activations",
        "n_texts",
        "token_frequency",
        "text_frequency",
        "mean_activation",
        "p50_activation",
        "p95_activation",
        "p99_activation",
        "max_activation",
        "bimodality_score",
        "max_decoder_cosine",
        "max_coactivation_jaccard",
        "max_coactivation_pmi",
        "gca_at_5",
        "gca_at_10",
        "gca_at_20",
        "graph_alignment_bucket",
        "pc_norm_mass_top_20",
        "effective_pc_dim",
        "pc_entropy",
        "coverage_bucket",
        "decoder_pc1",
        "decoder_pc2",
        "decoder_umap_x",
        "decoder_umap_y",
        "decoder_lda1",
        "decoder_lda2",
        "top_examples_json",
        "top_decoder_neighbors_json",
        "top_coactivation_neighbors_json",
    ]

    existing_preferred = [col for col in preferred_order if col in cards.columns]
    remaining = [col for col in cards.columns if col not in existing_preferred]
    cards = cards[existing_preferred + remaining]
    cards.to_parquet(cfg.feature_cards_path, index=False)
    return cards
