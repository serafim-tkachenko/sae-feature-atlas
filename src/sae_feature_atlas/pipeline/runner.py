from __future__ import annotations

import json

import pandas as pd

from sae_feature_atlas.activations.collect import collect_sparse_sae_activations
from sae_feature_atlas.analysis.bimodality import compute_bimodality_candidates
from sae_feature_atlas.inspection.activation_regimes import build_bimodal_peak_examples
from sae_feature_atlas.analysis.coactivation import compute_same_token_coactivation
from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.analysis.coverage import compute_feature_coverage_profiles
from sae_feature_atlas.config.datasets import build_text_dataset, build_token_metadata, save_text_dataset
from sae_feature_atlas.inspection.feature_cards import build_and_save_feature_outputs, enrich_feature_cards
from sae_feature_atlas.analysis.feature_filters import apply_activation_row_filters
from sae_feature_atlas.analysis.geometry import compute_decoder_neighbors, merge_geometry_with_coactivation
from sae_feature_atlas.analysis.graph_alignment import compute_graph_alignment
from sae_feature_atlas.inspection.inspection import (
    feature_summaries_to_frame,
    pair_summaries_to_frame,
    summarize_features_batch,
    summarize_pair,
    write_inspection_reports,
)
from sae_feature_atlas.util.io import ensure_project_dirs
from sae_feature_atlas.runtime.loaders import get_device, load_model, load_sae, validate_model_sae_compatibility
from sae_feature_atlas.pipeline.manifest import build_run_manifest, write_run_manifest
from sae_feature_atlas.report.markdown import write_report
from sae_feature_atlas.analysis.space import (
    compute_decoder_lda,
    compute_decoder_pca,
    compute_decoder_umap,
    compute_residual_pca,
)


CORE_STEPS = ["collect"]

ATLAS_STEPS = [
    "collect",
    "features",
    "coactivation",
    "geometry",
    "geometry-vs-coactivation",
    "bimodality",
    "inspection",
    "space",
    "cards",
    "report"]

RESEARCH_STEPS = [
    "collect",
    "features",
    "coactivation",
    "geometry",
    "geometry-vs-coactivation",
    "bimodality",
    "inspection",
    "space",
    "coverage",
    "alignment",
    "cards",
    "report"]

STEP_PRESETS: dict[str, list[str]] = {
    "core": CORE_STEPS,
    "atlas": ATLAS_STEPS,
    "research": RESEARCH_STEPS,
}

ALL_STEPS = [
    "collect",
    "features",
    "coactivation",
    "geometry",
    "geometry-vs-coactivation",
    "bimodality",
    "inspection",
    "space",
    "coverage",
    "alignment",
    "cards",
    "report"]


def normalize_steps(steps: str | list[str] = "all", preset: str | None = None) -> list[str]:
    """Normalize user step selection into an executable step list.

    Presets define the architectural layers of the project:
    - core: data collection and Gemma Scope activation extraction;
    - atlas: reusable feature-card atlas generation;
    - research: atlas plus geometry-aware research metrics;
    """
    if preset is not None:
        if preset not in STEP_PRESETS:
            raise ValueError(f"Unknown preset {preset!r}. Available: {sorted(STEP_PRESETS)}")
        if steps in (None, "all"):
            return STEP_PRESETS[preset]

    raw = [s.strip() for s in steps.split(",")] if isinstance(steps, str) else list(steps)
    raw = [s for s in raw if s]

    if not raw or "all" in raw:
        return STEP_PRESETS.get(preset or "atlas", ATLAS_STEPS)

    if len(raw) == 1 and raw[0] in STEP_PRESETS:
        return STEP_PRESETS[raw[0]]

    unknown = sorted(set(raw) - set(ALL_STEPS))
    if unknown:
        raise ValueError(
            f"Unknown steps: {unknown}. Available steps: {ALL_STEPS}. "
            f"Available presets: {sorted(STEP_PRESETS)}"
        )
    return raw


def run_collect(cfg: ExperimentConfig) -> dict:
    ensure_project_dirs()
    cfg.run_data_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)
    print("Loading SAE...")
    sae = load_sae(cfg, device)
    print("Loading model...")
    model = load_model(cfg, device)

    print("Running smoke test...")
    print(json.dumps(validate_model_sae_compatibility(model, sae, cfg, device), indent=2))

    texts = build_text_dataset(cfg)
    save_text_dataset(cfg, texts)
    token_meta = build_token_metadata(model, cfg, texts, device)
    acts, residual_meta, token_summary = collect_sparse_sae_activations(model, sae, cfg, texts, device)

    return {
        "texts": len(texts),
        "tokens": int(token_meta[["text_id", "token_pos"]].drop_duplicates().shape[0]),
        "activation_rows": int(len(acts)),
        "unique_active_features": int(acts["feature_id"].nunique()),
        "residual_sample_rows": int(len(residual_meta)),
        "token_summary_rows": int(len(token_summary)),
    }


def run_features(cfg: ExperimentConfig) -> dict:
    return build_and_save_feature_outputs(
        pd.read_parquet(cfg.sae_activations_path),
        pd.read_parquet(cfg.token_metadata_path),
        cfg,
    )


def run_coactivation(cfg: ExperimentConfig) -> dict:
    acts = apply_activation_row_filters(pd.read_parquet(cfg.sae_activations_path), cfg.activation_filter)
    filtered = pd.read_parquet(cfg.filtered_features_path)
    pairs = compute_same_token_coactivation(
        acts,
        set(filtered["feature_id"].astype(int).tolist()),
        max_pairs=cfg.analysis.coactivation_max_pairs,
    )
    pairs.to_parquet(cfg.coactivation_pairs_path, index=False)
    return {"coactivation_pairs_rows": int(len(pairs))}


def run_geometry(cfg: ExperimentConfig) -> dict:
    sae = load_sae(cfg, get_device())
    filtered = pd.read_parquet(cfg.filtered_features_path)
    neighbors = compute_decoder_neighbors(
        sae,
        filtered["feature_id"].astype(int).tolist(),
        top_k=cfg.analysis.decoder_neighbors_top_k,
        batch_size=cfg.analysis.decoder_neighbors_batch_size,
    )
    neighbors.to_parquet(cfg.decoder_neighbors_path, index=False)
    return {"decoder_neighbors_rows": int(len(neighbors))}


def run_geometry_vs_coactivation(cfg: ExperimentConfig) -> dict:
    merged = merge_geometry_with_coactivation(
        pd.read_parquet(cfg.decoder_neighbors_path),
        pd.read_parquet(cfg.coactivation_pairs_path),
    )
    merged.to_parquet(cfg.geometry_vs_coactivation_path, index=False)
    return {"geometry_vs_coactivation_rows": int(len(merged))}


def run_bimodality(cfg: ExperimentConfig) -> dict:
    acts = apply_activation_row_filters(pd.read_parquet(cfg.sae_activations_path), cfg.activation_filter)
    token_meta = pd.read_parquet(cfg.token_metadata_path)
    candidates = compute_bimodality_candidates(acts, min_points=cfg.analysis.bimodality_min_points)
    candidates.to_parquet(cfg.bimodal_candidates_path, index=False)

    peak_examples = build_bimodal_peak_examples(
        acts,
        token_meta,
        candidates,
        top_features=cfg.analysis.bimodality_top_features_for_examples,
        examples_per_peak=cfg.analysis.bimodality_examples_per_peak,
        context_window=cfg.analysis.context_window,
    )
    peak_examples.to_parquet(cfg.bimodal_peak_examples_path, index=False)
    return {
        "bimodal_candidates_rows": int(len(candidates)),
        "bimodal_peak_example_rows": int(len(peak_examples)),
    }

def run_space(cfg: ExperimentConfig) -> dict:
    """Compute residual-space and SAE decoder-space projections."""
    metrics: dict[str, int] = {}

    if cfg.residual_vectors_path.exists():
        residual = compute_residual_pca(
            cfg.residual_vectors_path,
            n_components=cfg.analysis.pca_components,
        )
        residual.to_parquet(cfg.residual_pca_summary_path, index=False)
        metrics["residual_pca_components"] = int(len(residual))

    sae = load_sae(cfg, get_device())
    feature_stats = pd.read_parquet(cfg.feature_stats_path) if cfg.feature_stats_path.exists() else None

    summary, projection = compute_decoder_pca(
        sae,
        feature_stats,
        n_components=cfg.analysis.pca_components,
    )
    summary.to_parquet(cfg.decoder_pca_summary_path, index=False)
    projection.to_parquet(cfg.decoder_feature_pca_path, index=False)
    metrics["decoder_pca_components"] = int(len(summary))
    metrics["decoder_feature_projection_rows"] = int(len(projection))

    cards = pd.read_parquet(cfg.feature_cards_path) if cfg.feature_cards_path.exists() else None
    try:
        umap_df = compute_decoder_umap(
            sae,
            cards,
            n_neighbors=cfg.analysis.umap_neighbors,
            min_dist=cfg.analysis.umap_min_dist,
            metric=cfg.analysis.umap_metric,
            random_state=cfg.analysis.umap_random_state,
        )
        umap_df.to_parquet(cfg.decoder_feature_umap_path, index=False)
        metrics["decoder_umap_rows"] = int(len(umap_df))
    except ImportError as exc:
        print(f"Skipping decoder UMAP: {exc}")
        metrics["decoder_umap_rows"] = 0

    if cfg.feature_cards_path.exists():
        labeled_cards = pd.read_parquet(cfg.feature_cards_path)
        lda_df = compute_decoder_lda(
            sae,
            labeled_cards,
            min_class_size=cfg.analysis.lda_min_class_size,
        )
        if not lda_df.empty:
            lda_df.to_parquet(cfg.decoder_feature_lda_path, index=False)
        metrics["decoder_lda_rows"] = int(len(lda_df))

    return metrics


def run_coverage(cfg: ExperimentConfig) -> dict:
    """Research-grade coverage: decoder directions vs residual PCA components."""
    sae = load_sae(cfg, get_device())
    filtered = pd.read_parquet(cfg.filtered_features_path)
    coverage = compute_feature_coverage_profiles(
        sae=sae,
        residual_vectors_path=cfg.residual_vectors_path,
        feature_ids=filtered["feature_id"].astype(int).tolist(),
        n_components=cfg.analysis.pca_components,
        top_components=cfg.analysis.coverage_top_components,
    )
    coverage.to_parquet(cfg.feature_coverage_profiles_path, index=False)
    return {"feature_coverage_rows": int(len(coverage))}


def run_alignment(cfg: ExperimentConfig) -> dict:
    """Research-grade graph alignment between decoder geometry and coactivation."""
    filtered = pd.read_parquet(cfg.filtered_features_path)
    alignment, summary = compute_graph_alignment(
        decoder_neighbors=pd.read_parquet(cfg.decoder_neighbors_path),
        coactivation_pairs=pd.read_parquet(cfg.coactivation_pairs_path),
        feature_ids=filtered["feature_id"].astype(int).tolist(),
        k_values=cfg.analysis.graph_alignment_k_values,
    )
    alignment.to_parquet(cfg.graph_alignment_path, index=False)
    summary.to_parquet(cfg.graph_alignment_summary_path, index=False)
    return {
        "feature_graph_alignment_rows": int(len(alignment)),
        "graph_alignment_summary_rows": int(len(summary)),
    }


def run_candidates(cfg: ExperimentConfig) -> dict:
    """Rank atlas-derived candidates for later steering validation."""
    # Refresh cards first so candidate scoring sees coverage/alignment metrics.
    cards = enrich_feature_cards(cfg)
    return {"feature_steering_score_rows": int(len(candidates))}


def run_inspection(cfg: ExperimentConfig) -> dict:
    acts = pd.read_parquet(cfg.sae_activations_path)
    top_examples = pd.read_parquet(cfg.top_examples_path)
    filtered = pd.read_parquet(cfg.filtered_features_path)

    all_feature_ids = filtered["feature_id"].astype(int).tolist()
    feature_summaries = summarize_features_batch(
        acts=acts,
        top_examples=top_examples,
        feature_ids=all_feature_ids,
        max_seq_len=cfg.collection.max_seq_len,
    )

    feature_summary_by_id = {
        int(item["feature_id"]): item
        for item in feature_summaries
        if item.get("status") != "missing"
    }

    pair_summaries = []
    if cfg.coactivation_pairs_path.exists():
        coact = pd.read_parquet(cfg.coactivation_pairs_path)
        for row in coact.head(cfg.analysis.inspection_top_pairs).to_dict("records"):
            pair_summaries.append(
                summarize_pair(
                    acts=acts,
                    top_examples=top_examples,
                    feature_i=int(row["feature_i"]),
                    feature_j=int(row["feature_j"]),
                    pair_row=row,
                    max_seq_len=cfg.collection.max_seq_len,
                    feature_summary_by_id=feature_summary_by_id,
                )
            )

    bimodal_summaries = []
    if cfg.bimodal_candidates_path.exists():
        bimodal = pd.read_parquet(cfg.bimodal_candidates_path)
        bimodal_ids = bimodal.head(cfg.analysis.inspection_top_features)["feature_id"].astype(int).tolist()
        for fid in bimodal_ids:
            item = feature_summary_by_id.get(fid)
            if item is not None:
                bimodal_summaries.append(item)

    feature_summaries_to_frame(feature_summaries).to_parquet(
        cfg.inspection_feature_summaries_path,
        index=False,
    )
    pair_summaries_to_frame(pair_summaries).to_parquet(
        cfg.inspection_pair_summaries_path,
        index=False,
    )

    write_inspection_reports(
        cfg.collection.run_name,
        cfg.run_reports_dir,
        feature_summaries,
        pair_summaries,
        bimodal_summaries,
    )

    return {
        "inspection_features": len(feature_summaries),
        "inspection_pairs": len(pair_summaries),
        "inspection_scope": "all_filtered_features",
    }


def run_cards(cfg: ExperimentConfig) -> dict:
    cards = enrich_feature_cards(cfg)
    return {"feature_cards_rows": int(len(cards)), "feature_cards_columns": int(len(cards.columns))}


def run_pipeline(
    cfg: ExperimentConfig,
    steps: str | list[str] = "all",
    preset: str | None = None,
) -> dict:
    selected = normalize_steps(steps, preset=preset)
    metrics: dict[str, object] = {"preset": preset or "custom", "steps": selected}

    for step in selected:
        print(f"=== Running step: {step} ===")
        if step == "collect":
            metrics[step] = run_collect(cfg)
        elif step == "features":
            metrics[step] = run_features(cfg)
        elif step == "coactivation":
            metrics[step] = run_coactivation(cfg)
        elif step == "geometry":
            metrics[step] = run_geometry(cfg)
        elif step == "geometry-vs-coactivation":
            metrics[step] = run_geometry_vs_coactivation(cfg)
        elif step == "bimodality":
            metrics[step] = run_bimodality(cfg)
        elif step == "space":
            metrics[step] = run_space(cfg)
        elif step == "coverage":
            metrics[step] = run_coverage(cfg)
        elif step == "alignment":
            metrics[step] = run_alignment(cfg)
        elif step == :
            metrics[step] = run_candidates(cfg)
        elif step == "inspection":
            metrics[step] = run_inspection(cfg)
        elif step == "cards":
            metrics[step] = run_cards(cfg)
        elif step == "report":
            write_report(cfg)
            metrics[step] = {"summary": str(cfg.summary_md_path), "html": str(cfg.html_report_path)}
        else:  # pragma: no cover - normalize_steps should prevent this.
            raise ValueError(f"Unhandled pipeline step: {step}")

    write_run_manifest(cfg, build_run_manifest(cfg, metrics, stage="pipeline"))
    return metrics
