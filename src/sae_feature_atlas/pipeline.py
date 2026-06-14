from __future__ import annotations

import json

import pandas as pd

from sae_feature_atlas.activations import collect_sparse_sae_activations
from sae_feature_atlas.bimodality import compute_bimodality_candidates
from sae_feature_atlas.coactivation import compute_same_token_coactivation
from sae_feature_atlas.config import ExperimentConfig
from sae_feature_atlas.datasets import build_text_dataset, build_token_metadata, save_text_dataset
from sae_feature_atlas.feature_cards import build_and_save_feature_outputs
from sae_feature_atlas.filters import apply_activation_row_filters
from sae_feature_atlas.geometry import compute_decoder_neighbors, merge_geometry_with_coactivation
from sae_feature_atlas.io_utils import ensure_project_dirs
from sae_feature_atlas.loaders import get_device, load_model, load_sae, validate_model_sae_compatibility
from sae_feature_atlas.manifest import build_run_manifest, write_run_manifest
from sae_feature_atlas.reporting import write_report


ALL_STEPS = [
    "collect",
    "features",
    "coactivation",
    "geometry",
    "geometry-vs-coactivation",
    "bimodality",
    "report",
]


def normalize_steps(steps: str | list[str]) -> list[str]:
    if isinstance(steps, str):
        raw = [step.strip() for step in steps.split(",") if step.strip()]
    else:
        raw = steps
    if not raw or "all" in raw:
        return ALL_STEPS
    unknown = sorted(set(raw) - set(ALL_STEPS))
    if unknown:
        raise ValueError(f"Unknown steps: {unknown}. Available: {ALL_STEPS}")
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
    smoke = validate_model_sae_compatibility(model, sae, cfg, device)
    print(json.dumps(smoke, indent=2))

    print("Building text dataset...")
    texts = build_text_dataset(cfg)
    save_text_dataset(cfg, texts)
    print(f"Saved {len(texts)} texts to {cfg.paths.raw_texts_path}")

    print("Building token metadata...")
    token_meta = build_token_metadata(model, cfg, texts, device)
    print(f"Token metadata: {token_meta.shape}")

    print("Collecting sparse SAE activations...")
    acts, residual_meta, token_summary = collect_sparse_sae_activations(model, sae, cfg, texts, device)

    metrics = {
        "texts": len(texts),
        "tokens": int(token_meta[["text_id", "token_pos"]].drop_duplicates().shape[0]),
        "activation_rows": int(len(acts)),
        "unique_active_features": int(acts["feature_id"].nunique()),
        "residual_sample_rows": int(len(residual_meta)),
        "token_summary_rows": int(len(token_summary)),
    }

    manifest = build_run_manifest(cfg, metrics, stage="collect")
    write_run_manifest(cfg, manifest)
    return metrics


def run_features(cfg: ExperimentConfig) -> dict:
    acts = pd.read_parquet(cfg.sae_activations_path)
    token_meta = pd.read_parquet(cfg.token_metadata_path)

    metrics = build_and_save_feature_outputs(acts, token_meta, cfg)
    manifest = build_run_manifest(cfg, metrics, stage="features")
    write_run_manifest(cfg, manifest)
    return metrics


def run_coactivation(cfg: ExperimentConfig) -> dict:
    acts = pd.read_parquet(cfg.sae_activations_path)
    filtered_features = pd.read_parquet(cfg.filtered_features_path)
    acts = apply_activation_row_filters(acts, cfg.activation_filter)

    feature_ids = set(filtered_features["feature_id"].astype(int).tolist())
    pairs = compute_same_token_coactivation(
        acts=acts,
        filtered_feature_ids=feature_ids,
        max_pairs=cfg.analysis.coactivation_max_pairs,
    )

    pairs.to_parquet(cfg.coactivation_pairs_path, index=False)
    return {"coactivation_pairs_rows": int(len(pairs))}


def run_geometry(cfg: ExperimentConfig) -> dict:
    device = get_device()
    sae = load_sae(cfg, device)
    filtered_features = pd.read_parquet(cfg.filtered_features_path)
    feature_ids = filtered_features["feature_id"].astype(int).tolist()

    neighbors = compute_decoder_neighbors(
        sae=sae,
        feature_ids=feature_ids,
        top_k=cfg.analysis.decoder_neighbors_top_k,
        batch_size=cfg.analysis.decoder_neighbors_batch_size,
    )

    neighbors.to_parquet(cfg.decoder_neighbors_path, index=False)
    return {"decoder_neighbors_rows": int(len(neighbors))}


def run_geometry_vs_coactivation(cfg: ExperimentConfig) -> dict:
    coact = pd.read_parquet(cfg.coactivation_pairs_path)
    neigh = pd.read_parquet(cfg.decoder_neighbors_path)
    merged = merge_geometry_with_coactivation(neigh, coact)
    merged.to_parquet(cfg.geometry_vs_coactivation_path, index=False)
    return {"geometry_vs_coactivation_rows": int(len(merged))}


def run_bimodality(cfg: ExperimentConfig) -> dict:
    acts = pd.read_parquet(cfg.sae_activations_path)
    acts = apply_activation_row_filters(acts, cfg.activation_filter)

    candidates = compute_bimodality_candidates(
        acts,
        min_points=cfg.analysis.bimodality_min_points,
    )
    candidates.to_parquet(cfg.bimodal_candidates_path, index=False)
    return {"bimodal_candidates_rows": int(len(candidates))}


def run_pipeline(cfg: ExperimentConfig, steps: str | list[str] = "all") -> dict:
    selected_steps = normalize_steps(steps)
    metrics: dict[str, object] = {"steps": selected_steps}

    for step in selected_steps:
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
        elif step == "report":
            write_report(cfg)
            metrics[step] = {"summary": str(cfg.summary_md_path), "html": str(cfg.html_report_path)}
        else:
            raise ValueError(step)

    manifest = build_run_manifest(cfg, metrics, stage="pipeline")
    write_run_manifest(cfg, manifest)
    return metrics
