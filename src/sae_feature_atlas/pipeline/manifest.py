from __future__ import annotations

from dataclasses import asdict

from sae_feature_atlas.config.config import ExperimentConfig
from sae_feature_atlas.util.io_utils import write_json


def build_run_manifest(cfg: ExperimentConfig, metrics: dict, stage: str) -> dict:
    return {
        "stage": stage,
        "run_name": cfg.collection.run_name,
        "model": asdict(cfg.model),
        "collection": asdict(cfg.collection),
        "activation_row_filter": asdict(cfg.activation_filter),
        "feature_filter": asdict(cfg.feature_filter),
        "analysis": asdict(cfg.analysis),
        "metrics": metrics,
        "artifacts": {
            "raw_texts": str(cfg.paths.raw_texts_path),
            "token_metadata": str(cfg.token_metadata_path),
            "sae_activations": str(cfg.sae_activations_path),
            "token_activation_summary": str(cfg.token_activation_summary_path),
            "residual_vectors_sample": str(cfg.residual_vectors_path),
            "residual_vectors_metadata": str(cfg.residual_metadata_path),
            "feature_stats": str(cfg.feature_stats_path),
            "filtered_features": str(cfg.filtered_features_path),
            "top_feature_examples": str(cfg.top_examples_path),
            "feature_cards": str(cfg.feature_cards_path),
            "coactivation_pairs": str(cfg.coactivation_pairs_path),
            "decoder_neighbors": str(cfg.decoder_neighbors_path),
            "geometry_vs_coactivation": str(cfg.geometry_vs_coactivation_path),
            "bimodal_candidates": str(cfg.bimodal_candidates_path),
            "summary_md": str(cfg.summary_md_path),
            "html_report": str(cfg.html_report_path),
        },
        "notes": [
            "SAE activations can be stored as top-k rows or all positive activations.",
            "Activation-row filtering is applied before feature statistics.",
            "Feature cards are the central reusable artifact.",
        ],
    }


def write_run_manifest(cfg: ExperimentConfig, manifest: dict) -> None:
    cfg.run_reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.manifest_path, manifest)
