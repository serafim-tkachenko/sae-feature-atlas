from __future__ import annotations

import json
from dataclasses import asdict

from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.pipeline.lineage import build_lineage
from sae_feature_atlas.util.io import write_json


def build_run_manifest(cfg: ExperimentConfig, metrics: dict, stage: str) -> dict:
    lineage = (
        json.loads(cfg.lineage_path.read_text(encoding="utf-8"))
        if cfg.lineage_path.exists()
        else build_lineage(cfg, stage)
    )
    return {
        "stage": stage,
        "run_name": cfg.collection.run_name,
        "artifact_schema_version": lineage["artifact_schema_version"],
        "fingerprints": lineage["fingerprints"],
        "git": lineage["git"],
        "model": asdict(cfg.model),
        "collection": asdict(cfg.collection),
        "activation_row_filter": asdict(cfg.activation_filter),
        "feature_filter": asdict(cfg.feature_filter),
        "analysis": asdict(cfg.analysis),
        "metrics": metrics,
        "artifacts": {
            "source_texts": str(cfg.source_texts_path),
            "token_metadata": str(cfg.token_metadata_path),
            "all_stored_activations": str(cfg.sae_activations_path),
            "token_activation_summary": str(cfg.token_activation_summary_path),
            "residual_vectors_sample": str(cfg.residual_vectors_path),
            "residual_vectors_metadata": str(cfg.residual_metadata_path),
            "feature_stats": str(cfg.feature_stats_path),
            "analysis_features": str(cfg.analysis_features_path),
            "top_feature_examples": str(cfg.top_examples_path),
            "feature_cards": str(cfg.feature_cards_path),
            "coactivation_pairs": str(cfg.coactivation_pairs_path),
            "coactivation_metadata": str(cfg.coactivation_metadata_path),
            "decoder_neighbors": str(cfg.decoder_neighbors_path),
            "geometry_vs_coactivation": str(cfg.geometry_vs_coactivation_path),
            "bimodality_evaluated_features": str(cfg.bimodality_evaluated_path),
            "bimodal_candidates": str(cfg.bimodal_candidates_path),
            "bimodal_peak_examples": str(cfg.bimodal_peak_examples_path),
            "decoder_residual_pc_alignment": str(cfg.decoder_residual_pc_alignment_path),
            "graph_alignment": str(cfg.graph_alignment_path),
            "lineage": str(cfg.lineage_path),
            "summary_md": str(cfg.summary_md_path),
            "html_report": str(cfg.html_report_path),
        },
        "population_semantics": {
            "all_stored_activations": (
                "Every persisted sparse activation row; under top-k mode these "
                "are retained top-k memberships, not all positive activations."
            ),
            "analysis_activations": (
                "Stored rows whose target tokens pass the configured analysis "
                "eligibility and activation-row policy."
            ),
            "analysis_features": (
                "Features meeting configured support criteria in the analysis population."
            ),
        },
        "notes": [
            "Frequency denominators come from explicit stored and eligible token universes.",
            "Human-facing contexts are decoded from contiguous token IDs.",
            "Feature cards combine empirical evidence and triage diagnostics; they are not semantic annotations.",
        ],
    }


def write_run_manifest(cfg: ExperimentConfig, manifest: dict) -> None:
    cfg.run_reports_dir.mkdir(parents=True, exist_ok=True)
    write_json(cfg.manifest_path, manifest)
