from __future__ import annotations

from dataclasses import replace

from sae_feature_atlas.app import _step_artifacts
from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.config.schema import PathsConfig
from sae_feature_atlas.pipeline.manifest import build_run_manifest


def _config(tmp_path):
    return replace(
        make_config(run_name="inventory-test"),
        paths=PathsConfig(
            raw_texts_path=tmp_path / "raw.jsonl",
            data_root=tmp_path / "data",
            reports_root=tmp_path / "reports",
        ),
    )


def test_manifest_names_population_and_lineage_artifacts(tmp_path) -> None:
    manifest = build_run_manifest(_config(tmp_path), metrics={}, stage="test")

    assert manifest["artifact_schema_version"] >= 2
    assert "analysis_features" in manifest["artifacts"]
    assert "filtered_features" not in manifest["artifacts"]
    assert "lineage" in manifest["artifacts"]
    assert "retained top-k memberships" in manifest["population_semantics"]["all_stored_activations"]


def test_cards_and_reports_declare_population_dependent_inputs(tmp_path) -> None:
    cfg = _config(tmp_path)
    artifacts = _step_artifacts(cfg)

    card_inputs = set(artifacts["cards"]["inputs"])
    assert cfg.analysis_features_path in card_inputs
    assert cfg.inspection_feature_summaries_path in card_inputs
    assert cfg.bimodal_candidates_path in card_inputs
    assert cfg.coactivation_pairs_path in card_inputs
    assert cfg.lineage_path in card_inputs

    report_inputs = set(artifacts["report"]["inputs"])
    assert cfg.feature_cards_path in report_inputs
    assert cfg.geometry_vs_coactivation_path in report_inputs
    assert cfg.decoder_residual_pc_alignment_path in report_inputs
    assert cfg.lineage_path in report_inputs
