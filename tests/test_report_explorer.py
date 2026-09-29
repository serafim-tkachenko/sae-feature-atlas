from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.config.schema import PathsConfig
from sae_feature_atlas.pipeline.lineage import write_lineage
from sae_feature_atlas.report.explorer import (
    EXAMPLES_PER_GROUP,
    build_explorer_payload,
    write_explorer,
)


@pytest.fixture
def saved_run(tmp_path):
    cfg = replace(
        make_config(run_name="report-test"),
        paths=PathsConfig(
            data_root=tmp_path / "data",
            reports_root=tmp_path / "reports",
        ),
    )
    cfg.run_data_dir.mkdir(parents=True)
    write_lineage(cfg, "cards")
    pd.DataFrame(
        [
            {
                "feature_id": 7,
                "analysis_token_frequency": 0.1,
                "analysis_activation_count": 2,
                "analysis_token_denominator": 20,
                "stored_token_denominator": 25,
                "p99_activation": 3.0,
                "primary_label": "manual_review",
            },
            {
                "feature_id": 9,
                "analysis_token_frequency": 0.0,
                "analysis_activation_count": 0,
                "analysis_token_denominator": 20,
                "stored_token_denominator": 25,
                "p99_activation": np.nan,
            },
        ]
    ).to_parquet(cfg.feature_cards_path)
    pd.DataFrame(
        [
            {
                "feature_id": 7,
                "activation": float(i),
                "text_id": i,
                "token_pos": 1,
                "left_context": "before ",
                "center_token": "<script>alert(1)</script>",
                "right_context": " after",
                "source": "synthetic",
            }
            for i in range(12)
        ]
    ).to_parquet(cfg.top_examples_path)
    pd.DataFrame(
        {"feature_id": [7] * 5 + [9], "activation": [1.0, 2.0, 3.0, np.inf, np.nan, 0.0]}
    ).to_parquet(cfg.sae_activations_path)
    pd.DataFrame(
        [{"feature_i": 7, "feature_j": 9, "coactivation_count": 2, "jaccard": 0.5, "pmi": 1.2}]
    ).to_parquet(cfg.coactivation_pairs_path)
    return cfg


def test_payload_preserves_populations_bounds_contexts_and_links_both_pair_directions(saved_run):
    payload = build_explorer_payload(saved_run.run_data_dir)
    f7, f9 = payload["features"]
    assert payload["run"]["provenance"] == "recorded"
    assert payload["run"]["analysis_tokens"] == 20
    assert f7["population"] == "analysis_activations"
    assert f7["denominator"] == 20
    assert f9["frequency"] == 0  # Missing and zero are distinct.
    assert f9["p99"] is None
    assert len(f7["examples"]["top"]) == EXAMPLES_PER_GROUP
    assert f7["examples"]["top"][0]["activation"] == 11
    assert f7["examples"]["top"][0]["evidence_id"] == "f7:t11:p1"
    assert f7["neighbors"]["coactivation"][0]["id"] == 9
    assert f9["neighbors"]["coactivation"][0]["id"] == 7
    assert f9["neighbors"]["coactivation"][0]["support"] == 2
    assert f7["histogram"]["population"] == "all_stored_activations"
    assert sum(f7["histogram"]["counts"]) == 3  # Finite stored rows, not analysis count.
    assert f9["histogram"]["count"] == 1
    json.dumps(payload, allow_nan=False)


def test_legacy_values_are_not_relabelled_as_current_analysis(tmp_path):
    pd.DataFrame(
        [
            {
                "feature_id": 1,
                "token_frequency": 0.25,
                "n_token_activations": 5,
                "semantic_score": 0.99,
            }
        ]
    ).to_parquet(tmp_path / "feature_cards.parquet")
    payload = build_explorer_payload(tmp_path)
    feature = payload["features"][0]
    assert payload["run"]["provenance"] == "legacy"
    assert payload["warnings"]
    assert feature["population"] == "legacy_unspecified"
    assert feature["frequency"] == 0.25
    assert feature["denominator"] is None
    assert feature["triage_score"] is None


def test_saved_formatting_targets_warn_without_silently_changing_metrics(saved_run):
    examples = pd.read_parquet(saved_run.top_examples_path)
    examples["center_token"] = "’"
    examples.to_parquet(saved_run.top_examples_path)
    payload = build_explorer_payload(saved_run.run_data_dir)
    assert any("8 formatting targets" in warning for warning in payload["warnings"])
    feature = payload["features"][0]
    assert feature["p99"] == 3.0
    assert feature["examples"]["top"][0]["center_token"] == "’"
    assert payload["export_policy"]["example_target_quality_counts"] == {"quote": 8}


def test_missing_empty_and_unreadable_are_distinct(tmp_path):
    (tmp_path / "feature_cards.parquet").write_bytes(b"broken")
    pd.DataFrame().to_parquet(tmp_path / "decoder_neighbors.parquet")
    (tmp_path / "lineage.json").write_text("{")
    payload = build_explorer_payload(tmp_path)
    statuses = {item["name"]: item["status"] for item in payload["artifacts"]}
    assert statuses["feature_cards.parquet"] == "unreadable"
    assert statuses["decoder_neighbors.parquet"] == "empty"
    assert statuses["coactivation_pairs.parquet"] == "missing"
    assert payload["run"]["provenance"] == "unreadable"
    assert payload["features"] == []


@pytest.mark.parametrize("selected_table", ["feature_cards.parquet", "analysis_features.parquet"])
def test_empty_selection_never_falls_back_to_unselected_statistics(tmp_path, selected_table):
    pd.DataFrame(columns=["feature_id"]).to_parquet(tmp_path / selected_table)
    pd.DataFrame([{"feature_id": 7, "n_token_activations": 1}]).to_parquet(tmp_path / "feature_stats.parquet")
    payload = build_explorer_payload(tmp_path)
    assert payload["features"] == []
    assert payload["run"]["feature_source"] == selected_table
    assert any("No features" in warning for warning in payload["warnings"])


def test_empty_evidence_tables_override_stale_embedded_card_evidence(saved_run):
    cards = pd.read_parquet(saved_run.feature_cards_path)
    old_example = json.dumps([{"feature_id": 7, "activation": 9., "text_id": 0, "token_pos": 1,
                               "center_token": "stale"}])
    for column in ["top_examples_json", "bimodal_low_examples_json", "bimodal_high_examples_json"]:
        cards[column] = old_example
    for column in ["top_decoder_neighbors_json", "top_coactivation_neighbors_json"]:
        cards[column] = json.dumps([{"neighbor_feature_id": 9, "decoder_cosine": .9, "jaccard": .8}])
    cards.to_parquet(saved_run.feature_cards_path)
    for path in [saved_run.top_examples_path, saved_run.bimodal_peak_examples_path,
                 saved_run.decoder_neighbors_path, saved_run.coactivation_pairs_path]:
        pd.DataFrame().to_parquet(path)
    feature = build_explorer_payload(saved_run.run_data_dir)["features"][0]
    assert all(not rows for rows in feature["examples"].values())
    assert all(not rows for rows in feature["neighbors"].values())


def test_missing_required_columns_are_unreadable(tmp_path):
    pd.DataFrame([{"feature_id": 1}]).to_parquet(tmp_path / "coactivation_pairs.parquet")
    payload = build_explorer_payload(tmp_path)
    status = next(a for a in payload["artifacts"] if a["name"] == "coactivation_pairs.parquet")
    assert status["status"] == "unreadable"
    assert "Missing columns" in status["detail"]


@pytest.mark.parametrize(
    "lineage",
    [
        {"artifact_schema_version": 2},
        {"fingerprint_payloads": {"collection": []}},
        {"fingerprints": None},
    ],
)
def test_invalid_lineage_structure_is_reported_as_unreadable(tmp_path, lineage):
    (tmp_path / "lineage.json").write_text(json.dumps(lineage))
    payload = build_explorer_payload(tmp_path)
    assert payload["run"]["provenance"] == "unreadable"


def test_ambiguous_legacy_storage_does_not_choose_an_arbitrary_histogram(tmp_path):
    pd.DataFrame([{"feature_id": 1}]).to_parquet(tmp_path / "feature_cards.parquet")
    for mode in ("topk", "positive"):
        pd.DataFrame([{"feature_id": 1, "activation": 2.0}]).to_parquet(
            tmp_path / f"sae_activations_{mode}.parquet"
        )
    payload = build_explorer_payload(tmp_path)
    assert payload["features"][0]["histogram"] is None
    assert any("Both activation" in warning for warning in payload["warnings"])


def test_card_identity_must_match_recorded_lineage(saved_run):
    cards = pd.read_parquet(saved_run.feature_cards_path)
    cards["model_name"] = "wrong-model"
    cards.to_parquet(saved_run.feature_cards_path)
    with pytest.raises(ValueError, match="conflicts with lineage"):
        build_explorer_payload(saved_run.run_data_dir)


@pytest.mark.parametrize("ids", [[1, 1], [-1, 2], [1.5, 2], [None, 2]])
def test_invalid_feature_ids_are_rejected(tmp_path, ids):
    pd.DataFrame({"feature_id": ids}).to_parquet(tmp_path / "feature_cards.parquet")
    with pytest.raises(ValueError, match="unique non-negative integer"):
        build_explorer_payload(tmp_path)


def test_html_is_self_contained_script_safe_and_does_not_modify_inputs(saved_run):
    before = {p.name: p.read_bytes() for p in saved_run.run_data_dir.iterdir()}
    path = write_explorer(saved_run.run_data_dir, saved_run.html_report_path)
    html = path.read_text()
    assert "<script src=" not in html
    assert '<link rel="stylesheet"' not in html
    assert "<script>alert(1)</script>" not in html
    data = re.search(r'<script id="atlas-data" type="application/json">(.*?)</script>', html).group(
        1
    )
    payload = json.loads(data)
    assert (
        payload["features"][0]["examples"]["top"][0]["center_token"] == "<script>alert(1)</script>"
    )
    assert {p.name: p.read_bytes() for p in saved_run.run_data_dir.iterdir()} == before


def test_saved_run_cli_does_not_import_model_or_plotting_runtime(saved_run):
    code = """
import sys
for name in ('torch', 'sae_lens', 'transformer_lens', 'datasets', 'matplotlib'):
    sys.modules[name] = None
from sae_feature_atlas.app import main
main()
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            "report",
            "--run-dir",
            str(saved_run.run_data_dir),
            "--output",
            str(saved_run.html_report_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert saved_run.html_report_path.exists()


def test_cli_output_requires_run_directory():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from sae_feature_atlas.app import main; main()",
            "report",
            "--output",
            "wrong.html",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--output requires --run-dir" in result.stderr


def test_full_report_keeps_static_diagnostics_and_links_them(saved_run):
    from sae_feature_atlas.report.markdown import write_report

    write_report(saved_run)
    assert saved_run.summary_md_path.exists()
    assert (saved_run.run_reports_dir / "diagnostics.html").exists()
    assert (saved_run.run_reports_dir / "tables" / "feature_cards.html").exists()
    html = saved_run.html_report_path.read_text()
    assert '"Static diagnostics":"diagnostics.html"' in html
    assert '"Markdown summary":"summary.md"' in html
