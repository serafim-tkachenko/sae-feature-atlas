"""CLI and notebook orchestration; all scientific work lives in package modules."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

import pandas as pd

from sae_feature_atlas.analysis.populations import (
    ActivationPopulations,
    TOKEN_KEY,
    build_analysis_token_population,
)
from sae_feature_atlas.analysis.feature_stats import compute_feature_stats
from sae_feature_atlas.analysis.feature_filters import apply_feature_filters
from sae_feature_atlas.pipeline.lineage import require_compatible_lineage, write_lineage
from sae_feature_atlas.scientific.collect import configuration, collect, sha256
from sae_feature_atlas.scientific.regimes import RegimeConfig, run_regimes
from sae_feature_atlas.util.io import write_json


def analyze(cfg, regime_cfg=RegimeConfig()):
    require_compatible_lineage(cfg)
    tokens = pd.read_parquet(cfg.token_metadata_path)
    eligible = build_analysis_token_population(tokens, cfg.activation_filter)
    stored = pd.read_parquet(cfg.sae_activations_path)
    acts = stored.merge(eligible[TOKEN_KEY], on=TOKEN_KEY, validate="many_to_one")
    # Collection is strictly finite and positive; reject row-threshold policies
    # here because partner absence would then acquire a different estimand.
    if cfg.activation_filter.min_activation is not None:
        raise ValueError("Regime experiment requires min_activation=None.")
    populations = ActivationPopulations(stored, acts, tokens, eligible)
    stats = compute_feature_stats(populations, cfg.collection.activation_mode)
    stats.to_parquet(cfg.feature_stats_path, index=False)
    apply_feature_filters(stats, cfg.feature_filter).to_parquet(
        cfg.analysis_features_path, index=False
    )
    del populations, stored
    support = pd.read_parquet(cfg.token_activation_summary_path)
    result = run_regimes(
        acts,
        eligible,
        support,
        regime_cfg,
        cfg.run_data_dir,
        storage_mode=cfg.collection.activation_mode,
    )
    result.update(stored_tokens=len(tokens), analysis_tokens=len(eligible), analysis_rows=len(acts))
    write_json(cfg.run_data_dir / "regime_run_summary.json", result)
    write_lineage(cfg, "scientific_regimes")
    return result


def manifest(cfg, regime_cfg):
    from sae_feature_atlas.pipeline.lineage import git_provenance
    from pathlib import Path

    provenance = json.loads((cfg.run_data_dir / "collection_provenance.json").read_text())
    payload = {
        "schema_version": 1,
        "git": git_provenance(),
        "experiment_config": asdict(cfg),
        "regime_config": asdict(regime_cfg),
        "collection_provenance": provenance,
        "analysis_summary": json.loads((cfg.run_data_dir / "regime_run_summary.json").read_text()),
        "code_hashes": {str(p): sha256(p) for p in sorted(Path("src").rglob("*.py"))},
        "artifact_hashes": {
            str(p.relative_to(cfg.run_data_dir)): sha256(p)
            for p in sorted(cfg.run_data_dir.glob("*"))
            if p.is_file() and p.name != "scientific_experiment_manifest.json"
        },
    }
    from datetime import datetime, timezone

    payload["manifest_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["report_hashes"] = {
        str(p.relative_to(cfg.run_reports_dir)): sha256(p)
        for p in cfg.run_reports_dir.rglob("*")
        if p.is_file()
    }
    completed = list((cfg.run_data_dir / "collection_chunks").glob("*.done.json"))
    if completed:
        times = [p.stat().st_mtime for p in completed]
        payload["checkpoint_completion_times_utc"] = {
            "first": datetime.fromtimestamp(min(times), timezone.utc).isoformat(),
            "last": datetime.fromtimestamp(max(times), timezone.utc).isoformat(),
            "meaning": "Filesystem completion timestamps, not inferred collection start time",
        }
    if Path("source_provenance.json").exists():
        payload["source_bundle_provenance"] = json.loads(Path("source_provenance.json").read_text())
    payload = json.loads(json.dumps(payload, default=str))
    write_json(cfg.run_data_dir / "scientific_experiment_manifest.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["collect", "analyze", "controls", "robustness", "report", "all"],
        default="all",
    )
    parser.add_argument("--run-name", default="gemma1b_regimes_positive")
    parser.add_argument("--max-texts", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--screen-features", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=24)
    args = parser.parse_args()
    cfg = configuration(args.run_name, args.max_texts, args.max_seq_len)
    rcfg = RegimeConfig(
        permutations=args.permutations,
        screen_features=args.screen_features,
        max_candidates=args.max_candidates,
    )
    if args.stage in ("collect", "all"):
        collect(cfg)
    if args.stage in ("analyze", "all"):
        analyze(cfg, rcfg)
    if args.stage in ("controls", "all"):
        from sae_feature_atlas.scientific.controls import weak_controls

        weak_controls(cfg, rcfg)
    if args.stage in ("robustness", "all"):
        from sae_feature_atlas.scientific.robustness import one_per_document

        one_per_document(cfg, rcfg)
    if args.stage in ("report", "all"):
        from sae_feature_atlas.scientific.report import report

        report(cfg, rcfg)
        manifest(cfg, rcfg)


if __name__ == "__main__":
    main()
