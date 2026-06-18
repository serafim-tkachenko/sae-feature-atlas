from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from sae_feature_atlas.config.datasets import describe_corpus, list_supported_corpora
from sae_feature_atlas.config.registry import list_supported_models, make_config
from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.inspection.commands import (
    print_bimodal_feature_examples,
    print_feature_examples,
    print_pair_examples,
)
from sae_feature_atlas.pipeline.runner import run_pipeline
from sae_feature_atlas.pipeline.steps import ALL_STEPS, STEP_PRESETS, normalize_steps
from sae_feature_atlas.report.markdown import write_report
from sae_feature_atlas.runtime.gemma_scope import GemmaScopeRuntime

# Short aliases are accepted by the CLI and normalized in config.registry
SUPPORTED_SITES = ["resid_post", "res", "mlp_out", "mlp", "attn_out", "att"]


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gemma-3-1b-pt")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--site", default="resid_post", choices=SUPPORTED_SITES)
    parser.add_argument("--width", default="16k")
    parser.add_argument("--l0", default=None)
    parser.add_argument("--corpus", default="pile-10k")
    parser.add_argument("--max-texts", type=int, default=1500)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--activation-mode", default="topk", choices=["topk", "positive"])
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--run-name", default=None)


def cfg_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return make_config(
        model=args.model,
        layer=args.layer,
        site=args.site,
        width=args.width,
        l0=args.l0,
        corpus=args.corpus,
        max_texts=args.max_texts,
        max_seq_len=args.max_seq_len,
        activation_mode=args.activation_mode,
        top_k=args.top_k,
        run_name=args.run_name,
    )


def _exists(path: Path) -> str:
    return "exists" if path.exists() else "missing"


def _step_artifacts(cfg: ExperimentConfig) -> dict[str, dict[str, list[Path]]]:
    return {
        "collect": {
            "inputs": [],
            "outputs": [
                cfg.token_metadata_path,
                cfg.sae_activations_path,
                cfg.token_activation_summary_path,
                cfg.residual_vectors_path,
                cfg.residual_metadata_path,
            ],
        },
        "features": {
            "inputs": [cfg.sae_activations_path, cfg.token_metadata_path],
            "outputs": [cfg.feature_stats_path, cfg.filtered_features_path, cfg.top_examples_path, cfg.feature_cards_path],
        },
        "coactivation": {
            "inputs": [cfg.sae_activations_path, cfg.filtered_features_path],
            "outputs": [cfg.coactivation_pairs_path],
        },
        "geometry": {
            "inputs": [cfg.filtered_features_path],
            "outputs": [cfg.decoder_neighbors_path],
        },
        "geometry-vs-coactivation": {
            "inputs": [cfg.decoder_neighbors_path, cfg.coactivation_pairs_path],
            "outputs": [cfg.geometry_vs_coactivation_path],
        },
        "bimodality": {
            "inputs": [cfg.sae_activations_path, cfg.token_metadata_path],
            "outputs": [cfg.bimodal_candidates_path, cfg.bimodal_peak_examples_path],
        },
        "inspection": {
            "inputs": [cfg.sae_activations_path, cfg.top_examples_path, cfg.filtered_features_path],
            "outputs": [cfg.inspection_feature_summaries_path, cfg.inspection_pair_summaries_path, cfg.inspection_report_md_path],
        },
        "space": {
            "inputs": [cfg.residual_vectors_path, cfg.feature_stats_path],
            "outputs": [cfg.residual_pca_summary_path, cfg.decoder_pca_summary_path, cfg.decoder_feature_pca_path, cfg.decoder_feature_umap_path],
        },
        "coverage": {
            "inputs": [cfg.residual_vectors_path, cfg.filtered_features_path],
            "outputs": [cfg.feature_coverage_profiles_path],
        },
        "alignment": {
            "inputs": [cfg.decoder_neighbors_path, cfg.coactivation_pairs_path, cfg.filtered_features_path],
            "outputs": [cfg.graph_alignment_path, cfg.graph_alignment_summary_path],
        },
        "cards": {
            "inputs": [cfg.filtered_features_path],
            "outputs": [cfg.feature_cards_path],
        },
        "report": {
            "inputs": [cfg.feature_cards_path],
            "outputs": [cfg.summary_md_path, cfg.html_report_path],
        },
    }


def _plan_artifact_status(cfg: ExperimentConfig, steps: list[str]) -> list[dict[str, str]]:
    artifacts = _step_artifacts(cfg)
    rows: list[dict[str, str]] = []
    produced_by_plan: set[Path] = set()
    for step in steps:
        spec = artifacts.get(step, {"inputs": [], "outputs": []})
        for path in spec["inputs"]:
            available = path.exists() or path in produced_by_plan
            rows.append(
                {
                    "step": step,
                    "kind": "input",
                    "status": "ready" if available else "missing",
                    "path": str(path),
                }
            )
        for path in spec["outputs"]:
            rows.append(
                {
                    "step": step,
                    "kind": "output",
                    "status": _exists(path),
                    "path": str(path),
                }
            )
            produced_by_plan.add(path)
    return rows


def _analysis_checklist(steps: list[str]) -> dict[str, bool]:
    return {
        "activation dataset": "collect" in steps,
        "feature filtering and top examples": "features" in steps,
        "same-token coactivation": "coactivation" in steps,
        "decoder geometry": "geometry" in steps,
        "geometry vs coactivation": "geometry-vs-coactivation" in steps,
        "bimodal activation-regime examples": "bimodality" in steps,
        "automated inspection summaries": "inspection" in steps,
        "residual/decoder PCA and decoder UMAP": "space" in steps,
        "decoder directions vs residual PCA coverage": "coverage" in steps,
        "decoder-neighbor vs coactivation-neighbor alignment": "alignment" in steps,
        "feature cards": "cards" in steps,
        "markdown/html report": "report" in steps,
    }


def cmd_list_models(_: argparse.Namespace) -> None:
    for model in list_supported_models():
        print(model)


def cmd_list_sites(_: argparse.Namespace) -> None:
    for site in SUPPORTED_SITES:
        print(site)


def cmd_list_corpora(args: argparse.Namespace) -> None:
    for name in list_supported_corpora():
        desc = describe_corpus(name)
        if args.json:
            print(json.dumps(asdict(desc), ensure_ascii=False))
        else:
            domains = ", ".join(desc.domains)
            hf = desc.hf_dataset or "manual/custom"
            if desc.hf_config:
                hf += f" ({desc.hf_config})"
            print(f"{name:22} {desc.size:16} {hf}")
            print(f"  domains: {domains}")
            print(f"  {desc.description}")
            if desc.notes:
                print(f"  note: {desc.notes}")


def cmd_list_presets(_: argparse.Namespace) -> None:
    for name, steps in STEP_PRESETS.items():
        print(f"{name}: {','.join(steps)}")


def cmd_resolve_sae(args: argparse.Namespace) -> None:
    print(json.dumps(cfg_from_args(args).model.__dict__, indent=2))


def cmd_plan(args: argparse.Namespace) -> None:
    cfg = cfg_from_args(args)
    steps = normalize_steps(args.steps, preset=args.preset)
    corpus = describe_corpus(cfg.collection.corpus)
    artifact_status = _plan_artifact_status(cfg, steps)
    checklist = _analysis_checklist(steps)
    plan = {
        "model": cfg.model.__dict__,
        "corpus": asdict(corpus),
        "collection": cfg.collection.__dict__,
        "preset": args.preset or "custom/default",
        "steps": steps,
        "analysis_checklist": checklist,
        "artifacts": artifact_status,
        "outputs": {
            "data_dir": str(cfg.run_data_dir),
            "reports_dir": str(cfg.run_reports_dir),
            "feature_cards": str(cfg.feature_cards_path),
            "summary_md": str(cfg.summary_md_path),
            "html_report": str(cfg.html_report_path),
        },
    }
    if args.json:
        print(json.dumps(plan, indent=2, default=str, ensure_ascii=False))
        return

    print("SAE Feature Atlas preflight plan")
    print("=" * 36)
    print(f"Model:      {cfg.model.model_name}")
    print(f"Layer/hook: {cfg.model.layer} / {cfg.model.hook_name}")
    print(f"SAE:        {cfg.model.sae_release} / {cfg.model.sae_id}")
    print(f"Corpus:     {cfg.collection.corpus} ({corpus.size})")
    print(f"Domains:    {', '.join(corpus.domains)}")
    print(f"Texts/seq:  {cfg.collection.max_texts} texts, max_seq_len={cfg.collection.max_seq_len}")
    print(f"Storage:    {cfg.collection.activation_mode}, top_k={cfg.collection.top_k_features_per_token}")
    print(f"Preset:     {args.preset or 'custom/default'}")
    print("Steps:      " + " -> ".join(steps))
    print("Outputs:")
    print(f"  data:     {cfg.run_data_dir}")
    print(f"  report:   {cfg.summary_md_path}")
    print(f"  html:     {cfg.html_report_path}")

    print("\nAnalysis checklist if selected steps finish:")
    for item, enabled in checklist.items():
        print(f"  [{'x' if enabled else ' '}] {item}")

    print("\nArtifact preflight:")
    for row in artifact_status:
        marker = "✓" if row["status"] in {"ready", "exists"} else "!"
        print(f"  {marker} {row['step']:24} {row['kind']:6} {row['status']:8} {row['path']}")


def cmd_smoke_test(args: argparse.Namespace) -> None:
    print(json.dumps(GemmaScopeRuntime(cfg_from_args(args)).load().validate(), indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    print(
        json.dumps(
            run_pipeline(cfg_from_args(args), steps=args.steps, preset=args.preset),
            indent=2,
        )
    )


def cmd_report(args: argparse.Namespace) -> None:
    cfg = cfg_from_args(args)
    write_report(cfg)
    print("Wrote", cfg.summary_md_path)
    print("Wrote", cfg.html_report_path)


def cmd_inspect_feature(args: argparse.Namespace) -> None:
    print_feature_examples(cfg_from_args(args), feature_id=args.feature_id, n=args.n)


def cmd_inspect_pair(args: argparse.Namespace) -> None:
    print_pair_examples(
        cfg_from_args(args),
        feature_i=args.feature_i,
        feature_j=args.feature_j,
        n=args.n,
    )


def cmd_inspect_bimodal_feature(args: argparse.Namespace) -> None:
    print_bimodal_feature_examples(cfg_from_args(args), feature_id=args.feature_id, n=args.n)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sae-atlas")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-models")
    p.set_defaults(func=cmd_list_models)

    p = sub.add_parser("list-sites")
    p.set_defaults(func=cmd_list_sites)

    p = sub.add_parser("list-corpora")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_list_corpora)

    p = sub.add_parser("list-presets")
    p.set_defaults(func=cmd_list_presets)

    p = sub.add_parser("resolve-sae")
    add_common_args(p)
    p.set_defaults(func=cmd_resolve_sae)

    p = sub.add_parser("plan", help="Preview resolved config, selected steps, and artifact status.")
    add_common_args(p)
    p.add_argument("--preset", default="research", choices=sorted(STEP_PRESETS))
    p.add_argument("--steps", default="all")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("smoke-test", help="Load and validate the selected model/SAE without running analysis.")
    add_common_args(p)
    p.set_defaults(func=cmd_smoke_test)

    p = sub.add_parser("run")
    add_common_args(p)
    p.add_argument(
        "--preset",
        default=None,
        choices=sorted(STEP_PRESETS),
        help="Pipeline preset: core, atlas, or research.",
    )
    p.add_argument(
        "--steps",
        default="all",
        help=(
            "Comma-separated steps or `all`. You can also pass a preset name. "
            f"Available steps: {','.join(ALL_STEPS)}"
        ),
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("report")
    add_common_args(p)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("inspect-feature", help="Print top activation examples for one feature.")
    add_common_args(p)
    p.add_argument("--feature-id", type=int, required=True)
    p.add_argument("--n", type=int, default=10)
    p.set_defaults(func=cmd_inspect_feature)

    p = sub.add_parser("inspect-pair", help="Print diagnostics and examples for a feature pair.")
    add_common_args(p)
    p.add_argument("--feature-i", type=int, required=True)
    p.add_argument("--feature-j", type=int, required=True)
    p.add_argument("--n", type=int, default=10)
    p.set_defaults(func=cmd_inspect_pair)

    p = sub.add_parser("inspect-bimodal-feature", help="Print low/median/high activation examples for one feature.")
    add_common_args(p)
    p.add_argument("--feature-id", type=int, required=True)
    p.add_argument("--n", type=int, default=6)
    p.set_defaults(func=cmd_inspect_bimodal_feature)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
