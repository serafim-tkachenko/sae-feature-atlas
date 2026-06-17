
from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from sae_feature_atlas.config.datasets import describe_corpus, list_supported_corpora
from sae_feature_atlas.config.registry import list_supported_models, make_config
from sae_feature_atlas.pipeline.runner import ALL_STEPS, STEP_PRESETS, normalize_steps, run_pipeline
from sae_feature_atlas.report.markdown import write_report
from sae_feature_atlas.runtime.gemma_scope import GemmaScopeRuntime

# Supported sites from where activations can be extracted
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


def cfg_from_args(args: argparse.Namespace):
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

# Dry-run to see what
def cmd_plan(args: argparse.Namespace) -> None:
    cfg = cfg_from_args(args)
    steps = normalize_steps(args.steps, preset=args.preset)
    corpus = describe_corpus(cfg.collection.corpus)
    plan = {
        "model": cfg.model.__dict__,
        "corpus": asdict(corpus),
        "collection": cfg.collection.__dict__,
        "preset": args.preset or "custom/default",
        "steps": steps,
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

    print("SAE Feature Atlas run plan")
    print("=" * 32)
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
    print("\nMentor-plan coverage if the selected steps finish:")
    coverage = {
        "0 activation dataset": "collect" in steps,
        "1 feature filtering": "features" in steps,
        "2 coactivation": "coactivation" in steps,
        "3 bimodality regimes": "bimodality" in steps,
        "4 decoder geometry vs coactivation": "geometry-vs-coactivation" in steps,
        "5 residual PCA coverage": "coverage" in steps or "space" in steps,
        "6 decoder direction PC spread": "coverage" in steps,
    }
    for item, enabled in coverage.items():
        print(f"  [{'x' if enabled else ' '}] {item}")


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

    p = sub.add_parser("plan")
    add_common_args(p)
    p.add_argument("--preset", default="research", choices=sorted(STEP_PRESETS))
    p.add_argument("--steps", default="all")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("smoke-test")
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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)
