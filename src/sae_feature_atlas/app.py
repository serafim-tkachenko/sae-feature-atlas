from __future__ import annotations

import argparse
import json

from sae_feature_atlas.bundle import GemmaScopeBundle
from sae_feature_atlas.pipeline import ALL_STEPS, run_pipeline
from sae_feature_atlas.registry import list_supported_models, make_config
from sae_feature_atlas.reporting import write_report


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="gemma-3-1b-pt")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--site", default="resid_post", choices=["resid_post", "res", "mlp_out", "mlp", "attn_out", "att"])
    parser.add_argument("--width", default="16k")
    parser.add_argument("--l0", default=None)
    parser.add_argument("--corpus", default="pile-10k")
    parser.add_argument("--max-texts", type=int, default=1500)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--activation-mode", default="topk", choices=["topk", "positive"])
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--run-name", default=None)


def cfg_from_args(args: argparse.Namespace):
    return make_config(model=args.model, layer=args.layer, site=args.site, width=args.width, l0=args.l0, corpus=args.corpus, max_texts=args.max_texts, max_seq_len=args.max_seq_len, activation_mode=args.activation_mode, top_k=args.top_k, run_name=args.run_name)


def cmd_list_models(_: argparse.Namespace) -> None:
    for model in list_supported_models(): print(model)


def cmd_resolve_sae(args: argparse.Namespace) -> None:
    print(json.dumps(cfg_from_args(args).model.__dict__, indent=2))


def cmd_smoke_test(args: argparse.Namespace) -> None:
    print(json.dumps(GemmaScopeBundle(cfg_from_args(args)).load().validate(), indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    print(json.dumps(run_pipeline(cfg_from_args(args), steps=args.steps), indent=2))


def cmd_report(args: argparse.Namespace) -> None:
    cfg = cfg_from_args(args); write_report(cfg); print("Wrote", cfg.summary_md_path); print("Wrote", cfg.html_report_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sae-atlas")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list-models"); p.set_defaults(func=cmd_list_models)
    p = sub.add_parser("resolve-sae"); add_common_args(p); p.set_defaults(func=cmd_resolve_sae)
    p = sub.add_parser("smoke-test"); add_common_args(p); p.set_defaults(func=cmd_smoke_test)
    p = sub.add_parser("run"); add_common_args(p); p.add_argument("--steps", default="all", help=f"Comma-separated steps or all. Available: {','.join(ALL_STEPS)}"); p.set_defaults(func=cmd_run)
    p = sub.add_parser("report"); add_common_args(p); p.set_defaults(func=cmd_report)
    return parser


def main() -> None:
    args = build_parser().parse_args(); args.func(args)
