from __future__ import annotations

from pathlib import Path

import pandas as pd

from sae_feature_atlas.config import ExperimentConfig
from sae_feature_atlas.visualization import generate_html_tables, generate_plots


def _shape(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return str(pd.read_parquet(path).shape)
    except Exception:
        return "unreadable"


def write_markdown_summary(cfg: ExperimentConfig) -> None:
    cfg.run_reports_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# SAE Feature Atlas report: `{cfg.collection.run_name}`", "",
        "## Setup",
        f"- Model: `{cfg.model.model_name}`",
        f"- SAE: `{cfg.model.sae_release}` / `{cfg.model.sae_id}`",
        f"- Layer/hook: `{cfg.model.layer}` / `{cfg.model.hook_name}`",
        f"- Corpus: `{cfg.collection.corpus}`",
        f"- Activation mode: `{cfg.collection.activation_mode}`",
        f"- Top-k: `{cfg.collection.top_k_features_per_token}`", "",
        "## Artifacts",
        f"- Feature cards: `{_shape(cfg.feature_cards_path)}`",
        f"- Feature stats: `{_shape(cfg.feature_stats_path)}`",
        f"- Top examples: `{_shape(cfg.top_examples_path)}`",
        f"- Coactivation pairs: `{_shape(cfg.coactivation_pairs_path)}`",
        f"- Decoder neighbors: `{_shape(cfg.decoder_neighbors_path)}`",
        f"- Geometry vs coactivation: `{_shape(cfg.geometry_vs_coactivation_path)}`",
        f"- Bimodal candidates: `{_shape(cfg.bimodal_candidates_path)}`",
        f"- Inspection feature summaries: `{_shape(cfg.inspection_feature_summaries_path)}`",
        f"- Inspection pair summaries: `{_shape(cfg.inspection_pair_summaries_path)}`", "",
        "## Interpretation caveats",
        "- Frequencies depend on activation storage mode.",
        "- In top-k mode, frequency means occurrence among saved top-k features.",
        "- Automated inspection labels are heuristic triage, not final explanations.",
        "- Bimodality scores are statistical candidates and require manual validation.", "",
    ]
    cfg.summary_md_path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(cfg: ExperimentConfig) -> None:
    cfg.run_reports_dir.mkdir(parents=True, exist_ok=True)
    plots = generate_plots(cfg.run_data_dir, cfg.run_reports_dir)
    tables = generate_html_tables(cfg.run_data_dir, cfg.run_reports_dir)
    plot_html = "\n".join(f'<section class="plot"><h3>{name}</h3><img src="{Path(path).relative_to(cfg.run_reports_dir)}"/></section>' for name, path in plots.items())
    table_html = "\n".join(f'<section><h3>{name}</h3><iframe src="{Path(path).relative_to(cfg.run_reports_dir)}" width="100%" height="440"></iframe></section>' for name, path in tables.items())
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"/><title>SAE Feature Atlas Report: {cfg.collection.run_name}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; max-width: 1180px; margin: 40px auto; padding: 0 24px; line-height: 1.55; color: #222; }}
h1 {{ font-size: 34px; margin-bottom: 4px; }} h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 36px; }}
.subtitle {{ color: #666; font-size: 16px; margin-top: 0; }} .card {{ border: 1px solid #ddd; border-radius: 12px; padding: 16px 20px; margin: 16px 0; background: #fafafa; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 16px 0; }} .metric {{ border: 1px solid #e5e5e5; border-radius: 10px; padding: 12px; background: white; }}
.metric .label {{ color: #666; font-size: 13px; }} .metric .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }} .metric .value.small {{ font-size: 14px; word-break: break-all; }}
img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 10px; margin: 8px 0 20px; background: white; }} code {{ background: #f2f2f2; padding: 2px 4px; border-radius: 4px; }} iframe {{ border: 1px solid #ddd; border-radius: 10px; background: white; }} .note {{ color: #555; font-size: 14px; }}
</style></head><body>
<h1>SAE Feature Atlas Report</h1><p class="subtitle">Pilot analysis of Gemma Scope SAE features on <code>{cfg.collection.corpus}</code></p>
<div class="card"><h2>What this report is</h2><p>This exploratory report summarizes sparse SAE activations, feature cards, co-activation, decoder geometry, activation-space/feature-space PCA, bimodality candidates, and automated artifact triage.</p><p class="note">Automated labels are heuristic. Manual inspection is still required before making semantic claims.</p></div>
<h2>Setup</h2><div class="grid">
<div class="metric"><div class="label">Model</div><div class="value small"><code>{cfg.model.model_name}</code></div></div>
<div class="metric"><div class="label">SAE</div><div class="value small"><code>{cfg.model.sae_id}</code></div></div>
<div class="metric"><div class="label">Run ID</div><div class="value small"><code>{cfg.collection.run_name}</code></div></div>
<div class="metric"><div class="label">Layer</div><div class="value">{cfg.model.layer}</div></div>
<div class="metric"><div class="label">Activation mode</div><div class="value"><code>{cfg.collection.activation_mode}</code></div></div>
<div class="metric"><div class="label">Top-k</div><div class="value">{cfg.collection.top_k_features_per_token}</div></div>
</div>
<div class="card"><h2>How to read this report</h2><ul><li>Feature frequency depends on activation storage mode.</li><li>Decoder geometry and empirical co-activation answer different questions.</li><li>Artifact scores highlight quote/punctuation/boundary/source-concentration patterns.</li><li>Bimodality candidates indicate activation distribution structure, not final semantic meaning.</li></ul></div>
<h2>Plots</h2>{plot_html}<h2>Tables</h2>{table_html}</body></html>"""
    cfg.html_report_path.write_text(html, encoding="utf-8")


def write_report(cfg: ExperimentConfig) -> None:
    write_markdown_summary(cfg)
    write_html_report(cfg)
