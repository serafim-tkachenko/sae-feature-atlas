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
        f"# SAE Feature Atlas report: `{cfg.collection.run_name}`",
        "",
        "## Setup",
        "",
        f"- Model: `{cfg.model.model_name}`",
        f"- SAE release: `{cfg.model.sae_release}`",
        f"- SAE id: `{cfg.model.sae_id}`",
        f"- Layer: `{cfg.model.layer}`",
        f"- Hook: `{cfg.model.hook_name}`",
        f"- Corpus: `{cfg.collection.corpus}`",
        f"- Activation mode: `{cfg.collection.activation_mode}`",
        f"- Top-k: `{cfg.collection.top_k_features_per_token}`",
        "",
        "## Artifacts",
        "",
        f"- Token metadata: `{_shape(cfg.token_metadata_path)}`",
        f"- SAE activations: `{_shape(cfg.sae_activations_path)}`",
        f"- Feature stats: `{_shape(cfg.feature_stats_path)}`",
        f"- Filtered features: `{_shape(cfg.filtered_features_path)}`",
        f"- Feature cards: `{_shape(cfg.feature_cards_path)}`",
        f"- Coactivation pairs: `{_shape(cfg.coactivation_pairs_path)}`",
        f"- Decoder neighbors: `{_shape(cfg.decoder_neighbors_path)}`",
        f"- Geometry vs coactivation: `{_shape(cfg.geometry_vs_coactivation_path)}`",
        f"- Bimodal candidates: `{_shape(cfg.bimodal_candidates_path)}`",
        "",
        "## Notes",
        "",
        "- Frequencies depend on the activation storage mode",
        "- In `topk` mode, frequency means occurrence among saved top-K features",
        "- Custom interpretation should be added manually or by future NLA steps",
        "",
    ]

    cfg.summary_md_path.write_text("\n".join(lines), encoding="utf-8")


def write_html_report(cfg: ExperimentConfig) -> None:
    cfg.run_reports_dir.mkdir(parents=True, exist_ok=True)
    plots = generate_plots(cfg.run_data_dir, cfg.run_reports_dir)
    tables = generate_html_tables(cfg.run_data_dir, cfg.run_reports_dir)

    plot_html = "\n".join(
        f'<h3>{name}</h3><img src="{Path(path).relative_to(cfg.run_reports_dir)}" width="800"/>'
        for name, path in plots.items()
    )
    table_html = "\n".join(
        f'<h3>{name}</h3><iframe src="{Path(path).relative_to(cfg.run_reports_dir)}" width="100%" height="420"></iframe>'
        for name, path in tables.items()
    )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>SAE Feature Atlas: {cfg.collection.run_name}</title>
</head>
<body>
  <h1>SAE Feature Atlas: {cfg.collection.run_name}</h1>
  <h2>Setup</h2>
  <ul>
    <li>Model: <code>{cfg.model.model_name}</code></li>
    <li>SAE: <code>{cfg.model.sae_release}</code> / <code>{cfg.model.sae_id}</code></li>
    <li>Hook: <code>{cfg.model.hook_name}</code></li>
    <li>Corpus: <code>{cfg.collection.corpus}</code></li>
    <li>Activation mode: <code>{cfg.collection.activation_mode}</code></li>
    <li>Top-k: <code>{cfg.collection.top_k_features_per_token}</code></li>
  </ul>
  <h2>Plots</h2>
  {plot_html}
  <h2>Tables</h2>
  {table_html}
</body>
</html>
"""
    cfg.html_report_path.write_text(html, encoding="utf-8")


def write_report(cfg: ExperimentConfig) -> None:
    write_markdown_summary(cfg)
    write_html_report(cfg)
