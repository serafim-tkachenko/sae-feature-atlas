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
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 1180px;
      margin: 40px auto;
      padding: 0 24px;
      line-height: 1.55;
      color: #222;
      background: #fff;
    }}
    h1 {{
      font-size: 34px;
      margin-bottom: 4px;
    }}
    h2 {{
      border-bottom: 1px solid #ddd;
      padding-bottom: 6px;
      margin-top: 36px;
    }}
    h3 {{
      margin-top: 24px;
    }}
    .subtitle {{
      color: #666;
      font-size: 16px;
      margin-top: 0;
    }}
    .card {{
      border: 1px solid #ddd;
      border-radius: 12px;
      padding: 16px 20px;
      margin: 16px 0;
      background: #fafafa;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin: 16px 0;
    }}
    .metric {{
      border: 1px solid #e5e5e5;
      border-radius: 10px;
      padding: 12px;
      background: white;
    }}
    .metric .label {{
      color: #666;
      font-size: 13px;
    }}
    .metric .value {{
      font-size: 22px;
      font-weight: 700;
      margin-top: 4px;
    }}
    img {{
      max-width: 100%;
      border: 1px solid #ddd;
      border-radius: 10px;
      margin: 8px 0 20px;
      background: white;
    }}
    code {{
      background: #f2f2f2;
      padding: 2px 4px;
      border-radius: 4px;
    }}
    iframe {{
      border: 1px solid #ddd;
      border-radius: 10px;
      background: white;
    }}
    .note {{
      color: #555;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <h1>SAE Feature Atlas</h1>
  <p class="subtitle"><code>{cfg.collection.run_name}</code></p>

  <div class="card">
    <h2>What this report is</h2>
    <p>
      This is an exploratory SAE feature-atlas report. It summarizes a pilot run
      over a corpus, extracts sparse SAE activations, builds feature cards, and
      computes first-pass analyses such as co-activation, decoder-neighbor geometry,
      and bimodality candidates.
    </p>
    <p class="note">
      This report is not a final interpretation. Feature explanations should be
      validated by manual inspection or future NLA / SAEExplainer steps.
    </p>
  </div>

  <h2>Setup</h2>
  <div class="grid">
    <div class="metric"><div class="label">Model</div><div class="value"><code>{cfg.model.model_name}</code></div></div>
    <div class="metric"><div class="label">Layer</div><div class="value">{cfg.model.layer}</div></div>
    <div class="metric"><div class="label">Corpus</div><div class="value"><code>{cfg.collection.corpus}</code></div></div>
    <div class="metric"><div class="label">Activation mode</div><div class="value"><code>{cfg.collection.activation_mode}</code></div></div>
    <div class="metric"><div class="label">Top-k</div><div class="value">{cfg.collection.top_k_features_per_token}</div></div>
    <div class="metric"><div class="label">SAE</div><div class="value"><code>{cfg.model.sae_id}</code></div></div>
  </div>

  <div class="card">
    <h2>How to read this report</h2>
    <ul>
      <li>Feature frequency depends on the activation storage mode.</li>
      <li>In <code>topk</code> mode, frequency means appearance among the saved top-k activations, not true positive activation frequency.</li>
      <li>Decoder-neighbor geometry compares SAE decoder-vector cosine similarity.</li>
      <li>Co-activation measures empirical same-token overlap in the selected corpus.</li>
      <li>Bimodality candidates are statistical hints, not confirmed semantic claims.</li>
    </ul>
  </div>

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
