
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

import pandas as pd

from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.report.plots import generate_html_tables, generate_plots


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def _shape(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        return str(pd.read_parquet(path).shape)
    except Exception:
        return "unreadable"


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str], n: int = 10) -> str:
    if df is None or df.empty:
        return "_No rows available._"
    keep = [col for col in columns if col in df.columns]
    if not keep:
        return "_No requested columns available._"
    preview = df[keep].head(n).copy()
    header = "| " + " | ".join(keep) + " |"
    separator = "| " + " | ".join(["---"] * len(keep)) + " |"
    rows = []
    for _, row in preview.iterrows():
        cells = []
        for col in keep:
            text = _fmt(row[col]).replace("\n", " ").replace("|", "\\|")
            if len(text) > 120:
                text = text[:117] + "..."
            cells.append(text)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _count(df: pd.DataFrame, column: str, value: str) -> int:
    if df.empty or column not in df.columns:
        return 0
    return int(df[column].astype(str).eq(value).sum())


def _value_counts_lines(df: pd.DataFrame, column: str, n: int = 12) -> list[str]:
    if df.empty or column not in df.columns:
        return ["- _No values available._"]
    return [f"- `{idx}`: {int(value)}" for idx, value in df[column].value_counts(dropna=False).head(n).items()]


def _artifact_line(label: str, path: Path, note: str) -> str:
    return f"- `{path.name}` — **{label}**, shape/status: `{_shape(path)}`. {note}"


def _mentor_checklist(cfg: ExperimentConfig) -> list[str]:
    checks = [
        ("0", "Activation dataset", cfg.sae_activations_path, "SAE activations and token metadata are saved for notebook access."),
        ("1", "Feature filtering", cfg.filtered_features_path, "Rare/over-common features are filtered before pair/geometry analysis."),
        ("2", "Coactivation", cfg.coactivation_pairs_path, "Same-token feature coactivation pairs are computed."),
        ("3", "Bimodal activation regimes", cfg.bimodal_peak_examples_path, "Low/high activation-regime examples are available for candidate features."),
        ("4", "Decoder geometry vs coactivation", cfg.geometry_vs_coactivation_path, "Nearest decoder directions are compared with empirical coactivation."),
        ("5", "Decoder directions vs residual PCA", cfg.feature_coverage_profiles_path, "SAE decoder directions are projected onto sampled residual PCA components."),
        ("6", "Decoder direction PC spread", cfg.feature_coverage_profiles_path, "Effective PC dimension / entropy describe how spread decoder directions are.")]
    lines = []
    for idx, title, path, note in checks:
        status = "done" if path.exists() else "missing"
        lines.append(f"- [{status}] **{idx}. {title}** — `{path.name}`. {note}")
    return lines


METRIC_GLOSSARY: list[tuple[str, str]] = [
    ("token_frequency", "How often a feature appears among saved activations; in top-k mode this is frequency among stored top-k rows."),
    ("coactivation_jaccard", "Same-token feature overlap normalized by union count."),
    ("pmi", "Pointwise mutual information for feature-pair coactivation; sensitive to rare pairs."),
    ("bimodality_score", "BIC improvement of a two-component activation distribution over a one-component fit."),
    ("decoder_cosine", "Cosine similarity between SAE decoder directions; not proof of semantic equivalence."),
    ("effective_pc_dim", "Participation-ratio estimate of how many residual PCA components carry a decoder direction."),
    ("pc_center_of_mass", "Average residual-PC index weighted by squared decoder projection mass."),
    ("gca_at_k", "Overlap between top-k decoder-neighbor and top-k coactivation-neighbor sets.")]


def write_markdown_summary(cfg: ExperimentConfig) -> None:
    cfg.run_reports_dir.mkdir(parents=True, exist_ok=True)

    cards = _read(cfg.feature_cards_path)
    feature_stats = _read(cfg.feature_stats_path)
    coactivation = _read(cfg.coactivation_pairs_path)
    geometry = _read(cfg.geometry_vs_coactivation_path)
    bimodal = _read(cfg.bimodal_candidates_path)
    regimes = _read(cfg.bimodal_peak_examples_path)
    coverage = _read(cfg.feature_coverage_profiles_path)
    alignment = _read(cfg.graph_alignment_path)

    top_bimodal = bimodal.sort_values("bimodality_score", ascending=False) if "bimodality_score" in bimodal.columns else bimodal

    lines: list[str] = [
        f"# SAE Feature Atlas report: `{cfg.collection.run_name}`",
        "",
        "This report is organized around the mentor research plan: collect SAE activations, filter features, study coactivation, inspect bimodal activation regimes, compare decoder geometry with empirical coactivation, and relate SAE decoder directions to residual PCA structure.",
        "",
        "## 1. Run overview",
        "",
        f"- Model: `{cfg.model.model_name}`",
        f"- SAE: `{cfg.model.sae_release}` / `{cfg.model.sae_id}`",
        f"- Layer / hook: `{cfg.model.layer}` / `{cfg.model.hook_name}`",
        f"- Corpus: `{cfg.collection.corpus}`",
        f"- Activation mode: `{cfg.collection.activation_mode}`",
        f"- Top-k: `{cfg.collection.top_k_features_per_token}`",
        f"- Max texts / seq len: `{cfg.collection.max_texts}` / `{cfg.collection.max_seq_len}`",
        "",
        "## 2. Mentor-plan checklist",
        "",
        *_mentor_checklist(cfg),
        "",
        "## 3. Core activation dataset",
        "",
        _artifact_line("token metadata", cfg.token_metadata_path, "Token-level source/position metadata."),
        _artifact_line("SAE activations", cfg.sae_activations_path, "Sparse SAE activation rows used by downstream analyses."),
        _artifact_line("residual sample", cfg.residual_vectors_path, "Sampled residual vectors for PCA/coverage diagnostics."),
        "",
        "## 4. Feature filtering and population",
        "",
        f"- Feature stats: `{_shape(cfg.feature_stats_path)}`",
        f"- Filtered features: `{_shape(cfg.filtered_features_path)}`",
        f"- Feature cards: `{_shape(cfg.feature_cards_path)}`",
        "",
        "Primary labels:",
        "",
        *_value_counts_lines(cards, "primary_label"),
        "",
        _markdown_table(cards, ["feature_id", "primary_label", "manual_priority", "token_frequency", "n_token_activations", "n_texts", "p99_activation", "artifact_score", "semantic_score"], n=15),
        "",
        "## 5. Same-token coactivation",
        "",
        "Coactivation is computed on the same token positions after activation-row and feature filters. It is empirical usage overlap, not causal evidence.",
        "",
        f"- Coactivation pairs: `{_shape(cfg.coactivation_pairs_path)}`",
        "",
        _markdown_table(coactivation, ["feature_i", "feature_j", "coactivation_count", "jaccard", "pmi", "p_j_given_i", "p_i_given_j"], n=15),
        "",
        "## 6. Bimodal activation regimes",
        "",
        "This section addresses whether a feature has weak/high activation regimes and shows examples from both regimes for manual interpretation.",
        "",
        f"- Bimodal candidates: `{_shape(cfg.bimodal_candidates_path)}`",
        f"- Low/high regime examples: `{_shape(cfg.bimodal_peak_examples_path)}`",
        "",
        _markdown_table(top_bimodal, ["feature_id", "n_points", "bimodality_score", "log_mean_low", "log_mean_high", "activation_p50", "activation_p95", "activation_max"], n=15),
        "",
        "Example low/high rows:",
        "",
        _markdown_table(regimes, ["feature_id", "peak_label", "activation", "source", "text_id", "token_pos", "left_context", "center_token", "right_context"], n=12),
        "",
        "## 7. Decoder geometry vs empirical coactivation",
        "",
        "Decoder-neighbor geometry and empirical coactivation are different signals. Agreement is useful; disagreement is often scientifically interesting.",
        "",
        f"- Geometry/coactivation pairs: `{_shape(cfg.geometry_vs_coactivation_path)}`",
        "",
        _markdown_table(geometry, ["feature_i", "feature_j", "decoder_cosine", "jaccard", "pmi", "geometry_coactivation_quadrant"], n=15),
        "",
        "## 8. Residual PCA coverage",
        "",
        "Residual activations are original model residual-stream vectors. PCA coordinates are a diagnostic basis fitted on sampled residual vectors. Decoder coverage asks where SAE decoder directions lie relative to that PCA basis.",
        "",
        f"- Coverage profiles: `{_shape(cfg.feature_coverage_profiles_path)}`",
        "",
        "Coverage buckets:",
        "",
        *_value_counts_lines(coverage, "coverage_bucket"),
        "",
        _markdown_table(coverage, ["feature_id", "pc_mass_observed", "pc_mass_unobserved_tail", "effective_pc_dim", "pc_entropy", "pc_center_of_mass", "pc_norm_mass_top_1", "pc_norm_mass_top_5", "pc_norm_mass_top_20", "coverage_bucket"], n=15),
        "",
        "## 9. Graph alignment",
        "",
        "These are research-extension artifacts.",
        "",
        "Graph-alignment buckets:",
        "",
        *_value_counts_lines(alignment, "graph_alignment_bucket"),
        "",
        "",
        "## 10. Interpretation caveats",
        "",
        "- Feature cards are multi-evidence profiles, not final explanations.",
        "- Automated labels are heuristic triage, not ground-truth semantics.",
        "- In `topk` mode, feature frequency means occurrence among stored top-k activations, not true positive activation frequency.",
        "- Bimodality is a statistical candidate signal; low/high examples require manual interpretation.",
        "- Decoder cosine does not prove semantic similarity or causal interaction.",
        "- Residual PCA coverage depends on sampled corpus, layer, and number of PCA components.",
        "",
        "## 11. Metric glossary",
        "",
        *[f"- `{name}`: {description}" for name, description in METRIC_GLOSSARY],
        ""]
    cfg.summary_md_path.write_text("\n".join(lines), encoding="utf-8")


def _relative(report_dir: Path, path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(report_dir))
    except Exception:
        return str(path)


def _html_table(df: pd.DataFrame, columns: list[str], n: int = 12) -> str:
    if df.empty:
        return "<p><em>No rows available.</em></p>"
    keep = [col for col in columns if col in df.columns]
    if not keep:
        return "<p><em>No requested columns available.</em></p>"
    headers = "".join(f"<th>{escape(col)}</th>" for col in keep)
    rows = []
    for _, row in df[keep].head(n).iterrows():
        cells = "".join(f"<td>{escape(_fmt(row[col]))}</td>" for col in keep)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_html_report(cfg: ExperimentConfig) -> None:
    cfg.run_reports_dir.mkdir(parents=True, exist_ok=True)
    plots = generate_plots(cfg.run_data_dir, cfg.run_reports_dir)
    tables = generate_html_tables(cfg.run_data_dir, cfg.run_reports_dir)

    cards = _read(cfg.feature_cards_path)
    regimes = _read(cfg.bimodal_peak_examples_path)
    coverage = _read(cfg.feature_coverage_profiles_path)

    plot_links = "".join(
        f'<li><a href="{escape(_relative(cfg.run_reports_dir, path))}">{escape(name)}</a></li>'
        for name, path in sorted(plots.items())
    )
    table_links = "".join(
        f'<li><a href="{escape(_relative(cfg.run_reports_dir, path))}">{escape(name)}</a></li>'
        for name, path in sorted(tables.items())
    )
    checklist = "".join(f"<li>{escape(line)}</li>" for line in _mentor_checklist(cfg))

    html = f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>SAE Feature Atlas: {escape(cfg.collection.run_name)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; line-height: 1.45; max-width: 1200px; }}
    code {{ background: #f4f4f4; padding: 0.1rem 0.25rem; border-radius: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.35rem 0.5rem; vertical-align: top; }}
    th {{ background: #f7f7f7; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem; background: #fafafa; }}
  </style>
</head>
<body>
  <h1>SAE Feature Atlas Report</h1>
  <p><code>{escape(cfg.collection.run_name)}</code></p>
  <div class="grid">
    <div class="card"><strong>Model</strong><br>{escape(cfg.model.model_name)}</div>
    <div class="card"><strong>Layer / hook</strong><br>{escape(str(cfg.model.layer))} / {escape(cfg.model.hook_name)}</div>
    <div class="card"><strong>Corpus</strong><br>{escape(cfg.collection.corpus)}</div>
    <div class="card"><strong>Feature cards</strong><br>{len(cards):,}</div>
  </div>

  <h2>Mentor-plan checklist</h2>
  <ul>{checklist}</ul>

  <h2>Feature population</h2>
  {_html_table(cards, ["feature_id", "primary_label", "manual_priority", "token_frequency", "p99_activation", "artifact_score", "semantic_score", "bimodality_score"], n=16)}

  <h2>Bimodal low/high activation-regime examples</h2>
  <p>These rows are designed for manual comparison of weak vs strong feature activation contexts.</p>
  {_html_table(regimes, ["feature_id", "peak_label", "activation", "source", "text_id", "token_pos", "left_context", "center_token", "right_context"], n=16)}

  <h2>Residual PCA coverage</h2>
  {_html_table(coverage, ["feature_id", "effective_pc_dim", "pc_entropy", "pc_center_of_mass", "pc_norm_mass_top_20", "coverage_bucket"], n=16)}

  <h2>Generated plots</h2>
  <ul>{plot_links or '<li>No plots generated.</li>'}</ul>

  <h2>Generated table previews</h2>
  <ul>{table_links or '<li>No tables generated.</li>'}</ul>

  <h2>Interpretation caveats</h2>
  <ul>
    <li>Feature cards are multi-evidence profiles, not final explanations.</li>
    <li>Bimodality is a candidate signal; low/high examples require manual review.</li>
    <li>Decoder geometry and coactivation are related but distinct signals.</li>
  </ul>
</body>
</html>
"""
    cfg.html_report_path.write_text(html, encoding="utf-8")


def write_report(cfg: ExperimentConfig) -> None:
    write_markdown_summary(cfg)
    write_html_report(cfg)
