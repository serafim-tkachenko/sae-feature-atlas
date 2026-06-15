from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FIGSIZE = (8.8, 5.2)
DPI = 150


def _finalize(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _hist(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 50) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    values = pd.to_numeric(series, errors="coerce").dropna()
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(alpha=0.18)
    _finalize(fig, path)


def _scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    work = df.dropna(subset=[x, y])
    ax.scatter(work[x], work[y], s=9, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.18)
    _finalize(fig, path)


def _scatter_by_value(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    work = df.dropna(subset=[x, y, value])
    sc = ax.scatter(work[x], work[y], c=work[value], s=9, alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.16)
    fig.colorbar(sc, ax=ax, label=value)
    _finalize(fig, path)


def _scatter_by_category(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    work = df.dropna(subset=[x, y, category])
    for label, group in work.groupby(category):
        ax.scatter(group[x], group[y], s=9, alpha=0.55, label=str(label)[:40])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.16)
    if work[category].nunique() <= 12:
        ax.legend(fontsize=7, markerscale=2, frameon=True)
    _finalize(fig, path)


def _bar(counts: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111)
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="y", alpha=0.18)
    _finalize(fig, path)


def _write_table_page(title: str, description: str, df: pd.DataFrame, path: Path) -> None:
    html_table = df.to_html(index=False, escape=True, classes="data-table")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f4f7fb; color: #172033; }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 28px 18px 64px; }}
    header {{ background: #111827; color: white; border-radius: 18px; padding: 22px 26px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px; }}
    header p {{ margin: 0; color: #d1d5db; }}
    .table-wrap {{ overflow-x: auto; background: white; border: 1px solid #dbe4ef; border-radius: 14px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f8fafc; color: #334155; font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }}
    code {{ background: #eaf1ff; padding: 2px 5px; border-radius: 6px; }}
  </style>
</head>
<body>
<main>
  <header><h1>{title}</h1><p>{description}</p></header>
  <div class="table-wrap">{html_table}</div>
</main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def generate_plots(run_data_dir: Path, report_dir: Path) -> dict[str, str]:
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    def add(name: str, path: Path) -> None:
        generated[name] = str(path)

    p = run_data_dir / "feature_stats.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            out = plots_dir / "feature_frequency_hist.png"
            _hist(df["token_frequency"], "Feature token frequency", "token frequency", out)
            add("feature_frequency_hist", out)

            out = plots_dir / "activation_p99_hist.png"
            _hist(df["p99_activation"], "Feature p99 activation", "p99 activation", out)
            add("activation_p99_hist", out)

    p = run_data_dir / "token_activation_summary.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and "n_positive_features" in df.columns:
            out = plots_dir / "positive_features_per_token_hist.png"
            _hist(df["n_positive_features"], "Positive SAE features per token", "n positive features", out)
            add("positive_features_per_token_hist", out)

    p = run_data_dir / "geometry_vs_coactivation.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            if {"decoder_cosine", "jaccard"}.issubset(df.columns):
                out = plots_dir / "geometry_vs_coactivation_scatter.png"
                _scatter(df, "decoder_cosine", "jaccard", "Decoder cosine vs coactivation", "decoder cosine", "Jaccard", out)
                add("geometry_vs_coactivation_scatter", out)
            if "quadrant" in df.columns:
                out = plots_dir / "geometry_quadrants_bar.png"
                _bar(df["quadrant"].value_counts(), "Geometry/coactivation quadrants", "pairs", out)
                add("geometry_quadrants_bar", out)

    p = run_data_dir / "bimodal_feature_candidates.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and "bimodality_score" in df.columns:
            out = plots_dir / "bimodality_score_hist.png"
            _hist(df["bimodality_score"], "Bimodality score", "BIC improvement", out)
            add("bimodality_score_hist", out)

    p = run_data_dir / "inspection_feature_summaries.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and "artifact_score" in df.columns:
            out = plots_dir / "artifact_score_hist.png"
            _hist(df["artifact_score"], "Automated artifact score", "artifact score", out)
            add("artifact_score_hist", out)

    p = run_data_dir / "residual_pca_summary.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and {"component", "explained_variance_ratio"}.issubset(df.columns):
            out = plots_dir / "residual_pca_variance.png"
            _bar(df.set_index("component")["explained_variance_ratio"], "Residual PCA explained variance", "variance ratio", out)
            add("residual_pca_variance", out)

    p = run_data_dir / "decoder_pca_summary.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and {"component", "explained_variance_ratio"}.issubset(df.columns):
            out = plots_dir / "decoder_pca_variance.png"
            _bar(df.set_index("component")["explained_variance_ratio"], "Decoder PCA explained variance", "variance ratio", out)
            add("decoder_pca_variance", out)

    p = run_data_dir / "decoder_feature_pca.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and {"decoder_pc1", "decoder_pc2"}.issubset(df.columns):
            out = plots_dir / "decoder_feature_pca_scatter.png"
            _scatter(df.head(5000), "decoder_pc1", "decoder_pc2", "Decoder feature PCA", "PC1", "PC2", out)
            add("decoder_feature_pca_scatter", out)

    p = run_data_dir / "decoder_feature_umap.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and {"decoder_umap_x", "decoder_umap_y"}.issubset(df.columns):
            if "artifact_score" in df.columns and df["artifact_score"].notna().any():
                out = plots_dir / "decoder_umap_by_artifact_score.png"
                _scatter_by_value(df, "decoder_umap_x", "decoder_umap_y", "artifact_score", "Decoder UMAP by artifact score", "UMAP-1", "UMAP-2", out)
                add("decoder_umap_by_artifact_score", out)
            if "bimodality_score" in df.columns and df["bimodality_score"].notna().any():
                out = plots_dir / "decoder_umap_by_bimodality.png"
                _scatter_by_value(df, "decoder_umap_x", "decoder_umap_y", "bimodality_score", "Decoder UMAP by bimodality score", "UMAP-1", "UMAP-2", out)
                add("decoder_umap_by_bimodality", out)
            if "manual_priority" in df.columns and df["manual_priority"].notna().any():
                out = plots_dir / "decoder_umap_by_manual_priority.png"
                _scatter_by_category(df, "decoder_umap_x", "decoder_umap_y", "manual_priority", "Decoder UMAP by manual priority", "UMAP-1", "UMAP-2", out)
                add("decoder_umap_by_manual_priority", out)

    p = run_data_dir / "decoder_feature_lda.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty and {"decoder_lda1", "decoder_lda2", "primary_label"}.issubset(df.columns):
            out = plots_dir / "decoder_lda_by_primary_label.png"
            _scatter_by_category(df, "decoder_lda1", "decoder_lda2", "primary_label", "Decoder LDA by heuristic primary label", "LDA-1", "LDA-2", out)
            add("decoder_lda_by_primary_label", out)

    p = run_data_dir / "feature_coverage_profiles.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            if "effective_pc_dim" in df.columns:
                out = plots_dir / "coverage_effective_pc_dim_hist.png"
                _hist(df["effective_pc_dim"], "Effective residual-PC dimension per feature", "effective PC dimension", out)
                add("coverage_effective_pc_dim_hist", out)
            if "pc_norm_mass_top_20" in df.columns:
                out = plots_dir / "coverage_top20_mass_hist.png"
                _hist(df["pc_norm_mass_top_20"], "Normalized mass in top residual PCs", "top-20 normalized mass", out)
                add("coverage_top20_mass_hist", out)
            if "coverage_bucket" in df.columns:
                out = plots_dir / "coverage_bucket_bar.png"
                _bar(df["coverage_bucket"].value_counts(), "Coverage buckets", "features", out)
                add("coverage_bucket_bar", out)

    p = run_data_dir / "feature_graph_alignment.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            if "gca_at_10" in df.columns:
                out = plots_dir / "gca_at_10_hist.png"
                _hist(df["gca_at_10"], "Geometry/coactivation agreement @10", "GCA@10", out)
                add("gca_at_10_hist", out)
            if "graph_alignment_bucket" in df.columns:
                out = plots_dir / "graph_alignment_bucket_bar.png"
                _bar(df["graph_alignment_bucket"].value_counts(), "Graph alignment buckets", "features", out)
                add("graph_alignment_bucket_bar", out)

    p = run_data_dir / "feature_steering_scores.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            if "atlas_steering_score" in df.columns:
                out = plots_dir / "steering_candidate_score_hist.png"
                _hist(df["atlas_steering_score"], "Atlas steering-candidate score", "score", out)
                add("steering_candidate_score_hist", out)
            if {"atlas_steering_score", "steering_risk_score"}.issubset(df.columns):
                out = plots_dir / "steering_score_vs_risk.png"
                _scatter(df, "atlas_steering_score", "steering_risk_score", "Candidate score vs risk", "atlas steering score", "risk", out)
                add("steering_score_vs_risk", out)

    return generated


def generate_html_tables(run_data_dir: Path, report_dir: Path, n: int = 40) -> dict[str, str]:
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    specs = {
        "feature_cards": ("feature_cards.parquet", ["manual_priority", "artifact_score"], "Canonical feature-card preview"),
        "top_features": ("feature_stats.parquet", ["n_token_activations", "p99_activation"], "Top raw feature statistics"),
        "top_bimodal_features": ("bimodal_feature_candidates.parquet", ["bimodality_score"], "Top bimodality candidates"),
        "geometry_quadrants": ("geometry_vs_coactivation.parquet", ["decoder_cosine", "jaccard"], "Geometry/coactivation pair diagnostics"),
        "inspection_features": ("inspection_feature_summaries.parquet", ["manual_priority", "artifact_score"], "Automated inspection feature summaries"),
        "inspection_pairs": ("inspection_pair_summaries.parquet", ["pair_artifact_score"], "Automated inspection pair summaries"),
        "coverage_profiles": ("feature_coverage_profiles.parquet", ["pc_norm_mass_top_20", "effective_pc_dim"], "Residual coverage profiles"),
        "graph_alignment": ("feature_graph_alignment.parquet", ["gca_at_10"], "Feature graph-alignment metrics"),
        "steering_candidates": ("feature_steering_scores.parquet", ["atlas_steering_score", "steering_risk_score"], "Steering-candidate hypotheses"),
    }

    for name, (filename, sort_cols, description) in specs.items():
        path = run_data_dir / filename
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        cols = [c for c in sort_cols if c in df.columns]
        if cols:
            ascending = [False] + [True] * (len(cols) - 1)
            df = df.sort_values(cols, ascending=ascending)
        out = tables_dir / f"{name}.html"
        _write_table_page(name.replace("_", " ").title(), description, df.head(n), out)
        generated[name] = str(out)

    return generated
