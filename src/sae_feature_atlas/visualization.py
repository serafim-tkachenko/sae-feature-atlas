from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _hist(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 50) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _scatter(df: pd.DataFrame, x: str, y: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    work = df.dropna(subset=[x, y])
    ax.scatter(work[x], work[y], s=8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _scatter_by_value(df: pd.DataFrame, x: str, y: str, value: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    work = df.dropna(subset=[x, y, value])
    sc = ax.scatter(work[x], work[y], c=work[value], s=8, alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(sc, ax=ax, label=value)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _scatter_by_category(df: pd.DataFrame, x: str, y: str, category: str, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    work = df.dropna(subset=[x, y, category])
    for label, group in work.groupby(category):
        ax.scatter(group[x], group[y], s=8, alpha=0.55, label=str(label)[:40])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if work[category].nunique() <= 12:
        ax.legend(fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _bar(counts: pd.Series, title: str, ylabel: str, path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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
        if not df.empty:
            out = plots_dir / "positive_features_per_token_hist.png"
            _hist(df["n_positive_features"], "Positive SAE features per token", "n positive features", out)
            add("positive_features_per_token_hist", out)

    p = run_data_dir / "geometry_vs_coactivation.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            out = plots_dir / "geometry_vs_coactivation_scatter.png"
            _scatter(df, "decoder_cosine", "jaccard", "Decoder cosine vs co-activation", "decoder cosine", "Jaccard", out)
            add("geometry_vs_coactivation_scatter", out)
            out = plots_dir / "geometry_quadrants_bar.png"
            _bar(df["quadrant"].value_counts(), "Geometry/co-activation quadrants", "pairs", out)
            add("geometry_quadrants_bar", out)

    p = run_data_dir / "bimodal_feature_candidates.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            out = plots_dir / "bimodality_score_hist.png"
            _hist(df["bimodality_score"], "Bimodality score", "BIC improvement", out)
            add("bimodality_score_hist", out)

    p = run_data_dir / "inspection_feature_summaries.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            out = plots_dir / "artifact_score_hist.png"
            _hist(df["artifact_score"], "Automated artifact score", "artifact score", out)
            add("artifact_score_hist", out)

    p = run_data_dir / "residual_pca_summary.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            out = plots_dir / "residual_pca_variance.png"
            _bar(df.set_index("component")["explained_variance_ratio"], "Residual PCA explained variance", "variance ratio", out)
            add("residual_pca_variance", out)

    p = run_data_dir / "decoder_pca_summary.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            out = plots_dir / "decoder_pca_variance.png"
            _bar(df.set_index("component")["explained_variance_ratio"], "Decoder PCA explained variance", "variance ratio", out)
            add("decoder_pca_variance", out)

    p = run_data_dir / "decoder_feature_pca.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
            out = plots_dir / "decoder_feature_pca_scatter.png"
            _scatter(df.head(5000), "decoder_pc1", "decoder_pc2", "Decoder feature PCA", "PC1", "PC2", out)
            add("decoder_feature_pca_scatter", out)

    p = run_data_dir / "decoder_feature_umap.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        if not df.empty:
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
        if not df.empty:
            out = plots_dir / "decoder_lda_by_primary_label.png"
            _scatter_by_category(df, "decoder_lda1", "decoder_lda2", "primary_label", "Decoder LDA by heuristic primary label", "LDA-1", "LDA-2", out)
            add("decoder_lda_by_primary_label", out)

    return generated


def generate_html_tables(run_data_dir: Path, report_dir: Path, n: int = 40) -> dict[str, str]:
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}

    specs = {
        "feature_cards": ("feature_cards.parquet", ["manual_priority", "artifact_score"]),
        "top_features": ("feature_stats.parquet", ["n_token_activations", "p99_activation"]),
        "top_bimodal_features": ("bimodal_feature_candidates.parquet", ["bimodality_score"]),
        "geometry_quadrants": ("geometry_vs_coactivation.parquet", ["decoder_cosine", "jaccard"]),
        "inspection_features": ("inspection_feature_summaries.parquet", ["manual_priority", "artifact_score"]),
        "inspection_pairs": ("inspection_pair_summaries.parquet", ["pair_artifact_score"]),
    }
    for name, (filename, sort_cols) in specs.items():
        path = run_data_dir / filename
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        cols = [c for c in sort_cols if c in df.columns]
        if cols:
            df = df.sort_values(cols, ascending=False)
        out = tables_dir / f"{name}.html"
        df.head(n).to_html(out, index=False, escape=False)
        generated[name] = str(out)

    return generated
