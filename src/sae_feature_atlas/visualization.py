from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save_hist(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 50) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(series.dropna(), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_scatter(x: pd.Series, y: pd.Series, title: str, xlabel: str, ylabel: str, path: Path) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_plots(run_data_dir: Path, report_dir: Path) -> dict[str, str]:
    plots_dir = report_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, str] = {}

    feature_stats_path = run_data_dir / "feature_stats.parquet"
    token_summary_path = run_data_dir / "token_activation_summary.parquet"
    decoder_neighbors_path = run_data_dir / "decoder_neighbors.parquet"
    geometry_path = run_data_dir / "geometry_vs_coactivation.parquet"
    bimodal_path = run_data_dir / "bimodal_feature_candidates.parquet"

    if feature_stats_path.exists():
        stats = pd.read_parquet(feature_stats_path)
        path = plots_dir / "feature_frequency_hist.png"
        _save_hist(stats["token_frequency"], "Feature token frequency", "token frequency", path)
        generated["feature_frequency_hist"] = str(path)

        path = plots_dir / "activation_p95_hist.png"
        _save_hist(stats["p95_activation"], "Feature p95 activation", "p95 activation", path)
        generated["activation_p95_hist"] = str(path)

    if token_summary_path.exists():
        token_summary = pd.read_parquet(token_summary_path)
        path = plots_dir / "positive_features_per_token_hist.png"
        _save_hist(
            token_summary["n_positive_features"],
            "Positive SAE features per token",
            "n positive features",
            path,
        )
        generated["positive_features_per_token_hist"] = str(path)

    if decoder_neighbors_path.exists():
        neigh = pd.read_parquet(decoder_neighbors_path)
        path = plots_dir / "decoder_cosine_hist.png"
        _save_hist(neigh["decoder_cosine"], "Decoder-neighbor cosine similarity", "cosine", path)
        generated["decoder_cosine_hist"] = str(path)

    if geometry_path.exists():
        geom = pd.read_parquet(geometry_path)
        path = plots_dir / "geometry_vs_coactivation_scatter.png"
        _save_scatter(
            geom["decoder_cosine"],
            geom["jaccard"],
            "Decoder cosine vs co-activation Jaccard",
            "decoder cosine",
            "co-activation Jaccard",
            path,
        )
        generated["geometry_vs_coactivation_scatter"] = str(path)

    if bimodal_path.exists():
        bimodal = pd.read_parquet(bimodal_path)
        if not bimodal.empty:
            path = plots_dir / "bimodality_score_hist.png"
            _save_hist(
                bimodal["bimodality_score"],
                "Bimodality score distribution",
                "BIC improvement",
                path,
            )
            generated["bimodality_score_hist"] = str(path)

    return generated


def generate_html_tables(run_data_dir: Path, report_dir: Path, n: int = 30) -> dict[str, str]:
    tables_dir = report_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    generated: dict[str, str] = {}

    table_specs = {
        "top_features": ("feature_stats.parquet", ["n_token_activations", "p99_activation"]),
        "filtered_features": ("filtered_features.parquet", ["n_token_activations", "p99_activation"]),
        "top_bimodal_features": ("bimodal_feature_candidates.parquet", ["bimodality_score"]),
        "geometry_quadrants": ("geometry_vs_coactivation.parquet", ["decoder_cosine", "jaccard"]),
    }

    for name, (filename, sort_cols) in table_specs.items():
        path = run_data_dir / filename
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.empty:
            continue
        existing_sort_cols = [col for col in sort_cols if col in df.columns]
        if existing_sort_cols:
            df = df.sort_values(existing_sort_cols, ascending=False)
        out = tables_dir / f"{name}.html"
        df.head(n).to_html(out, index=False, escape=False)
        generated[name] = str(out)

    return generated
