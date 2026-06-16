from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def _top_neighbors(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    sort_cols: list[str],
    k: int,
) -> dict[int, list[int]]:
    if df.empty:
        return {}

    work = df.copy()
    ascending = [False for _ in sort_cols]
    if "rank" in sort_cols:
        ascending = [col == "rank" for col in sort_cols]

    work = work.sort_values([source_col] + sort_cols, ascending=[True] + ascending)
    out: dict[int, list[int]] = {}
    for fid, group in work.groupby(source_col):
        out[int(fid)] = group[target_col].astype(int).head(k).tolist()
    return out


def _coactivation_neighbor_table(coactivation_pairs: pd.DataFrame) -> pd.DataFrame:
    if coactivation_pairs.empty:
        return pd.DataFrame(columns=["feature_id", "neighbor_feature_id", "jaccard", "pmi", "coactivation_count"])

    left = coactivation_pairs.rename(
        columns={"feature_i": "feature_id", "feature_j": "neighbor_feature_id"}
    )
    right = coactivation_pairs.rename(
        columns={"feature_j": "feature_id", "feature_i": "neighbor_feature_id"}
    )
    both = pd.concat([left, right], ignore_index=True)
    keep = [
        c
        for c in ["feature_id", "neighbor_feature_id", "jaccard", "pmi", "coactivation_count"]
        if c in both.columns
    ]
    return both[keep]


def _alignment_bucket(gca: float, has_geometry: bool, has_coactivation: bool) -> str:
    if not has_geometry and not has_coactivation:
        return "no_neighbors_observed"
    if has_geometry and not has_coactivation:
        return "geometry_only"
    if has_coactivation and not has_geometry:
        return "coactivation_only"
    if gca >= 0.30:
        return "strong_geometry_coactivation_agreement"
    if gca > 0:
        return "weak_geometry_coactivation_agreement"
    return "geometry_coactivation_disagreement"


def compute_graph_alignment(
    decoder_neighbors: pd.DataFrame,
    coactivation_pairs: pd.DataFrame,
    feature_ids: Iterable[int] | None = None,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare decoder-neighbor geometry with empirical coactivation neighborhoods.

    For each feature i, GCA@k is |N_geom^k(i) ∩ N_coact^k(i)| / k.
    It answers whether features that are close in SAE decoder space are also
    used together on the same tokens.
    """
    if not k_values:
        k_values = (10,)
    max_k = max(k_values)

    decoder_neighbors = decoder_neighbors.copy() if decoder_neighbors is not None else pd.DataFrame()
    coactivation_pairs = coactivation_pairs.copy() if coactivation_pairs is not None else pd.DataFrame()
    coact_neighbors = _coactivation_neighbor_table(coactivation_pairs)

    if feature_ids is None:
        ids: set[int] = set()
        if not decoder_neighbors.empty:
            ids.update(decoder_neighbors["feature_i"].astype(int).tolist())
        if not coact_neighbors.empty:
            ids.update(coact_neighbors["feature_id"].astype(int).tolist())
        feature_ids = sorted(ids)
    else:
        feature_ids = sorted({int(fid) for fid in feature_ids})

    geom_top = _top_neighbors(
        decoder_neighbors,
        source_col="feature_i",
        target_col="feature_j",
        sort_cols=["rank"],
        k=max_k,
    )
    coact_top = _top_neighbors(
        coact_neighbors,
        source_col="feature_id",
        target_col="neighbor_feature_id",
        sort_cols=["jaccard", "pmi", "coactivation_count"],
        k=max_k,
    )

    rows: list[dict] = []
    for fid in feature_ids:
        geom = geom_top.get(int(fid), [])
        coact = coact_top.get(int(fid), [])
        row: dict[str, int | float | str | bool] = {
            "feature_id": int(fid),
            "n_geometry_neighbors": int(len(geom)),
            "n_coactivation_neighbors": int(len(coact)),
        }

        for k in k_values:
            geom_k = set(geom[:k])
            coact_k = set(coact[:k])
            denom = max(1, int(k))
            row[f"gca_at_{k}"] = float(len(geom_k & coact_k) / denom)

        primary_k = 10 if 10 in k_values else k_values[0]
        gca = float(row[f"gca_at_{primary_k}"])
        row["graph_alignment_bucket"] = _alignment_bucket(
            gca,
            has_geometry=len(geom) > 0,
            has_coactivation=len(coact) > 0,
        )
        rows.append(row)

    alignment = pd.DataFrame(rows)
    if alignment.empty:
        return alignment, pd.DataFrame()

    summary_rows = []
    for k in k_values:
        col = f"gca_at_{k}"
        summary_rows.append(
            {
                "metric": col,
                "mean": float(alignment[col].mean()),
                "median": float(alignment[col].median()),
                "p90": float(alignment[col].quantile(0.90)),
                "nonzero_share": float((alignment[col] > 0).mean()),
            }
        )

    bucket_counts = alignment["graph_alignment_bucket"].value_counts().reset_index()
    bucket_counts.columns = ["bucket", "count"]
    bucket_counts["share"] = bucket_counts["count"] / max(1, len(alignment))
    bucket_counts["metric"] = "bucket_count"

    summary = pd.concat([pd.DataFrame(summary_rows), bucket_counts], ignore_index=True, sort=False)
    return alignment, summary
