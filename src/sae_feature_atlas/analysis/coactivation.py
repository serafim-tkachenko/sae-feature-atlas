from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import math

import pandas as pd
from tqdm import tqdm

from sae_feature_atlas.analysis.populations import SPARSE_ACTIVATION_KEY, TOKEN_KEY


@dataclass(frozen=True)
class CoactivationResult:
    pairs: pd.DataFrame
    metadata: dict


def canonical_pair(feature_i: int, feature_j: int) -> tuple[int, int]:
    i, j = int(feature_i), int(feature_j)
    return (i, j) if i < j else (j, i)


def _retain_per_feature_neighbors(
    pairs: pd.DataFrame,
    neighbors_per_feature: int,
) -> pd.DataFrame:
    if pairs.empty or neighbors_per_feature <= 0:
        return pairs
    left = pairs.assign(feature_id=pairs["feature_i"], neighbor_id=pairs["feature_j"])
    right = pairs.assign(feature_id=pairs["feature_j"], neighbor_id=pairs["feature_i"])
    directed = pd.concat([left, right], ignore_index=True)
    directed = directed.sort_values(
        ["feature_id", "jaccard", "coactivation_count", "pmi"],
        ascending=[True, False, False, False],
    )
    retained = directed.groupby("feature_id", sort=False).head(neighbors_per_feature)
    keys = {
        canonical_pair(row.feature_id, row.neighbor_id)
        for row in retained.itertuples(index=False)
    }
    mask = [
        canonical_pair(i, j) in keys
        for i, j in zip(pairs["feature_i"], pairs["feature_j"])
    ]
    return pairs.loc[mask].copy()


def compute_same_token_coactivation(
    analysis_activations: pd.DataFrame,
    analysis_feature_ids: set[int],
    eligible_tokens: pd.DataFrame,
    *,
    min_pair_support: int = 10,
    neighbors_per_feature: int = 50,
    max_pairs: int = 100_000,
    storage_mode: str = "topk",
) -> CoactivationResult:
    """Compute joint stored-feature membership on the same eligible token.

    In top-k mode this is joint retained top-k membership, not joint positive SAE
    activation. PMI uses the complete eligible-token population supplied by the
    caller, including tokens with none of the selected analysis features.
    """
    unique = analysis_activations.drop_duplicates(SPARSE_ACTIVATION_KEY).copy()
    duplicate_rows_removed = int(len(analysis_activations) - len(unique))
    eligible = eligible_tokens[TOKEN_KEY].drop_duplicates()
    unique = unique.merge(eligible, on=TOKEN_KEY, how="inner", validate="many_to_one")
    unique = unique[unique["feature_id"].isin(analysis_feature_ids)].copy()
    n_tokens = int(len(eligible))
    if n_tokens == 0:
        raise ValueError("Eligible-token universe must be non-empty.")

    feature_counts = unique.groupby("feature_id").size().to_dict()
    grouped = unique.groupby(TOKEN_KEY)["feature_id"].apply(lambda x: sorted(set(x)))
    pair_counter: Counter[tuple[int, int]] = Counter()
    for features in tqdm(grouped, desc="Computing same-token coactivation"):
        for i, j in combinations(features, 2):
            pair_counter[canonical_pair(i, j)] += 1

    rows = []
    for (i, j), cij in pair_counter.items():
        if cij < min_pair_support:
            continue
        ci = int(feature_counts.get(i, 0))
        cj = int(feature_counts.get(j, 0))
        union = ci + cj - cij
        p_i, p_j, p_ij = ci / n_tokens, cj / n_tokens, cij / n_tokens
        rows.append(
            {
                "feature_i": int(i),
                "feature_j": int(j),
                "pair_key": f"{i}:{j}",
                "coactivation_count": int(cij),
                "feature_i_count": ci,
                "feature_j_count": cj,
                "eligible_token_count": n_tokens,
                "jaccard": float(cij / union) if union else 0.0,
                "pmi": float(math.log(p_ij / (p_i * p_j))),
                "p_j_given_i": float(cij / ci) if ci else 0.0,
                "p_i_given_j": float(cij / cj) if cj else 0.0,
                "coactivation_status": "observed_supported",
                "activation_population": "analysis_activations",
                "storage_semantics": (
                    "joint_retained_topk_membership"
                    if storage_mode == "topk"
                    else "joint_stored_positive_membership"
                ),
            }
        )

    supported = pd.DataFrame(rows)
    before_neighbor_retention = int(len(supported))
    retained = _retain_per_feature_neighbors(supported, neighbors_per_feature)
    before_global_cap = int(len(retained))
    truncated = before_global_cap > max_pairs
    if not retained.empty:
        retained = retained.sort_values(
            ["coactivation_count", "jaccard", "pmi"],
            ascending=False,
        ).head(max_pairs).reset_index(drop=True)

    metadata = {
        "activation_population": "analysis_activations",
        "storage_mode": storage_mode,
        "eligible_token_count": n_tokens,
        "analysis_feature_count": len(analysis_feature_ids),
        "duplicate_rows_removed": duplicate_rows_removed,
        "minimum_pair_support": int(min_pair_support),
        "neighbors_per_feature": int(neighbors_per_feature),
        "supported_pairs_before_neighbor_retention": before_neighbor_retention,
        "pairs_before_global_cap": before_global_cap,
        "global_cap": int(max_pairs),
        "global_cap_truncated": bool(truncated),
        "missing_pair_semantics": "unsupported_or_not_retained; never imputed as observed zero",
    }
    return CoactivationResult(pairs=retained, metadata=metadata)
