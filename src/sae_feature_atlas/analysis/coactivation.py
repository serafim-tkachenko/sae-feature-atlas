from __future__ import annotations

from collections import Counter
from itertools import combinations
import math

import pandas as pd
from tqdm import tqdm


def compute_same_token_coactivation(
    acts: pd.DataFrame,
    filtered_feature_ids: set[int],
    max_pairs: int = 100_000,
) -> pd.DataFrame:
    acts = acts[acts["feature_id"].isin(filtered_feature_ids)].copy()

    feature_counts = acts.groupby("feature_id").size().to_dict()
    n_tokens = acts[["text_id", "token_pos"]].drop_duplicates().shape[0]

    pair_counter: Counter[tuple[int, int]] = Counter()
    grouped = acts.groupby(["text_id", "token_pos"])["feature_id"].apply(lambda x: sorted(set(x)))

    for features in tqdm(grouped, desc="Computing same-token coactivation"):
        if len(features) < 2:
            continue
        for i, j in combinations(features, 2):
            pair_counter[(i, j)] += 1

    rows = []
    for (i, j), cij in pair_counter.items():
        ci = feature_counts.get(i, 0)
        cj = feature_counts.get(j, 0)

        union = ci + cj - cij
        jaccard = cij / union if union > 0 else 0.0

        p_i = ci / n_tokens
        p_j = cj / n_tokens
        p_ij = cij / n_tokens
        pmi = math.log(p_ij / (p_i * p_j)) if p_i > 0 and p_j > 0 and p_ij > 0 else 0.0

        rows.append(
            {
                "feature_i": int(i),
                "feature_j": int(j),
                "coactivation_count": int(cij),
                "feature_i_count": int(ci),
                "feature_j_count": int(cj),
                "jaccard": float(jaccard),
                "pmi": float(pmi),
                "p_j_given_i": float(cij / ci) if ci else 0.0,
                "p_i_given_j": float(cij / cj) if cj else 0.0,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["pmi", "jaccard", "coactivation_count"], ascending=False)
    return df.head(max_pairs).reset_index(drop=True)
