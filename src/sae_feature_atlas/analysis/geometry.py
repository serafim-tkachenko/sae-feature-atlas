from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm

from sae_feature_atlas.analysis.coactivation import canonical_pair


def get_decoder_weight(sae) -> torch.Tensor:
    if hasattr(sae, "W_dec"):
        return sae.W_dec
    if hasattr(sae, "W_dec_DF"):
        return sae.W_dec_DF
    raise AttributeError("Could not find decoder weight on SAE object.")


def compute_decoder_neighbors(
    sae,
    feature_ids: list[int],
    top_k: int = 20,
    batch_size: int = 512,
    candidate_feature_ids: list[int] | None = None,
) -> pd.DataFrame:
    """Compute directed decoder neighbors in an explicit candidate universe."""
    w_dec = torch.nn.functional.normalize(get_decoder_weight(sae).detach().float(), dim=-1)
    query_ids = torch.tensor(feature_ids, device=w_dec.device, dtype=torch.long)
    candidate_ids_list = candidate_feature_ids if candidate_feature_ids is not None else list(
        range(w_dec.shape[0])
    )
    candidate_ids = torch.tensor(candidate_ids_list, device=w_dec.device, dtype=torch.long)
    candidate_vectors = w_dec[candidate_ids]
    effective_k = min(top_k, max(0, len(candidate_ids_list) - 1))
    rows = []
    if effective_k == 0:
        return pd.DataFrame()

    candidate_position = {feature_id: idx for idx, feature_id in enumerate(candidate_ids_list)}
    universe = (
        "analysis_features"
        if candidate_feature_ids is not None
        else "all_decoder_features"
    )
    for start in tqdm(range(0, len(feature_ids), batch_size), desc="Decoder neighbors"):
        batch_ids = query_ids[start : start + batch_size]
        sims = w_dec[batch_ids] @ candidate_vectors.T
        for local_idx, feature_id in enumerate(batch_ids.tolist()):
            candidate_idx = candidate_position.get(feature_id)
            if candidate_idx is not None:
                sims[local_idx, candidate_idx] = -1.0
        values, indices = torch.topk(sims, k=effective_k, dim=-1)
        for local_idx, feature_id in enumerate(batch_ids.tolist()):
            for rank in range(effective_k):
                neighbor_idx = int(indices[local_idx, rank].item())
                rows.append(
                    {
                        "feature_i": int(feature_id),
                        "feature_j": int(candidate_ids[neighbor_idx].item()),
                        "rank": int(rank + 1),
                        "decoder_cosine": float(values[local_idx, rank].item()),
                        "geometry_target_universe": universe,
                    }
                )
    return pd.DataFrame(rows)


def merge_geometry_with_coactivation(
    decoder_neighbors: pd.DataFrame,
    coactivation_pairs: pd.DataFrame,
    analysis_feature_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Match directed geometry edges to orientation-invariant empirical pairs."""
    geometry = decoder_neighbors.copy()
    geometry["pair_key"] = [
        f"{i}:{j}" for i, j in (
            canonical_pair(i, j)
            for i, j in zip(geometry["feature_i"], geometry["feature_j"])
        )
    ]
    coactivation = coactivation_pairs.copy()
    if not coactivation.empty:
        coactivation["pair_key"] = [
            f"{i}:{j}" for i, j in (
                canonical_pair(i, j)
                for i, j in zip(coactivation["feature_i"], coactivation["feature_j"])
            )
        ]
        empirical_columns = [
            column
            for column in [
                "pair_key",
                "coactivation_count",
                "jaccard",
                "pmi",
                "p_j_given_i",
                "p_i_given_j",
                "coactivation_status",
            ]
            if column in coactivation.columns
        ]
        empirical = coactivation[empirical_columns].drop_duplicates("pair_key")
        merged = geometry.merge(empirical, on="pair_key", how="left", validate="many_to_one")
    else:
        merged = geometry.copy()

    for column in [
        "coactivation_count",
        "jaccard",
        "pmi",
        "p_j_given_i",
        "p_i_given_j",
        "coactivation_status",
    ]:
        if column not in merged.columns:
            merged[column] = pd.NA

    observed = merged.get(
        "coactivation_status", pd.Series(index=merged.index, dtype="object")
    ).eq("observed_supported")
    if analysis_feature_ids is None:
        eligible = pd.Series(True, index=merged.index)
    else:
        eligible = (
            merged["feature_i"].isin(analysis_feature_ids)
            & merged["feature_j"].isin(analysis_feature_ids)
        )
    merged["coactivation_match_status"] = "unsupported_or_not_retained"
    merged.loc[~eligible, "coactivation_match_status"] = "not_eligible"
    merged.loc[observed, "coactivation_match_status"] = "observed_supported"

    if merged.empty:
        merged["quadrant"] = pd.Series(dtype="object")
        return merged

    cosine_threshold = float(merged["decoder_cosine"].quantile(0.90))
    observed_jaccard = merged.loc[observed, "jaccard"]
    jaccard_threshold = (
        float(observed_jaccard.quantile(0.90)) if not observed_jaccard.empty else float("nan")
    )

    def quadrant(row) -> str:
        if row["coactivation_match_status"] != "observed_supported":
            return "coactivation_unavailable"
        high_cosine = row["decoder_cosine"] >= cosine_threshold
        high_coactivation = row["jaccard"] >= jaccard_threshold
        if high_cosine and high_coactivation:
            return "high_cosine_high_coactivation"
        if high_cosine:
            return "high_cosine_low_coactivation"
        if high_coactivation:
            return "low_cosine_high_coactivation"
        return "low_cosine_low_coactivation"

    merged["quadrant"] = merged.apply(quadrant, axis=1)
    merged["decoder_cosine_threshold"] = cosine_threshold
    merged["coactivation_jaccard_threshold"] = jaccard_threshold
    return merged
