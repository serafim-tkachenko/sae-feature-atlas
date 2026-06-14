from __future__ import annotations

import pandas as pd
import torch
from tqdm import tqdm


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
) -> pd.DataFrame:
    W_dec = get_decoder_weight(sae).detach().float()
    W_dec = torch.nn.functional.normalize(W_dec, dim=-1)

    feature_ids_tensor = torch.tensor(feature_ids, device=W_dec.device, dtype=torch.long)

    rows = []
    for start in tqdm(range(0, len(feature_ids), batch_size), desc="Decoder neighbors"):
        batch_ids = feature_ids_tensor[start : start + batch_size]
        batch_vecs = W_dec[batch_ids]

        sims = batch_vecs @ W_dec.T

        for local_idx, fid in enumerate(batch_ids.tolist()):
            sims[local_idx, fid] = -1.0

        values, indices = torch.topk(sims, k=top_k, dim=-1)

        for local_idx, fid in enumerate(batch_ids.tolist()):
            for rank in range(top_k):
                rows.append(
                    {
                        "feature_i": int(fid),
                        "feature_j": int(indices[local_idx, rank].item()),
                        "rank": int(rank + 1),
                        "decoder_cosine": float(values[local_idx, rank].item()),
                    }
                )

    return pd.DataFrame(rows)


def merge_geometry_with_coactivation(
    decoder_neighbors: pd.DataFrame,
    coactivation_pairs: pd.DataFrame,
) -> pd.DataFrame:
    merged = decoder_neighbors.merge(
        coactivation_pairs,
        on=["feature_i", "feature_j"],
        how="left",
    )

    for col in ["coactivation_count", "jaccard", "pmi", "p_j_given_i", "p_i_given_j"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    if merged.empty:
        merged["quadrant"] = []
        return merged

    cos_hi = merged["decoder_cosine"].quantile(0.90)
    co_hi = merged["jaccard"].quantile(0.90)

    def quadrant(row):
        high_cos = row["decoder_cosine"] >= cos_hi
        high_co = row["jaccard"] >= co_hi
        if high_cos and high_co:
            return "high_cosine_high_coactivation"
        if high_cos and not high_co:
            return "high_cosine_low_coactivation"
        if not high_cos and high_co:
            return "low_cosine_high_coactivation"
        return "low_cosine_low_coactivation"

    merged["quadrant"] = merged.apply(quadrant, axis=1)
    return merged
