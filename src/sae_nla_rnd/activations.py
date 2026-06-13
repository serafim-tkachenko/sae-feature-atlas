from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sae_nla_rnd.config import ExperimentConfig


def collect_sparse_sae_activations(
    model,
    sae,
    cfg: ExperimentConfig,
    texts: list[dict],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    activation_rows: list[dict] = []
    residual_vectors: list[np.ndarray] = []
    residual_meta: list[dict] = []

    for item in tqdm(texts, desc="Collecting SAE activations"):
        text_id = item["text_id"]
        source = item["source"]
        text = item["text"]

        tokens = model.to_tokens(
            text,
            truncate=True,
            prepend_bos=True,
        )[:, : cfg.collection.max_seq_len].to(device)

        token_strs = model.to_str_tokens(tokens[0])

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[cfg.model.hook_name])
            resid = cache[cfg.model.hook_name]

            if not torch.isfinite(resid).all():
                print(f"Skipping text_id={text_id}: non-finite residual activations")
                continue

            feature_acts = sae.encode(resid.float())

            if not torch.isfinite(feature_acts).all():
                print(f"Skipping text_id={text_id}: non-finite SAE activations")
                continue

            values, indices = torch.topk(
                feature_acts[0],
                k=cfg.collection.top_k_features_per_token,
                dim=-1,
            )

        values_cpu = values.detach().cpu()
        indices_cpu = indices.detach().cpu()
        resid_cpu = resid[0].detach().float().cpu()

        seq_len = values_cpu.shape[0]

        for pos in range(seq_len):
            for k in range(cfg.collection.top_k_features_per_token):
                activation = float(values_cpu[pos, k])
                if activation <= 0:
                    continue

                activation_rows.append(
                    {
                        "text_id": int(text_id),
                        "source": source,
                        "token_pos": int(pos),
                        "token_str": token_strs[pos],
                        "feature_id": int(indices_cpu[pos, k]),
                        "activation": activation,
                    }
                )

        for pos in range(0, resid_cpu.shape[0], cfg.collection.residual_sample_stride):
            residual_vectors.append(resid_cpu[pos].numpy())
            residual_meta.append(
                {
                    "text_id": int(text_id),
                    "source": source,
                    "token_pos": int(pos),
                }
            )

    acts_df = pd.DataFrame(activation_rows)
    residual_meta_df = pd.DataFrame(residual_meta)

    cfg.run_data_dir.mkdir(parents=True, exist_ok=True)
    acts_df.to_parquet(cfg.sae_activations_path, index=False)

    if residual_vectors:
        np.save(cfg.residual_vectors_path, np.stack(residual_vectors))
    else:
        np.save(cfg.residual_vectors_path, np.empty((0, cfg.model.d_model), dtype=np.float32))

    residual_meta_df.to_parquet(cfg.residual_metadata_path, index=False)

    return acts_df, residual_meta_df
