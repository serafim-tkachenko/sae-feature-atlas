from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sae_feature_atlas.config.config import ExperimentConfig


def _append_topk_rows(
    rows: list[dict],
    values: torch.Tensor,
    indices: torch.Tensor,
    text_id: int,
    source: str,
    token_strs: list[str],
) -> None:
    values_cpu = values.detach().cpu()
    indices_cpu = indices.detach().cpu()

    for pos in range(values_cpu.shape[0]):
        for k in range(values_cpu.shape[1]):
            activation = float(values_cpu[pos, k])
            if activation <= 0:
                continue
            rows.append(
                {
                    "text_id": int(text_id),
                    "source": source,
                    "token_pos": int(pos),
                    "token_str": token_strs[pos],
                    "feature_id": int(indices_cpu[pos, k]),
                    "activation": activation,
                }
            )


def _append_positive_rows(
    rows: list[dict],
    feature_acts: torch.Tensor,
    text_id: int,
    source: str,
    token_strs: list[str],
) -> None:
    acts = feature_acts[0]
    pos_indices, feature_indices = torch.nonzero(acts > 0, as_tuple=True)
    values = acts[pos_indices, feature_indices]

    pos_cpu = pos_indices.detach().cpu().tolist()
    feature_cpu = feature_indices.detach().cpu().tolist()
    values_cpu = values.detach().cpu().tolist()

    for pos, feature_id, activation in zip(pos_cpu, feature_cpu, values_cpu):
        rows.append(
            {
                "text_id": int(text_id),
                "source": source,
                "token_pos": int(pos),
                "token_str": token_strs[pos],
                "feature_id": int(feature_id),
                "activation": float(activation),
            }
        )


def collect_sparse_sae_activations(
    model,
    sae,
    cfg: ExperimentConfig,
    texts: list[dict],
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activation_rows: list[dict] = []
    token_summary_rows: list[dict] = []
    residual_vectors: list[np.ndarray] = []
    residual_meta: list[dict] = []

    for item in tqdm(texts, desc="Collecting SAE activations"):
        tokens = model.to_tokens(
            item["text"],
            truncate=True,
            prepend_bos=True,
        )[:, : cfg.collection.max_seq_len].to(device)

        token_strs = model.to_str_tokens(tokens[0])

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[cfg.model.hook_name])
            resid = cache[cfg.model.hook_name]

            if not torch.isfinite(resid).all():
                print(f"Skipping text_id={item['text_id']}: non-finite residual activations")
                continue

            feature_acts = sae.encode(resid.float())

            if not torch.isfinite(feature_acts).all():
                print(f"Skipping text_id={item['text_id']}: non-finite SAE activations")
                continue

            nonzero_counts = (feature_acts[0] > 0).sum(dim=-1).detach().cpu().tolist()

            if cfg.collection.activation_mode == "topk":
                values, indices = torch.topk(
                    feature_acts[0],
                    k=cfg.collection.top_k_features_per_token,
                    dim=-1,
                )
                _append_topk_rows(
                    rows=activation_rows,
                    values=values,
                    indices=indices,
                    text_id=int(item["text_id"]),
                    source=item["source"],
                    token_strs=token_strs,
                )
            elif cfg.collection.activation_mode == "positive":
                _append_positive_rows(
                    rows=activation_rows,
                    feature_acts=feature_acts,
                    text_id=int(item["text_id"]),
                    source=item["source"],
                    token_strs=token_strs,
                )
            else:
                raise ValueError(f"Unsupported activation mode: {cfg.collection.activation_mode}")

        resid_cpu = resid[0].detach().float().cpu()

        for pos, count in enumerate(nonzero_counts):
            token_summary_rows.append(
                {
                    "text_id": int(item["text_id"]),
                    "source": item["source"],
                    "token_pos": int(pos),
                    "token_str": token_strs[pos],
                    "n_positive_features": int(count),
                }
            )

        for pos in range(0, resid_cpu.shape[0], cfg.collection.residual_sample_stride):
            residual_vectors.append(resid_cpu[pos].numpy())
            residual_meta.append(
                {
                    "text_id": int(item["text_id"]),
                    "source": item["source"],
                    "token_pos": int(pos),
                }
            )

    acts_df = pd.DataFrame(activation_rows)
    residual_meta_df = pd.DataFrame(residual_meta)
    token_summary_df = pd.DataFrame(token_summary_rows)

    cfg.run_data_dir.mkdir(parents=True, exist_ok=True)
    acts_df.to_parquet(cfg.sae_activations_path, index=False)
    token_summary_df.to_parquet(cfg.token_activation_summary_path, index=False)

    d_model = cfg.model.d_model or (residual_vectors[0].shape[0] if residual_vectors else 0)
    if residual_vectors:
        np.save(cfg.residual_vectors_path, np.stack(residual_vectors))
    else:
        np.save(cfg.residual_vectors_path, np.empty((0, d_model), dtype=np.float32))

    residual_meta_df.to_parquet(cfg.residual_metadata_path, index=False)

    return acts_df, residual_meta_df, token_summary_df
