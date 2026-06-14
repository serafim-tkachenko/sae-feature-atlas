from __future__ import annotations

import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_feature_atlas.config import ExperimentConfig


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_sae(cfg: ExperimentConfig, device: str) -> SAE:
    return SAE.from_pretrained(
        release=cfg.model.sae_release,
        sae_id=cfg.model.sae_id,
        device=device,
    )


def load_model(cfg: ExperimentConfig, device: str) -> HookedTransformer:
    # bfloat16 is intentional: float16 produced NaNs in the initial Gemma 3 run.
    model = HookedTransformer.from_pretrained(
        cfg.model.model_name,
        device=device,
        dtype=torch.bfloat16,
    )
    model.eval()
    return model


def validate_model_sae_compatibility(
    model: HookedTransformer,
    sae: SAE,
    cfg: ExperimentConfig,
    device: str,
    prompt: str = "The Eiffel Tower is located in the city of",
) -> dict:
    hook_name = cfg.model.hook_name

    if hook_name not in model.hook_dict:
        available = [name for name in model.hook_dict if "resid" in name]
        raise ValueError(f"Hook {hook_name!r} not found. Residual hooks: {available[:50]}")

    tokens = model.to_tokens(prompt).to(device)

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])

    resid = cache[hook_name]

    if resid.shape[-1] != sae.cfg.d_in:
        raise ValueError(
            f"Residual d_model={resid.shape[-1]} does not match SAE d_in={sae.cfg.d_in}"
        )

    if not torch.isfinite(resid).all():
        raise ValueError(
            f"Residual activations contain non-finite values: "
            f"nan={torch.isnan(resid).sum().item()}, inf={torch.isinf(resid).sum().item()}"
        )

    with torch.no_grad():
        feature_acts = sae.encode(resid.float())

    if not torch.isfinite(feature_acts).all():
        raise ValueError(
            f"SAE activations contain non-finite values: "
            f"nan={torch.isnan(feature_acts).sum().item()}, "
            f"inf={torch.isinf(feature_acts).sum().item()}"
        )

    return {
        "model_name": cfg.model.model_name,
        "sae_release": cfg.model.sae_release,
        "sae_id": cfg.model.sae_id,
        "hook_name": cfg.model.hook_name,
        "resid_shape": tuple(resid.shape),
        "resid_dtype": str(resid.dtype),
        "sae_d_in": int(sae.cfg.d_in),
        "feature_acts_shape": tuple(feature_acts.shape),
        "mean_nonzero_features_per_token": float(
            (feature_acts > 0).sum(dim=-1).float().mean().item()
        ),
        "max_feature_activation": float(feature_acts.max().item()),
    }
