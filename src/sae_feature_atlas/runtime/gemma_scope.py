from __future__ import annotations

from dataclasses import dataclass

import torch

from sae_feature_atlas.config.schema import ExperimentConfig
from sae_feature_atlas.runtime.loaders import (
    get_device,
    load_model,
    load_sae,
    validate_model_sae_compatibility,
)


@dataclass
class GemmaScopeRuntime:
    cfg: ExperimentConfig
    device: str | None = None
    model: object | None = None
    sae: object | None = None

    def load(self) -> "GemmaScopeRuntime":
        self.device = self.device or get_device()
        self.sae = load_sae(self.cfg, self.device)
        self.model = load_model(self.cfg, self.device)
        return self

    def validate(self) -> dict:
        self._require_loaded()
        return validate_model_sae_compatibility(
            model=self.model,
            sae=self.sae,
            cfg=self.cfg,
            device=self.device,
        )

    def residual_activations(self, text: str) -> torch.Tensor:
        self._require_loaded()
        tokens = self.model.to_tokens(
            text,
            truncate=True,
            prepend_bos=True,
        )[:, : self.cfg.collection.max_seq_len].to(self.device)

        with torch.no_grad():
            _, cache = self.model.run_with_cache(
                tokens,
                names_filter=[self.cfg.model.hook_name],
            )

        return cache[self.cfg.model.hook_name]

    def sae_activations(self, text: str) -> torch.Tensor:
        resid = self.residual_activations(text)
        with torch.no_grad():
            return self.sae.encode(resid.float())

    def topk_features(self, text: str, top_k: int | None = None):
        acts = self.sae_activations(text)
        k = top_k or self.cfg.collection.top_k_features_per_token
        return torch.topk(acts[0], k=k, dim=-1)

    def _require_loaded(self) -> None:
        if self.model is None or self.sae is None or self.device is None:
            raise RuntimeError("Bundle is not loaded. Call bundle.load() first.")
