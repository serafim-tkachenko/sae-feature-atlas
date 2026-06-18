from __future__ import annotations

from dataclasses import replace

from sae_feature_atlas.config.schema import (
    ActivationMode,
    CollectionConfig,
    ExperimentConfig,
    ModelConfig,
    ModelSelection,
)


MODEL_ALIASES: dict[str, str] = {
    "gemma-3-270m-pt": "google/gemma-3-270m-pt",
    "gemma-3-270m-it": "google/gemma-3-270m-it",
    "gemma-3-1b-pt": "google/gemma-3-1b-pt",
    "gemma-3-1b-it": "google/gemma-3-1b-it",
    "gemma-3-4b-pt": "google/gemma-3-4b-pt",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma-3-12b-pt": "google/gemma-3-12b-pt",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "gemma-3-27b-pt": "google/gemma-3-27b-pt",
    "gemma-3-27b-it": "google/gemma-3-27b-it",
}

DEFAULT_LAYERS: dict[str, int] = {
    "gemma-3-270m-pt": 12,
    "gemma-3-270m-it": 12,
    "gemma-3-1b-pt": 13,
    "gemma-3-1b-it": 13,
    "gemma-3-4b-pt": 12,
    "gemma-3-4b-it": 12,
    "gemma-3-12b-pt": 20,
    "gemma-3-12b-it": 20,
    "gemma-3-27b-pt": 20,
    "gemma-3-27b-it": 20,
}

DEFAULT_L0: dict[str, str] = {
    "gemma-3-4b-pt": "small",
    "gemma-3-4b-it": "small",
}

SITE_TO_RELEASE_SUFFIX: dict[str, str] = {
    "resid_post": "res",
    "res": "res",
    "mlp_out": "mlp",
    "mlp": "mlp",
    "attn_out": "att",
    "att": "att",
}


def list_supported_models() -> list[str]:
    return sorted(MODEL_ALIASES)


def normalize_model_alias(model: str) -> str:
    if model.startswith("google/"):
        for alias, hf_name in MODEL_ALIASES.items():
            if model == hf_name:
                return alias
    if model not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model {model!r}. Supported: {list_supported_models()}")
    return model


def hf_model_name(model: str) -> str:
    alias = normalize_model_alias(model)
    return MODEL_ALIASES[alias]


def infer_gemma_scope_release(model: str, site: str) -> str:
    alias = normalize_model_alias(model)
    suffix = SITE_TO_RELEASE_SUFFIX.get(site)
    if suffix is None:
        raise ValueError(f"Unsupported site {site!r}. Supported: {sorted(SITE_TO_RELEASE_SUFFIX)}")
    short = alias.replace("gemma-3-", "")
    return f"gemma-scope-2-{short}-{suffix}"


SITE_ALIASES: dict[str, str] = {
    "res": "resid_post",
    "mlp": "mlp_out",
    "att": "attn_out",
}


def normalize_site(site: str) -> str:
    normalized = SITE_ALIASES.get(site, site)
    if normalized not in {"resid_post", "mlp_out", "attn_out"}:
        raise ValueError(f"Unsupported site {site!r}. Supported: {sorted(SITE_TO_RELEASE_SUFFIX)}")
    return normalized


def infer_hook_name(layer: int, site: str) -> str:
    normalized_site = normalize_site(site)
    if normalized_site == "resid_post":
        return f"blocks.{layer}.hook_resid_post"
    if normalized_site == "mlp_out":
        return f"blocks.{layer}.hook_mlp_out"
    if normalized_site == "attn_out":
        return f"blocks.{layer}.hook_attn_out"
    raise ValueError(f"Unsupported site {site!r}")


def infer_sae_id(layer: int, width: str, l0: str) -> str:
    return f"layer_{layer}_width_{width}_l0_{l0}"


def infer_run_name(
    model: str,
    layer: int,
    site: str,
    width: str,
    corpus: str,
    activation_mode: ActivationMode,
    top_k: int,
) -> str:
    model_part = normalize_model_alias(model).replace("-", "")
    normalized_site = normalize_site(site)
    site_part = "res" if normalized_site == "resid_post" else normalized_site.replace("_out", "")
    corpus_part = corpus.replace("/", "_").replace("-", "")
    if activation_mode == "topk":
        return f"{model_part}_l{layer}_{site_part}{width}_{corpus_part}_top{top_k}"
    return f"{model_part}_l{layer}_{site_part}{width}_{corpus_part}_positive"


def resolve_gemma_scope_sae(selection: ModelSelection) -> ModelConfig:
    alias = normalize_model_alias(selection.model)
    layer = selection.layer if selection.layer is not None else DEFAULT_LAYERS[alias]
    l0 = selection.l0 or DEFAULT_L0.get(alias, "medium")

    return ModelConfig(
        model_name=hf_model_name(alias),
        sae_release=infer_gemma_scope_release(alias, selection.site),
        sae_id=infer_sae_id(layer=layer, width=selection.width, l0=l0),
        layer=layer,
        hook_name=infer_hook_name(layer=layer, site=selection.site),
        site=normalize_site(selection.site),
        width=selection.width,
        l0=l0,
        d_model=None,
        d_sae=None,
    )


def make_config(
    model: str = "gemma-3-1b-pt",
    layer: int | None = None,
    site: str = "resid_post",
    width: str = "16k",
    l0: str | None = None,
    corpus: str = "pile-10k",
    max_texts: int = 1500,
    max_seq_len: int = 256,
    activation_mode: ActivationMode = "topk",
    top_k: int = 64,
    run_name: str | None = None,
) -> ExperimentConfig:
    alias = normalize_model_alias(model)
    effective_l0 = l0 or DEFAULT_L0.get(alias, "medium")
    selection = ModelSelection(model=alias, layer=layer, site=site, width=width, l0=effective_l0)
    model_cfg = resolve_gemma_scope_sae(selection)
    effective_layer = model_cfg.layer

    if run_name is None:
        run_name = infer_run_name(
            model=alias,
            layer=effective_layer,
            site=site,
            width=width,
            corpus=corpus,
            activation_mode=activation_mode,
            top_k=top_k,
        )

    cfg = ExperimentConfig(
        model=model_cfg,
        collection=CollectionConfig(
            run_name=run_name,
            corpus=corpus,
            max_texts=max_texts,
            max_seq_len=max_seq_len,
            activation_mode=activation_mode,
            top_k_features_per_token=top_k,
        ),
    )

    return cfg


def with_collection_overrides(
    cfg: ExperimentConfig,
    run_name: str | None = None,
    max_texts: int | None = None,
    max_seq_len: int | None = None,
    activation_mode: ActivationMode | None = None,
    top_k: int | None = None,
) -> ExperimentConfig:
    collection = cfg.collection
    if run_name is not None:
        collection = replace(collection, run_name=run_name)
    if max_texts is not None:
        collection = replace(collection, max_texts=max_texts)
    if max_seq_len is not None:
        collection = replace(collection, max_seq_len=max_seq_len)
    if activation_mode is not None:
        collection = replace(collection, activation_mode=activation_mode)
    if top_k is not None:
        collection = replace(collection, top_k_features_per_token=top_k)
    return replace(cfg, collection=collection)
