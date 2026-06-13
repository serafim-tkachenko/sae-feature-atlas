from __future__ import annotations

import json

from sae_nla_rnd.activations import collect_sparse_sae_activations
from sae_nla_rnd.config import CFG
from sae_nla_rnd.dataset import build_text_dataset, build_token_metadata, save_text_dataset
from sae_nla_rnd.io_utils import ensure_project_dirs
from sae_nla_rnd.manifest import build_run_manifest, write_run_manifest
from sae_nla_rnd.model_utils import (
    get_device,
    load_model,
    load_sae,
    validate_model_sae_compatibility,
)


def main() -> None:
    ensure_project_dirs()
    CFG.run_data_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Device:", device)

    print("Loading SAE...")
    sae = load_sae(CFG, device)

    print("Loading model...")
    model = load_model(CFG, device)

    print("Running smoke test...")
    smoke = validate_model_sae_compatibility(model, sae, CFG, device)
    print(json.dumps(smoke, indent=2))

    print("Building text dataset...")
    texts = build_text_dataset(CFG)
    save_text_dataset(CFG, texts)
    print(f"Saved {len(texts)} texts to {CFG.paths.raw_texts_path}")

    print("Building token metadata...")
    token_meta = build_token_metadata(model, CFG, texts, device)
    print(f"Token metadata: {token_meta.shape}")

    print("Collecting sparse SAE activations...")
    acts, residual_meta = collect_sparse_sae_activations(model, sae, CFG, texts, device)

    metrics = {
        "texts": len(texts),
        "tokens": int(token_meta[["text_id", "token_pos"]].drop_duplicates().shape[0]),
        "sparse_activation_rows": int(len(acts)),
        "unique_active_features_topk": int(acts["feature_id"].nunique()),
        "residual_sample_rows": int(len(residual_meta)),
    }

    manifest = build_run_manifest(CFG, metrics, stage="collect_activations")
    write_run_manifest(CFG, manifest)

    print("Done")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
