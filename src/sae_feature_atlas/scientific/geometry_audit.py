"""Verify replayed native residuals reproduce stored SAE activations at sampled keys."""

import argparse
import numpy as np
import pyarrow.dataset as ds
import torch
from sae_feature_atlas.scientific.geometry_analysis import load_residuals
from sae_feature_atlas.scientific.foundation import settings
from sae_feature_atlas.runtime.loaders import load_sae
from sae_feature_atlas.util.io import write_json


def main(layer):
    cfg, _ = settings(layer, "data/raw/foundation_v1/texts.jsonl")
    root = cfg.run_data_dir
    native, keys = load_residuals(root)
    chosen = keys.sample(n=32, random_state=20261025).reset_index(drop=True)
    chosen["row"] = np.arange(len(chosen))
    stored = (
        ds.dataset(root / "sae_activations_positive.parquet")
        .to_table(filter=ds.field("text_id").isin(chosen.text_id.tolist()))
        .to_pandas()
    )
    stored = stored.merge(
        chosen[["text_id", "token_pos", "row"]], on=["text_id", "token_pos"], validate="many_to_one"
    )
    expected = torch.zeros((32, 65536), device="cuda")
    expected[stored.row.to_numpy(), stored.feature_id.to_numpy()] = torch.as_tensor(
        stored.activation.to_numpy().copy(), device="cuda"
    )
    sae = load_sae(cfg, "cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    with torch.inference_mode():
        actual = sae.encode(native[chosen.native_index.to_numpy()].to("cuda"))
    error = (actual - expected).abs()
    mismatch = int(((actual > 0) != (expected > 0)).sum())
    passed = bool(torch.allclose(actual, expected, atol=0.03, rtol=1e-4)) and mismatch == 0
    record = {
        "sampled_locations": 32,
        "seed": 20261025,
        "max_absolute_error": float(error.max()),
        "maximum_expected_activation": float(expected.max()),
        "positive_membership_disagreements": mismatch,
        "absolute_tolerance": 0.03,
        "relative_tolerance": 1e-4,
        "passed": passed,
        "purpose": "Replay hook and coordinate consistency across all dictionary entries at seeded locations",
    }
    write_json(root / "geometry/replay_activation_audit.json", record)
    print("REPLAY AUDIT", layer, record, flush=True)
    if not passed:
        raise ValueError("Native replay does not reproduce stored SAE activation sample")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, choices=[17, 22], required=True)
    main(p.parse_args().layer)
