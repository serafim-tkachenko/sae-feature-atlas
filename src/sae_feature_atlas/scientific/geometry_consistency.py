"""Descriptive decoder neighborhoods and binary-partner/native-shift agreement."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sae_feature_atlas.scientific.geometry_analysis import load_residuals
from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.util.io import write_json


def main(layer):
    root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
    out = root / "geometry"
    native, keys = load_residuals(root)
    targets = pd.read_parquet(out / "targets.parquet")
    targets = targets[targets.purpose == "primary"].merge(
        keys[["text_id", "token_pos", "native_index"]],
        on=["text_id", "token_pos"],
        validate="many_to_one",
    )
    p = json.loads((root / "collection_provenance.json").read_text())
    path = hf_hub_download(
        p["sae_repo"],
        f"resid_post/layer_{layer}_width_65k_l0_medium/params.safetensors",
        revision=p["sae_revision"],
        local_files_only=True,
    )
    weights = load_file(path)
    decoder = weights["w_dec"].to("cuda", torch.float64)
    del weights
    unit = decoder / decoder.norm(dim=1, keepdim=True)
    centered = unit - unit.mean(0)
    eigenvalues, basis = torch.linalg.eigh(centered.T @ centered / (len(unit) - 1))
    projection = (centered @ basis[:, -2:].flip(1)).cpu().numpy()
    pd.DataFrame(
        {"feature_id": np.arange(len(unit)), "pc1": projection[:, 0], "pc2": projection[:, 1]}
    ).to_parquet(out / "decoder_pca.parquet", index=False)
    explained = float(eigenvalues[-2:].sum() / eigenvalues.sum())
    del centered, basis, eigenvalues
    rows = []
    for fid, group in targets.groupby("feature_id"):
        similarity = unit @ unit[int(fid)]
        similarity[int(fid)] = -torch.inf
        nearest = set(torch.topk(similarity, 20).indices.cpu().tolist())
        for source, frame in group.groupby("source"):
            low, high = frame[frame.level == 0], frame[frame.level == 1]
            if min(len(low), len(high)) < 100:
                continue
            edges = pd.read_parquet(root / f"source_confirmation/{fid}_{source}.edges.parquet")
            wide = edges.pivot(
                index="partner_id", columns="regime", values="conditional_probability"
            )
            delta = (wide.high - wide.low).to_numpy()
            binary = torch.as_tensor(delta, device="cuda") @ decoder[wide.index.to_numpy()]
            x = native[high.native_index.to_numpy()].double().mean(0) - native[
                low.native_index.to_numpy()
            ].double().mean(0)
            x = x.to("cuda")
            direction = unit[int(fid)]
            bx = binary - direction * (binary @ direction)
            xx = x - direction * (x @ direction)

            def cosine(a, b):
                denominator = float(a.norm() * b.norm())
                return float(a @ b / denominator) if denominator > 1e-12 else np.nan

            low_neighbors = set(wide.low.nlargest(20).index)
            high_neighbors = set(wide.high.nlargest(20).index)
            rows.append(
                {
                    "feature_id": fid,
                    "source": source,
                    "binary_native_cosine": cosine(binary, x),
                    "binary_native_orthogonal_cosine": cosine(bx, xx),
                    "decoder_low_neighbor_jaccard": len(nearest & low_neighbors)
                    / len(nearest | low_neighbors),
                    "decoder_high_neighbor_jaccard": len(nearest & high_neighbors)
                    / len(nearest | high_neighbors),
                    "decoder_neighbor_ids": json.dumps(sorted(nearest)),
                }
            )
    pd.DataFrame(rows).to_parquet(out / "decoder_context_consistency.parquet", index=False)
    write_json(
        out / "consistency_complete.json",
        {
            "layer": layer,
            "source_sha256": sha256(__file__),
            "decoder_pca_variance_2d": explained,
            "status": "Prespecified descriptive checks; no significance test",
            "binary_partner_sum": "Sum decoder vectors weighted by signed changes in positive membership probability; not an amplitude-weighted SAE reconstruction",
        },
    )
    print("CONSISTENCY COMPLETE", layer, explained, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=[17, 22], required=True)
    main(parser.parse_args().layer)
