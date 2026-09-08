"""Post-result sensitivity: remove the discovery covariance-times-encoder direction.

This diagnostic is explicitly exploratory and does not replace frozen endpoints.
For an elliptical Gaussian reference, conditioning on a linear encoder score
produces mean displacement parallel to Cw. Removing span(decoder,Cw) addresses
that simple alternative, without claiming a complete conditional generative model.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sae_feature_atlas.scientific.geometry_analysis import load_residuals, permutation_shift
from sae_feature_atlas.scientific.regimes import RegimeConfig, _strata, bh_adjust
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
    pc = np.load(out / "residual_pca.npz")
    basis = torch.as_tensor(pc["basis"], device="cuda", dtype=torch.float64)
    eigenvalues = torch.as_tensor(pc["eigenvalues"], device="cuda", dtype=torch.float64)
    cfg = RegimeConfig(**json.loads((root / "foundation_design.json").read_text())["regimes"])
    plan = json.loads((out / "plan.json").read_text())
    rows = []
    for fid in plan["primary_ids"]:
        u = weights["w_dec"][fid].to("cuda", torch.float64)
        u = u / u.norm()
        w = weights["w_enc"][:, fid].to("cuda", torch.float64)
        covariance_direction = basis @ (eigenvalues * (basis.T @ w))
        v = covariance_direction - u * (u @ covariance_direction)
        additional_norm = float(v.norm())
        if additional_norm > 1e-12:
            v = v / v.norm()
        for source in ["fineweb-edu-sample", "wikimedia-en"]:
            frame = (
                targets[(targets.feature_id == fid) & (targets.source == source)]
                .sort_values(["text_id", "token_pos"])
                .reset_index(drop=True)
            )
            labels = frame.level.to_numpy()
            ns = [int((labels == r).sum()) for r in (0, 1)]
            row = {
                "feature_id": fid,
                "source": source,
                "n_low": ns[0],
                "n_high": ns[1],
                "encoder_decoder_cosine": float((w / w.norm()) @ u),
            }
            if min(ns) < 100:
                rows.append({**row, "status": "insufficient_regime_support", "p": 1.0})
                continue
            x = native[frame.native_index.to_numpy()].to("cuda", torch.float64)
            delta = x[labels == 1].mean(0) - x[labels == 0].mean(0)
            row["shift_covariance_encoder_cosine"] = float(
                (delta @ covariance_direction) / (delta.norm() * covariance_direction.norm())
            )
            if additional_norm > 1e-12:
                x = x - (x @ v)[:, None] * v
            result, _, nulls = permutation_shift(
                x,
                labels,
                u,
                _strata(frame, cfg, "token_identity"),
                seed=cfg.seed + fid + 314,
                permutations=4999,
            )
            rows.append({**row, "status": "ok", **result})
            np.save(out / f"{fid}_{source}.encoder_control_null.npy", np.asarray(nulls))
            print("ENCODER CONTROL", layer, fid, source, result["p"], flush=True)
    table = pd.DataFrame(rows)
    table.to_parquet(out / "encoder_control_source.parquet", index=False)
    q = table.groupby("feature_id").p.max().rename("conjunction_p").reset_index()
    q["q_by"] = np.minimum(
        1, bh_adjust(q.conjunction_p.to_numpy()) * sum(1 / np.arange(1, len(q) + 1))
    )
    q.to_parquet(out / "encoder_control_conjunction.parquet", index=False)
    write_json(
        out / "encoder_control_design.json",
        {
            "status": "Exploratory sensitivity added after primary and orthogonal outcomes were inspected",
            "direction": "Remove span of focal decoder and full discovery residual covariance times focal encoder",
            "covariance": "Same frozen discovery reference PCA covariance",
            "permutations": 4999,
            "seed_offset": 314,
            "family": "BY across all selected candidates per layer; source conjunction",
            "code_sha256": sha256(__file__),
            "interpretation": "Simple linear covariance alternative, not a complete conditional generative null",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, choices=[17, 22], required=True)
    main(p.parse_args().layer)
