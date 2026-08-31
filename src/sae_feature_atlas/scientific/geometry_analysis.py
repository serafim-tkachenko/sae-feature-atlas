"""Discovery-fitted native geometry and conditional orthogonal-shift tests."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.scientific.regimes import RegimeConfig, _strata, bh_adjust
from sae_feature_atlas.util.io import write_json


def participation(values):
    """Scale-invariant effective coordinates of row vectors in a stated basis."""
    mass = values.square()
    mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-30)
    effective = 1 / mass.square().sum(1)
    entropy = torch.exp(-(mass * mass.clamp_min(1e-30).log()).sum(1))
    n90 = (mass.sort(dim=1, descending=True).values.cumsum(1) < 0.9).sum(1) + 1
    return effective.cpu().numpy(), entropy.cpu().numpy(), n90.cpu().numpy()


def orthogonal_shift(x, labels, direction):
    delta = x[labels == 1].mean(0) - x[labels == 0].mean(0)
    parallel = delta @ direction
    perpendicular = delta - parallel * direction
    return delta, perpendicular, parallel


def permutation_shift(x, labels, direction, strata, *, seed, permutations=4999):
    """Conditional randomization; all full-space calculations use float64."""
    labels = np.asarray(labels)
    delta, perpendicular, parallel = orthogonal_shift(x, labels, direction)
    observed = float(perpendicular.square().sum())
    total = x.sum(0)
    n1, n0 = int((labels == 1).sum()), int((labels == 0).sum())
    rng = np.random.default_rng(seed)
    from sae_feature_atlas.scientific.geometry_primary import GroupedPermuter

    permute = GroupedPermuter()
    nulls = []
    for start in range(0, permutations, 64):
        y = np.stack([permute(labels, strata, rng) for _ in range(min(64, permutations - start))])
        sums = torch.as_tensor(y, dtype=x.dtype, device=x.device) @ x
        changes = sums / n1 - (total - sums) / n0
        orth = changes - (changes @ direction)[:, None] * direction
        nulls.extend(orth.square().sum(1).cpu().tolist())
    tolerance = 1e-10 * max(1.0, abs(observed))
    p = (1 + np.sum(np.asarray(nulls) >= observed - tolerance)) / (1 + permutations)
    movable = sum(len(i) for i in strata if len(np.unique(labels[i])) > 1) / len(labels)
    return (
        {
            "orthogonal_squared": observed,
            "total_squared": float(delta.square().sum()),
            "parallel_squared": float(parallel.square()),
            "p": float(p),
            "null_mean": float(np.mean(nulls)),
            "movable_fraction": movable,
            "tie_tolerance": tolerance,
        },
        perpendicular,
        nulls,
    )


def load_residuals(root):
    folder = root / "geometry/residual_chunks"
    arrays, frames = [], []
    for p in sorted(folder.glob("*.done.json")):
        hashes = json.loads(p.read_text())
        for name, digest in hashes.items():
            if sha256(folder / name) != digest:
                raise ValueError("Corrupt residual replay")
        stem = p.name.split(".")[0]
        arrays.append(np.load(folder / (stem + ".npy")))
        frames.append(pd.read_parquet(folder / (stem + ".parquet")))
    bits = np.concatenate(arrays)
    keys = pd.concat(frames, ignore_index=True)
    if keys.duplicated(["text_id", "token_pos"]).any():
        raise ValueError("Duplicate replay locations")
    keys["native_index"] = np.arange(len(keys))
    return torch.from_numpy(bits).view(torch.bfloat16).float(), keys


def analyze(layer):
    root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
    out = root / "geometry"
    if not (out / "replay_complete.json").exists():
        raise ValueError("Replay incomplete")
    torch.backends.cuda.matmul.allow_tf32 = False
    native, keys = load_residuals(root)
    targets = pd.read_parquet(out / "targets.parquet").merge(
        keys[["text_id", "token_pos", "native_index"]],
        on=["text_id", "token_pos"],
        validate="many_to_one",
    )
    if targets.native_index.isna().any():
        raise ValueError("Missing replay locations")
    p = json.loads((root / "collection_provenance.json").read_text())
    path = hf_hub_download(
        p["sae_repo"],
        f"resid_post/layer_{layer}_width_65k_l0_medium/params.safetensors",
        revision=p["sae_revision"],
        local_files_only=True,
    )
    weights = load_file(path)
    decoder = weights["w_dec"].float()
    if decoder.shape[1] != native.shape[1]:
        raise ValueError("Native decoder orientation mismatch")
    discovery = targets[targets.purpose == "reference_discovery"].drop_duplicates("native_index")
    x = native[discovery.native_index.to_numpy()].to("cuda", torch.float64)
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / (len(x) - 1)
    eigenvalues, basis = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.flip(0).clamp_min(0)
    basis = basis.flip(1)
    np.savez(
        out / "residual_pca.npz",
        eigenvalues=eigenvalues.cpu().numpy(),
        basis=basis.cpu().numpy(),
        mean=mean.cpu().numpy(),
    )
    rows = []
    for start in range(0, len(decoder), 1024):
        d = decoder[start : start + 1024].to("cuda", torch.float64)
        norms = d.norm(dim=1)
        unit = d / norms[:, None]
        projected = unit @ basis
        table = {
            "feature_id": np.arange(start, start + len(d)),
            "decoder_norm": norms.cpu().numpy(),
        }
        for name, v in [("raw", unit), ("pca", projected)] + [
            (f"white_{floor:g}", projected / eigenvalues.clamp_min(floor * eigenvalues[0]).sqrt())
            for floor in (0.0001, 0.001, 0.01)
        ]:
            pr, ent, n90 = participation(v)
            table.update({name + "_pr": pr, name + "_entropy_dim": ent, name + "_n90": n90})
        mass = projected.square()
        for k in (1, 5, 20, 64):
            table[f"pc_head_{k}"] = mass[:, :k].sum(1).cpu().numpy()
        table["pc_tail_quarter"] = mass[:, -len(mean) // 4 :].sum(1).cpu().numpy()
        rows.append(pd.DataFrame(table))
    pd.concat(rows).to_parquet(out / "decoder_metrics.parquet", index=False)
    del x, centered, covariance, weights
    # Small projections are descriptive; no inferential distances use truncated PCA.
    projections = []
    for start in range(0, len(native), 4096):
        projections.append(
            ((native[start : start + 4096].to("cuda", torch.float64) - mean) @ basis[:, :8])
            .cpu()
            .numpy()
        )
    projected = pd.DataFrame(np.concatenate(projections), columns=[f"pc{i + 1}" for i in range(8)])
    pd.concat([keys, projected], axis=1).to_parquet(out / "native_projections.parquet", index=False)
    plan = json.loads((out / "plan.json").read_text())
    rcfg = RegimeConfig(**json.loads((root / "foundation_design.json").read_text())["regimes"])
    results, directions = [], {}
    primary = targets[targets.purpose == "primary"]
    for fid in plan["primary_ids"]:
        directions[fid] = {}
        for source in ("fineweb-edu-sample", "wikimedia-en"):
            frame = (
                primary[(primary.feature_id == fid) & (primary.source == source)]
                .sort_values(["text_id", "token_pos"])
                .reset_index(drop=True)
            )
            labels = frame.level.to_numpy()
            sizes = [int((labels == r).sum()) for r in (0, 1)]
            row = {"feature_id": fid, "source": source, "n_low": sizes[0], "n_high": sizes[1]}
            if min(sizes) < 100:
                results.append({**row, "status": "insufficient_regime_support", "p": 1.0})
                continue
            # Use exactly the primary sample; fail rather than silently substitute.
            sample = root / f"source_confirmation/{fid}_{source}.sample.parquet"
            if sample.exists():
                expected = pd.read_parquet(sample).sort_values(["text_id", "token_pos"])
                if not np.array_equal(
                    expected[["text_id", "token_pos"]].to_numpy(),
                    frame[["text_id", "token_pos"]].to_numpy(),
                ):
                    raise ValueError("Geometry differs from primary source sample")
            xx = native[frame.native_index.to_numpy()].to("cuda", torch.float64)
            direction = decoder[fid].to("cuda", torch.float64)
            direction /= direction.norm()
            result, orth, nulls = permutation_shift(
                xx, labels, direction, _strata(frame, rcfg, "token_identity"), seed=rcfg.seed + fid
            )
            directions[fid][source] = orth.cpu().numpy()
            np.save(out / f"{fid}_{source}.orthogonal_null.npy", np.asarray(nulls))
            results.append({**row, "status": "ok", **result})
            pd.DataFrame(results).to_parquet(out / "orthogonal_source_results.parquet", index=False)
            print("ORTHOGONAL", layer, fid, source, result["p"], flush=True)
    pd.DataFrame(results).to_parquet(out / "orthogonal_source_results.parquet", index=False)
    table = pd.DataFrame(results)
    conjunction = table.groupby("feature_id").p.max().rename("conjunction_p").reset_index()
    conjunction["q_by"] = np.minimum(
        1,
        bh_adjust(conjunction.conjunction_p.to_numpy())
        * np.sum(1 / np.arange(1, len(conjunction) + 1)),
    )
    cosines = {}
    for fid, values in directions.items():
        if len(values) == 2:
            a, b = values.values()
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cosines[fid] = float(a @ b / denom) if denom > 1e-12 else np.nan
    conjunction["cross_source_cosine"] = conjunction.feature_id.map(cosines)
    conjunction.to_parquet(out / "orthogonal_conjunction.parquet", index=False)
    quantiles = targets[targets.purpose == "quantile"].merge(
        projected.assign(native_index=np.arange(len(projected))),
        on="native_index",
        validate="many_to_one",
    )
    quantiles.groupby(["feature_id", "source", "level"]).agg(
        n=("native_index", "size"),
        activation_mean=("activation", "mean"),
        **{f"pc{i}_mean": (f"pc{i}", "mean") for i in range(1, 9)},
    ).reset_index().to_parquet(out / "quantile_geometry.parquet", index=False)
    write_json(
        out / "analysis_complete.json",
        {
            "native_locations": len(native),
            "discovery_reference_locations": len(discovery),
            "dimension": len(mean),
            "analysis_sha256": sha256(__file__),
            "plan_sha256": sha256(out / "plan.json"),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=[17, 22], required=True)
    analyze(parser.parse_args().layer)
