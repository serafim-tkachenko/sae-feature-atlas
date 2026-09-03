"""Prepare a portable development pilot from cached native residuals; no GPU run."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.scientific.intervention_math import matched_contrast, null_projector
from sae_feature_atlas.util.io import write_json


def bucket(group, seed):
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def cached_states(root):
    """Check source chunks and keep one position key, without loading all as FP32."""
    matrices, keys = [], []
    for done in sorted((root / "geometry/residual_chunks").glob("*.done.json")):
        for name, digest in json.loads(done.read_text()).items():
            if sha256(done.parent / name) != digest:
                raise ValueError(f"Corrupt residual chunk: {name}")
        stem = done.name.split(".")[0]
        matrices.append(np.load(done.parent / f"{stem}.npy"))
        keys.append(pd.read_parquet(done.parent / f"{stem}.parquet"))
    frame = pd.concat(keys, ignore_index=True)
    if frame.duplicated(["text_id", "token_pos"]).any():
        raise ValueError("Duplicate native keys")
    frame["native_index"] = np.arange(len(frame))
    return np.concatenate(matrices), frame


def prepare(root, config_path, out):
    if out.exists():
        raise ValueError("Use a new output directory; preparation is immutable")
    cfg = json.loads(config_path.read_text())
    provenance = json.loads((root / "collection_provenance.json").read_text())
    if cfg["layer"] != provenance["model"]["layer"]:
        raise ValueError("Layer mismatch")
    if provenance["sae_native_config"]["architecture"] != "jump_relu":
        raise ValueError("Pilot supports the native affine JumpReLU encoder only")
    if "normalize_activations='none'" not in provenance["sae_config"]:
        raise ValueError("Unsupported input normalization")
    path = hf_hub_download(
        provenance["sae_repo"],
        f"resid_post/{provenance['model']['sae_id']}/params.safetensors",
        revision=provenance["sae_revision"],
        local_files_only=True,
    )
    if sha256(path) != provenance["sae_weights_sha256"]:
        raise ValueError("SAE checkpoint differs from collection")
    bits, keys = cached_states(root)
    population = pd.read_parquet(root / "regime_token_population.parquet")
    keys = keys.merge(
        population.drop(columns=["token_id"], errors="ignore"),
        on=["text_id", "token_pos"],
        validate="one_to_one",
    )
    keys["pilot_split"] = [
        "fit" if bucket(g, cfg["seed"]) < cfg["fit_fraction"] else "check"
        for g in keys.duplicate_group
    ]
    acts = (
        ds.dataset(root / "sae_activations_positive.parquet")
        .to_table(
            filter=ds.field("feature_id").isin(cfg["features"]),
            columns=["text_id", "token_pos", "feature_id", "activation"],
        )
        .to_pandas()
        .merge(keys, on=["text_id", "token_pos"], validate="many_to_one")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        provenance["model"]["model_name"],
        revision=provenance["model_revision"],
        local_files_only=True,
    )
    readouts = []
    for spec in cfg["readouts"]:
        item = dict(spec)
        for side in ("positive", "negative"):
            ids = [tokenizer.encode(s, add_special_tokens=False) for s in spec[side]]
            if any(len(x) != 1 for x in ids):
                raise ValueError(f"Readout strings must be single tokens: {spec['name']}")
            item[side + "_ids"] = [x[0] for x in ids]
        if set(item["positive_ids"]) & set(item["negative_ids"]):
            raise ValueError("Overlapping readout classes")
        readouts.append(item)
    texts = {
        int(x["text_id"]): x
        for x in map(json.loads, (root / "source_texts.jsonl").read_text().splitlines())
    }
    pc = np.load(root / "geometry/residual_pca.npz")
    vectors, records, summaries = {}, [], []
    vectors["reference_pc8"] = pc["basis"][:, :8]
    rng = np.random.default_rng(cfg["seed"])
    with safe_open(path, framework="numpy") as weights:
        for fid in cfg["features"]:
            frame = acts[acts.feature_id == fid].copy()
            fit = frame[frame.pilot_split == "fit"].copy()
            lo, hi = np.quantile(fit.activation, [0.25, 0.75])
            if hi <= lo:
                summaries.append(dict(feature_id=fid, status="collapsed_quartiles"))
                continue
            fit = fit[(fit.activation <= lo) | (fit.activation >= hi)].copy()
            labels = (fit.activation >= hi).to_numpy().astype(int)
            x = torch.from_numpy(bits[fit.native_index.to_numpy()]).view(torch.bfloat16)
            x = x.float().numpy()
            strata = list(
                zip(fit.source, fit.token_id, fit.token_pos // 64, fit.n_positive_features // 16)
            )
            try:
                c, support = matched_contrast(x, labels, strata)
            except ValueError:
                summaries.append(dict(feature_id=fid, status="no_overlap"))
                continue
            if support < cfg["minimum_overlap"]:
                summaries.append(
                    dict(feature_id=fid, status="insufficient_overlap", support=support)
                )
                continue
            w = weights.get_slice("w_enc")[:, fid].astype(np.float64)
            decoder = weights.get_slice("w_dec")[fid].astype(np.float64)
            u = decoder / np.linalg.norm(decoder)
            q = null_projector(w, u)
            projected = q @ c
            if np.linalg.norm(projected) < 1e-10:
                summaries.append(dict(feature_id=fid, status="zero_projected_contrast"))
                continue
            directions = [projected / np.linalg.norm(projected)]
            names = ["learned"]
            for i in range(cfg["random_directions"]):
                v = q @ rng.normal(size=len(w))
                directions.append(v / np.linalg.norm(v))
                names.append(f"random_{i}")
            # Match PCA mass before reapplying constraints; matching is approximate afterward.
            v = q @ (pc["basis"] @ ((pc["basis"].T @ directions[0]) * rng.choice([-1, 1], len(w))))
            directions.append(v / np.linalg.norm(v))
            names.append("approx_spectrum_matched")
            v = q @ pc["basis"][:, 0]
            directions.append(v / np.linalg.norm(v))
            names.append("leading_pc")
            shuffled = labels.copy()
            groups = {}
            for i, key in enumerate(strata):
                groups.setdefault(key, []).append(i)
            for indices in groups.values():
                indices = np.asarray(indices)
                shuffled[indices] = rng.permutation(shuffled[indices])
            shuffled_c, _ = matched_contrast(x, shuffled, strata)
            v = q @ shuffled_c
            if np.linalg.norm(v) > 1e-10:
                directions.append(v / np.linalg.norm(v))
                names.append("shuffled_labels")
            scale = float(
                np.median(frame[frame.pilot_split == "fit"].activation) * np.linalg.norm(decoder)
            )
            vectors[f"{fid}_w"] = w
            vectors[f"{fid}_u"] = u
            vectors[f"{fid}_directions"] = np.stack(directions)
            vectors[f"{fid}_b"] = weights.get_slice("b_enc")[fid]
            vectors[f"{fid}_threshold"] = weights.get_slice("threshold")[fid]
            for (source, split), group in frame.groupby(["source", "pilot_split"]):
                group = group.sort_values(["text_id", "token_pos"]).drop_duplicates(
                    "duplicate_group"
                )
                chosen = group.sample(
                    n=min(len(group), cfg["prompts_per_feature_source_split"]),
                    random_state=cfg["seed"] + fid,
                )
                for row in chosen.itertuples():
                    encoded = tokenizer(
                        texts[row.text_id]["text"],
                        truncation=True,
                        max_length=provenance["collection"]["max_seq_len"],
                    )["input_ids"]
                    if encoded[row.token_pos] != row.token_id:
                        raise ValueError("Tokenizer replay mismatch")
                    prefix = encoded[: row.token_pos + 1]
                    decoded = tokenizer.decode(prefix, skip_special_tokens=True)
                    number = re.search(r"[0-9]+$", decoded)
                    records.append(
                        dict(
                            feature_id=fid,
                            source=source,
                            split=split,
                            text_id=int(row.text_id),
                            token_pos=int(row.token_pos),
                            token_id=int(row.token_id),
                            duplicate_group=str(row.duplicate_group),
                            input_ids=prefix,
                            next_token_id=int(encoded[row.token_pos + 1])
                            if row.token_pos + 1 < len(encoded)
                            else None,
                            cached_activation=float(row.activation),
                            n_positive_features=int(row.n_positive_features),
                            trailing_digit_count=len(number.group()) if number else 0,
                            excerpt=decoded[-200:],
                        )
                    )
            summaries.append(
                dict(
                    feature_id=fid,
                    status="ok",
                    overlap=support,
                    q25=float(lo),
                    q75=float(hi),
                    projected_fraction=float(np.linalg.norm(projected) / np.linalg.norm(c)),
                    directions=names,
                    decoder_scale=scale,
                )
            )
    if not records:
        raise ValueError("No supported features: do not relax the design silently")
    out.mkdir(parents=True)
    np.savez(out / "vectors.npz", **vectors)
    (out / "prompts.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    write_json(out / "config.json", cfg)
    write_json(
        out / "plan.json",
        dict(
            status=cfg["status"],
            model=provenance["model"],
            model_revision=provenance["model_revision"],
            sae_revision=provenance["sae_revision"],
            source_provenance_sha256=sha256(root / "collection_provenance.json"),
            preparation_sha256=sha256(__file__),
            features=summaries,
            readouts=readouts,
            fit_rule="Duplicate-group hash; all old data are development, including check split",
            direction_rule="Overlap-weighted high/low quartile means within source/token/position64/support16",
            prompts=len(records),
            files={
                name: sha256(out / name) for name in ["vectors.npz", "prompts.jsonl", "config.json"]
            },
        ),
    )
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("data/processed/gemma4b_foundation_v1_l17")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/context_intervention_pilot.json")
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root, args.config, args.out)
