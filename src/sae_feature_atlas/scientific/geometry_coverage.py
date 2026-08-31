"""Descriptive dictionary coverage, exact screened cooccurrence, and reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from scipy import sparse
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from sae_feature_atlas.scientific.geometry_analysis import load_residuals
from sae_feature_atlas.scientific.foundation import settings
from sae_feature_atlas.runtime.loaders import load_sae
from sae_feature_atlas.util.io import write_json


def main(layer):
    root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
    out = root / "geometry"
    fits = pd.read_parquet(root / "regime_feature_summary.parquet")
    ids = np.sort(fits.feature_id.to_numpy())
    population = pd.read_parquet(root / "regime_token_population.parquet")
    acts = (
        ds.dataset(root / "sae_activations_positive.parquet")
        .to_table(filter=ds.field("feature_id").isin(ids))
        .to_pandas()
    )
    acts = acts.merge(
        population[["text_id", "token_pos", "token_index", "split"]],
        on=["text_id", "token_pos"],
        validate="many_to_one",
    )
    col = np.searchsorted(ids, acts.feature_id)
    matrix = sparse.csr_matrix(
        (np.ones(len(acts), dtype=np.float32), (acts.token_index, col)),
        shape=(len(population), len(ids)),
    )
    docs = acts[["text_id", "feature_id"]].drop_duplicates()
    dm = sparse.csr_matrix(
        (
            np.ones(len(docs), dtype=np.float32),
            (docs.text_id, np.searchsorted(ids, docs.feature_id)),
        ),
        shape=(12000, len(ids)),
    )
    p = json.loads((root / "collection_provenance.json").read_text())
    path = hf_hub_download(
        p["sae_repo"],
        f"resid_post/layer_{layer}_width_65k_l0_medium/params.safetensors",
        revision=p["sae_revision"],
        local_files_only=True,
    )
    decoder = load_file(path)["w_dec"][ids].to("cuda", torch.float64)
    unit = decoder / decoder.norm(dim=1, keepdim=True)
    cosine = (unit @ unit.T).cpu().numpy()
    i, j = np.triu_indices(len(ids), 1)
    pair = pd.DataFrame({"feature_i": ids[i], "feature_j": ids[j], "decoder_cosine": cosine[i, j]})
    for name, m in [("token", matrix), ("document", dm)]:
        joint = (m.T @ m).toarray()
        count = np.diag(joint)
        pair[name + "_intersection"] = joint[i, j].astype(int)
        pair[name + "_jaccard"] = joint[i, j] / np.maximum(count[i] + count[j] - joint[i, j], 1)
    # All ~2M screened pairs are retained; no missing pruned edge means zero.
    pair.to_parquet(out / "screened_pair_geometry.parquet", index=False)
    for name in ["token_jaccard", "document_jaccard", "decoder_cosine"]:
        pair.nlargest(100, name).to_csv(out / f"top_{name}.csv", index=False)
    caseids = fits.loc[fits.selected_candidate, "feature_id"]
    examples = acts[acts.feature_id.isin(caseids) & (acts.split == "discovery")]
    samples = pd.concat(
        [
            x.sample(n=min(10000, len(x)), random_state=20261017)
            for _, x in examples.groupby("feature_id")
        ],
        ignore_index=True,
    )
    samples[["feature_id", "activation"]].to_parquet(
        out / "discovery_activation_examples.parquet", index=False
    )
    del acts, examples, samples, matrix, dm, pair, decoder, unit
    native, keys = load_residuals(root)
    targets = pd.read_parquet(out / "targets.parquet")
    ref = (
        targets[targets.purpose == "reference_evaluation"]
        .merge(keys, on=["text_id", "token_pos", "token_id"], validate="many_to_one")
        .drop_duplicates("native_index")
    )
    pca = np.load(out / "residual_pca.npz")
    basis = torch.as_tensor(pca["basis"], device="cuda", dtype=torch.float64)
    mean = torch.as_tensor(pca["mean"], device="cuda", dtype=torch.float64)
    cfg, _ = settings(layer, "data/raw/foundation_v1/texts.jsonl")
    sae = load_sae(cfg, "cuda")
    residual_pc, error_pc = [], []
    errors, energy = 0.0, 0.0
    for start in range(0, len(ref), 128):
        x = native[ref.native_index.to_numpy()[start : start + 128]].to("cuda")
        with torch.inference_mode():
            reconstruction = sae.decode(sae.encode(x))
        residual_pc.append(((x.double() - mean) @ basis).cpu().numpy())
        error_pc.append(((x - reconstruction).double() @ basis).cpu().numpy())
        errors += float((x - reconstruction).double().square().sum())
        energy += float(x.double().square().sum())
    xx, ee = np.concatenate(residual_pc), np.concatenate(error_pc)
    variance = np.mean((xx - xx.mean(0)) ** 2, axis=0)
    mse = np.mean(ee**2, axis=0)
    pd.DataFrame(
        {
            "pc": np.arange(1, len(variance) + 1),
            "evaluation_variance": variance,
            "reconstruction_mse": mse,
            "r_squared": 1 - mse / np.maximum(variance, 1e-30),
        }
    ).to_parquet(out / "pc_reconstruction.parquet", index=False)
    write_json(
        out / "coverage_complete.json",
        {
            "screened_features": len(ids),
            "screened_pairs": len(i),
            "reference_evaluation_locations": len(ref),
            "centered_reconstruction_r_squared": 1 - float(mse.sum() / variance.sum()),
            "uncentered_relative_mse": errors / energy,
            "pair_population": "All eligible collected tokens; descriptive, not a held-out hypothesis test; restricted to discovery-supported random screen",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=[17, 22], required=True)
    main(parser.parse_args().layer)
