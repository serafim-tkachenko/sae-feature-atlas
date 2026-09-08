"""Freeze residual replay locations using discovery fits and prespecified sampling."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.scientific.regimes import assign_regimes
from sae_feature_atlas.util.io import write_json


def prepare(layer):
    root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
    out = root / "geometry"
    out.mkdir(exist_ok=True)
    if (out / "plan.json").exists():
        raise ValueError("Geometry plan already frozen.")
    population = pd.read_parquet(root / "regime_token_population.parquet")
    fits = pd.read_parquet(root / "regime_feature_summary.parquet")
    candidates = fits.loc[fits.selected_candidate, "feature_id"].astype(int).tolist()
    seed = 20260908
    rng = np.random.default_rng(seed + 103)
    random_ids = sorted(
        rng.choice(fits.feature_id.to_numpy(), min(64, len(fits)), replace=False).tolist()
    )
    ids = sorted(set(candidates) | set(random_ids))
    acts = (
        ds.dataset(root / "sae_activations_positive.parquet")
        .to_table(filter=ds.field("feature_id").isin(ids))
        .to_pandas()
    )
    acts = acts.merge(population, on=["text_id", "token_pos"], validate="many_to_one")
    rows, cuts = [], []
    fit_map = fits.set_index("feature_id").to_dict("index")
    for fid in ids:
        group = acts[acts.feature_id == fid]
        train = group[group.split == "discovery"]
        evaluation = group[group.split == "evaluation"].copy()
        if fid in candidates:
            labels, _ = assign_regimes(evaluation.activation, fit_map[fid], 0.9)
            evaluation["level"] = labels
            for source, frame in evaluation[evaluation.level >= 0].groupby("source"):
                chosen = frame.groupby("duplicate_group", group_keys=False).sample(
                    n=1, random_state=seed + fid
                )
                rows.append(chosen.assign(purpose="primary"))
        if fid in random_ids:
            thresholds = np.quantile(train.activation, [0.25, 0.5, 0.75])
            cuts.append(
                {
                    "feature_id": fid,
                    "q25": thresholds[0],
                    "q50": thresholds[1],
                    "q75": thresholds[2],
                }
            )
            evaluation["level"] = np.searchsorted(thresholds, evaluation.activation, side="right")
            for source, frame in evaluation.groupby("source"):
                chosen = frame.groupby("duplicate_group", group_keys=False).sample(
                    n=1, random_state=seed + fid
                )
                rows.append(chosen.assign(purpose="quantile"))
    docs = population[["text_id", "source", "split", "duplicate_group"]].drop_duplicates()
    for (source, split), frame in docs.groupby(["source", "split"], sort=True):
        frame = frame.sort_values("text_id").drop_duplicates("duplicate_group")
        if len(frame) < 300:
            raise ValueError("Insufficient reference documents for the frozen PCA design.")
        chosen = frame.sample(n=300, random_state=seed + 109).text_id
        reference = population[
            population.text_id.isin(chosen) & (population.token_pos % 16 == 0)
        ].copy()
        rows.append(
            reference.assign(
                feature_id=-1, activation=np.nan, level=-1, purpose=f"reference_{split}"
            )
        )
    targets = pd.concat(rows, ignore_index=True)
    targets.to_parquet(out / "targets.parquet", index=False)
    pd.DataFrame(cuts).to_parquet(out / "quantile_cuts.parquet", index=False)
    write_json(
        out / "plan.json",
        {
            "layer": layer,
            "seed": seed,
            "primary_ids": candidates,
            "quantile_ids": random_ids,
            "reference_documents_per_source_split": 300,
            "targets_sha256": sha256(out / "targets.parquet"),
            "discovery_fits_sha256": sha256(root / "regime_feature_summary.parquet"),
            "geometry_protocol_sha256": sha256("docs/foundation_geometry.md"),
            "planner_sha256": sha256(__file__),
            "native_token_count": len(targets[["text_id", "token_pos"]].drop_duplicates()),
        },
    )
    print(json.loads((out / "plan.json").read_text()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=[17, 22], required=True)
    prepare(parser.parse_args().layer)
