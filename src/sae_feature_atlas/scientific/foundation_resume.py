"""Resume pooled secondary comparisons with cached fits and grouped permutations."""

import argparse
import json
from types import SimpleNamespace
import pandas as pd

from sae_feature_atlas.scientific import regimes
from sae_feature_atlas.scientific.foundation import settings
from sae_feature_atlas.scientific.geometry_primary import GroupedPermuter
from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.util.io import write_json


def main(layer):
    cfg, rcfg = settings(layer, "data/raw/foundation_v1/texts.jsonl")
    root = cfg.run_data_dir
    frozen = (root / "regime_feature_summary.parquet").read_bytes()
    fits = pd.read_parquet(root / "regime_feature_summary.parquet")
    original = regimes.analyze_feature

    def cached(frame, matrix, feature_id, config, **kwargs):
        stem = root / "regime_checkpoints" / str(feature_id)
        names = [
            stem.with_suffix(".json"),
            str(stem) + ".assignments.parquet",
            str(stem) + ".edges.parquet",
            str(stem) + ".null.parquet",
        ]
        from pathlib import Path

        if all(Path(p).exists() for p in names):
            sample = pd.read_parquet(names[1])
            if set(sample.token_index) != set(frame.token_index):
                raise ValueError("Checkpoint token population changed")
            return (
                json.loads(names[0].read_text()),
                pd.read_parquet(names[2]).to_dict("records"),
                pd.read_parquet(names[3]).to_dict("records"),
                sample,
            )
        return original(frame, matrix, feature_id, config, **kwargs)

    regimes.analyze_feature = cached
    regimes.permute_labels = GroupedPermuter()

    def cached_fits(train, ids, **kwargs):
        if set(ids) != set(fits.feature_id):
            raise ValueError("Frozen discovery screen changed")
        return SimpleNamespace(evaluated_features=fits.copy())

    regimes.compute_bimodality = cached_fits
    tokens = pd.read_parquet(root / "regime_token_population.parquet")
    support = pd.read_parquet(root / "token_activation_summary.parquet")
    acts = pd.read_parquet(cfg.sae_activations_path).merge(
        tokens[["text_id", "token_pos"]], on=["text_id", "token_pos"], validate="many_to_one"
    )
    tokens = tokens.drop(columns=["token_index", "n_positive_features"])
    write_json(
        root / "pooled_execution.json",
        {
            "source_sha256": sha256(__file__),
            "method": "Frozen fits and completed checkpoints retained; grouped conditional permutations for remaining pooled comparisons",
        },
    )
    try:
        result = regimes.run_regimes(acts, tokens, support, rcfg, root, storage_mode="positive")
        write_json(root / "regime_run_summary.json", result)
    finally:
        (root / "regime_feature_summary.parquet").write_bytes(frozen)
    del acts, tokens, support
    import gc

    gc.collect()
    from sae_feature_atlas.scientific import controls

    controls.analyze_feature = original
    controls.weak_controls(cfg, rcfg, pool_size=2048, prospective=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=[17, 22], required=True)
    main(parser.parse_args().layer)
