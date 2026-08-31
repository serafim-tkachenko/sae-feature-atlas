"""Run the frozen source endpoint directly from frozen samples, without pooled-test latency.

Permutation generation groups movable strata by size. This changes the seeded
Monte Carlo realization, not the conditional null or the number of replicates.
The original pooled analysis is independent and remains unchanged.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

from sae_feature_atlas.scientific import regimes
from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.scientific.foundation import conjunction_table
from sae_feature_atlas.util.io import write_json


class GroupedPermuter:
    def __init__(self):
        self.strata = None

    def __call__(self, labels, strata, rng):
        if self.strata is not strata:
            grouped = {}
            for idx in strata:
                if len(np.unique(labels[idx])) == 2:
                    grouped.setdefault(len(idx), []).append(idx)
            self.groups = [np.stack(v) for v in grouped.values()]
            self.strata = strata
        result = labels.copy()
        for ids in self.groups:
            result[ids] = rng.permuted(labels[ids], axis=1)
        return result


def main(layer):
    root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
    out = root / "source_confirmation"
    out.mkdir(exist_ok=True)
    targets = pd.read_parquet(root / "geometry/targets.parquet")
    selected = targets[targets.purpose == "primary"].copy().rename(columns={"level": "regime"})
    keys = (
        selected[["text_id", "token_pos"]]
        .drop_duplicates()
        .sort_values(["text_id", "token_pos"])
        .reset_index(drop=True)
    )
    keys["matrix_row"] = np.arange(len(keys))
    rows, cols = [], []
    for batch in pq.ParquetFile(root / "sae_activations_positive.parquet").iter_batches(
        batch_size=1_000_000, columns=["text_id", "token_pos", "feature_id"]
    ):
        frame = batch.to_pandas().merge(
            keys, on=["text_id", "token_pos"], how="inner", validate="many_to_one"
        )
        rows.append(frame.matrix_row.to_numpy())
        cols.append(frame.feature_id.to_numpy())
    row, col = np.concatenate(rows), np.concatenate(cols)
    matrix = sparse.csr_matrix(
        (np.ones(len(row), dtype=np.float32), (row, col)), shape=(len(keys), 65536)
    )
    selected = selected.merge(keys, on=["text_id", "token_pos"], validate="many_to_one")
    selected["original_token_index"] = selected.token_index
    selected["token_index"] = selected.matrix_row
    cfg = regimes.RegimeConfig(
        **json.loads((root / "foundation_design.json").read_text())["regimes"]
    )
    plan = json.loads((root / "geometry/plan.json").read_text())
    write_json(
        out / "execution.json",
        {
            "method": "Grouped uniform within-stratum permutations; same frozen endpoint and 9999 replicates",
            "source_sha256": sha256(__file__),
            "plan_sha256": sha256(root / "geometry/plan.json"),
        },
    )
    regimes.permute_labels = GroupedPermuter()
    results = []
    sources = ["fineweb-edu-sample", "wikimedia-en"]
    for fid in plan["primary_ids"]:
        for source in sources:
            stem = out / f"{fid}_{source}"
            if stem.with_suffix(".json").exists():
                results.append(json.loads(stem.with_suffix(".json").read_text()))
                continue
            frame = selected[(selected.feature_id == fid) & (selected.source == source)].copy()
            result, edges, nulls, sample = regimes.analyze_feature(
                frame, matrix, fid, cfg, null_kinds=("token_identity",)
            )
            result = {"feature_id": fid, "source": source, **result}
            sample["token_index"] = sample.original_token_index
            sample.to_parquet(str(stem) + ".sample.parquet", index=False)
            pd.DataFrame(edges).to_parquet(str(stem) + ".edges.parquet", index=False)
            pd.DataFrame(nulls).to_parquet(str(stem) + ".nulls.parquet", index=False)
            write_json(stem.with_suffix(".json"), result)
            results.append(result)
            print(
                "SOURCE PRIMARY",
                layer,
                fid,
                source,
                result["status"],
                result.get("token_identity_p"),
                flush=True,
            )
    pd.DataFrame(results).to_parquet(out / "source_results.parquet", index=False)
    conjunction_table(results, plan["primary_ids"], sources).to_parquet(
        out / "conjunction.parquet", index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, choices=[17, 22], required=True)
    main(parser.parse_args().layer)
