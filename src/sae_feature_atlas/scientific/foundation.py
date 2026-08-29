"""Prospective 4B foundation experiment: fixed sources and source-wise confirmation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from sae_feature_atlas.scientific.collect import collect, configuration, sha256
from sae_feature_atlas.scientific.regimes import RegimeConfig, analyze_feature, bh_adjust
from sae_feature_atlas.scientific.run import analyze
from sae_feature_atlas.util.io import write_json


def settings(layer, corpus_path, run_prefix="gemma4b_foundation_v1"):
    count = sum(1 for line in Path(corpus_path).open() if line.strip())
    cfg = configuration(
        f"{run_prefix}_l{layer}",
        count,
        512,
        model="gemma-3-4b-pt",
        layer=layer,
        width="65k",
        l0="medium",
        corpus="foundation-two-source",
    )
    rcfg = RegimeConfig(
        seed=20260908,
        screen_features=2048,
        max_candidates=24,
        min_discovery_support=500,
        min_discovery_documents=100,
        min_regime_support=100,
        min_regime_documents=100,
        permutations=9999,
        bootstraps=1000,
    )
    return cfg, rcfg


def conjunction_table(rows, selected_ids, expected_sources):
    """Intersection-union p-value: evidence must hold in every specified source."""
    table = pd.DataFrame(rows)
    output = []
    for fid in selected_ids:
        group = table[table.feature_id == fid] if not table.empty else table
        ps = []
        for source in expected_sources:
            cell = group[group.source == source] if not group.empty else group
            if len(cell) > 1:
                raise ValueError("Duplicate source result.")
            p = 1.0
            if len(cell) == 1 and cell.iloc[0].status == "ok":
                p = float(cell.iloc[0].token_identity_p)
                if not np.isfinite(p):
                    p = 1.0
            ps.append(p)
        output.append({"feature_id": int(fid), "conjunction_p": max(ps)})
    result = pd.DataFrame(output, columns=["feature_id", "conjunction_p"])
    result["q_by"] = (
        np.minimum(
            1.0, bh_adjust(result.conjunction_p.to_numpy()) * sum(1 / np.arange(1, len(result) + 1))
        )
        if len(result)
        else []
    )
    return result


def confirm_sources(cfg, rcfg):
    root = cfg.run_data_dir
    from sae_feature_atlas.scientific.regimes import require_regime_config

    require_regime_config(root, rcfg)
    population = pd.read_parquet(root / "regime_token_population.parquet")
    acts = pd.read_parquet(cfg.sae_activations_path).merge(
        population[["text_id", "token_pos", "token_index"]],
        on=["text_id", "token_pos"],
        validate="many_to_one",
    )
    matrix = sparse.csr_matrix(
        (
            np.ones(len(acts), dtype=np.float32),
            (acts.token_index.to_numpy(), acts.feature_id.to_numpy()),
        ),
        shape=(len(population), int(acts.feature_id.max()) + 1),
    )
    del acts
    assignments = pd.read_parquet(root / "regime_assignments.parquet")
    fits = pd.read_parquet(root / "regime_feature_summary.parquet")
    ids = fits.loc[fits.selected_candidate, "feature_id"].astype(int).tolist()
    sources = ["fineweb-edu-sample", "wikimedia-en"]
    if set(population.source.unique()) != set(sources):
        raise ValueError("Foundation analysis requires the two frozen sources.")
    if not assignments.empty:
        assignments = assignments[assignments.role == "candidate"].merge(
            population[["token_index", "source", "duplicate_group"]],
            on="token_index",
            validate="many_to_one",
        )
    out = root / "source_confirmation"
    out.mkdir(exist_ok=True)
    rows = []
    for fid in ids:
        for source in sources:
            frame = assignments[(assignments.feature_id == fid) & (assignments.source == source)]
            confident = frame[frame.regime >= 0]
            selected = (
                confident.groupby("duplicate_group", group_keys=False)
                .sample(n=1, random_state=rcfg.seed + fid)
                .copy()
            )
            result, edges, nulls, selected = analyze_feature(
                selected, matrix, fid, rcfg, null_kinds=("token_identity",)
            )
            row = {"feature_id": fid, "source": source, **result}
            rows.append(row)
            stem = out / f"{fid}_{source}"
            write_json(stem.with_suffix(".json"), row)
            selected.to_parquet(str(stem) + ".sample.parquet", index=False)
            pd.DataFrame(edges).to_parquet(str(stem) + ".edges.parquet", index=False)
            pd.DataFrame(nulls).to_parquet(str(stem) + ".nulls.parquet", index=False)
            print(
                "CONFIRM", fid, source, result["status"], result.get("token_identity_p"), flush=True
            )
    pd.DataFrame(rows).to_parquet(out / "source_results.parquet", index=False)
    result = conjunction_table(rows, ids, sources)
    result.to_parquet(out / "conjunction.parquet", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/raw/foundation_v1/texts.jsonl")
    parser.add_argument("--layer", type=int, choices=[17, 22], default=17)
    parser.add_argument(
        "--stage", choices=["collect", "analyze", "confirm", "controls"], required=True
    )
    args = parser.parse_args()
    cfg, rcfg = settings(args.layer, args.corpus)
    corpus_manifest = json.loads((Path(args.corpus).parent / "corpus_manifest.json").read_text())
    if not corpus_manifest.get("near_duplicate_audit", "").startswith("completed"):
        raise ValueError("Complete the duplicate audit before foundation collection.")
    if corpus_manifest["texts_sha256"] != sha256(args.corpus):
        raise ValueError("Frozen corpus hash mismatch.")
    cfg.run_data_dir.mkdir(parents=True, exist_ok=True)
    design = {
        "experiment": asdict(cfg),
        "regimes": asdict(rcfg),
        "corpus_manifest": corpus_manifest,
        "protocol_sha256": sha256("docs/foundation_protocol.md"),
    }
    design = json.loads(json.dumps(design, default=str))
    path = cfg.run_data_dir / "foundation_design.json"
    if path.exists() and json.loads(path.read_text()) != design:
        raise ValueError("Foundation design changed; use a new run name.")
    write_json(path, design)
    if args.stage == "collect":
        collect(cfg, corpus_path=args.corpus, local_files_only=True)
    elif args.stage == "analyze":
        analyze(cfg, rcfg)
    elif args.stage == "confirm":
        confirm_sources(cfg, rcfg)
    else:
        from sae_feature_atlas.scientific.controls import weak_controls

        weak_controls(cfg, rcfg, pool_size=2048, prospective=True)


if __name__ == "__main__":
    main()
