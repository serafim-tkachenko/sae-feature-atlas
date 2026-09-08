"""Exploratory weak-separation controls, separate from the strict BIC control arm.

This arm was added after discovery-only matching found no strict BIC controls.
It does not replace, or retroactively validate, the original control definition.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import sparse

from sae_feature_atlas.analysis.bimodality import compute_bimodality
from sae_feature_atlas.scientific.regimes import assign_regimes, analyze_feature, match_controls


def weak_controls(cfg, rcfg, pool_size=2048, *, prospective=False):
    root = cfg.run_data_dir
    from sae_feature_atlas.scientific.regimes import require_regime_config

    require_regime_config(root, rcfg)
    tokens = pd.read_parquet(root / "regime_token_population.parquet")
    acts = pd.read_parquet(cfg.sae_activations_path).merge(
        tokens[["text_id", "token_pos", "token_index", "split"]], on=["text_id", "token_pos"]
    )
    train = acts[acts.split == "discovery"]
    counts = train.groupby("feature_id").agg(n=("activation", "size"), nd=("text_id", "nunique"))
    eligible = counts[
        (counts.n >= rcfg.min_discovery_support) & (counts.nd >= rcfg.min_discovery_documents)
    ].index.to_numpy()
    primary_fits = pd.read_parquet(root / "regime_feature_summary.parquet")
    candidates = primary_fits[primary_fits.selected_candidate]
    if candidates.empty:
        for name in ("comparison", "assignments", "nulls", "edges"):
            pd.DataFrame(columns=["feature_id", "candidate_id", "role", "status"]).to_parquet(
                root / f"regime_weak_control_{name}.parquet", index=False
            )
        return {"status": "no_candidates", "strict_control_arm_preserved": True}
    remaining = np.setdiff1d(eligible, primary_fits.feature_id)
    rng = np.random.default_rng(rcfg.seed + 1)
    ids = rng.choice(remaining, min(pool_size, len(remaining)), replace=False)
    cache = root / "regime_weak_control_fits.parquet"
    if cache.exists():
        extra = pd.read_parquet(cache)
        if set(extra.feature_id) != set(ids):
            raise ValueError("Supplementary control pool changed.")
        if not (
            extra.gmm_n_init.eq(rcfg.gmm_n_init).all() and extra.gmm_random_seed.eq(rcfg.seed).all()
        ):
            raise ValueError("Cached control GMM settings changed: use a new run name.")
    else:
        extra = compute_bimodality(
            train,
            set(ids),
            min_points=rcfg.min_discovery_support,
            n_init=rcfg.gmm_n_init,
            random_seed=rcfg.seed,
            storage_mode=cfg.collection.activation_mode,
        ).evaluated_features
        extra.to_parquet(cache, index=False)
    fits = pd.concat([primary_fits, extra], ignore_index=True)
    fits["n_documents"] = fits.feature_id.map(counts.nd)
    # Reuse the original nearest-neighbor matcher with an explicit eligibility
    # proxy; original BIC values are never modified in the exported fit table.
    proxy = fits.copy()
    weak = (proxy.fit_status == "ok") & (proxy.mode_separation < rcfg.min_separation)
    proxy["delta_bic"] = np.where(weak, 0.0, 100.0)
    matches = match_controls(proxy, candidates, rcfg.match_log_caliper)
    matches["control_definition"] = (
        "prespecified_converged_separation_below_2"
        if prospective
        else "exploratory_converged_separation_below_2"
    )
    matches.to_parquet(root / "regime_weak_matched_controls.parquet", index=False)
    design = {
        "pool_size": pool_size,
        "additional_features_fitted": len(extra),
        "seed": rcfg.seed + 1,
        "definition": "converged GMM, standardized separation <2",
        "amendment": (
            "Prespecified in the foundation protocol before this run's evaluation."
            if prospective
            else "Added after zero strict-BIC discovery support matches; exploratory arm."
        ),
        "matching": "original log-support/log-document caliper, no replacement",
        "evaluation": "same frozen discovery tail fractions; exact candidate regime counts",
        "strict_control_arm_preserved": True,
    }
    (root / "regime_weak_control_design.json").write_text(json.dumps(design, indent=2))
    primary = pd.read_parquet(root / "regime_neighborhood_comparison.parquet")
    matrix = sparse.csr_matrix(
        (
            np.ones(len(acts), dtype=np.float32),
            (acts.token_index.to_numpy(), acts.feature_id.to_numpy()),
        ),
        shape=(len(tokens), int(acts.feature_id.max()) + 1),
    )
    results, all_frames, all_nulls, all_edges = [], [], [], []
    for match in matches[matches.control_id.notna()].itertuples():
        fid, cid = int(match.feature_id), int(match.control_id)
        candidate = primary[(primary.feature_id == fid) & (primary.role == "candidate")].iloc[0]
        if candidate.status != "ok":
            continue
        fit = candidates[candidates.feature_id == fid].iloc[0]
        label, _ = assign_regimes(
            train.loc[train.feature_id == fid, "activation"], fit, rcfg.posterior_threshold
        )
        rates = [(label == r).mean() for r in (0, 1)]
        cuts = np.quantile(
            train.loc[train.feature_id == cid, "activation"], [rates[0], 1 - rates[1]]
        )
        frame = acts[(acts.feature_id == cid) & (acts.split == "evaluation")].merge(
            tokens[["token_index", "token_id", "n_positive_features"]], on="token_index"
        )
        frame["regime"] = np.where(
            frame.activation < cuts[0], 0, np.where(frame.activation > cuts[1], 1, -1)
        )
        frame["control_low_cut"], frame["control_high_cut"] = cuts
        frame["assignment_method"] = "frozen_quantile_tails"
        result, edges, nulls, frame = analyze_feature(
            frame, matrix, cid, rcfg, target_sizes=[int(candidate.n_low), int(candidate.n_high)]
        )
        result.update(
            feature_id=cid,
            candidate_id=fid,
            role="weak_control",
            discovery_delta_bic=float(fits.loc[fits.feature_id == cid, "delta_bic"].iloc[0]),
        )
        results.append(result)
        frame["candidate_id"], frame["role"] = fid, "weak_control"
        all_frames.append(frame)
        all_edges.extend({"feature_id": cid, "candidate_id": fid, **e} for e in edges)
        all_nulls.extend({"feature_id": cid, "candidate_id": fid, **n} for n in nulls)
        pd.DataFrame(results).to_parquet(
            root / "regime_weak_control_comparison.parquet", index=False
        )
        print("WEAK CONTROL", fid, cid, result["status"], result.get("js_bits"), flush=True)
    for name, frame in [
        (
            "regime_weak_control_comparison",
            pd.DataFrame(
                results,
                columns=None if results else ["feature_id", "candidate_id", "role", "status"],
            ),
        ),
        (
            "regime_weak_control_assignments",
            pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame(),
        ),
        ("regime_weak_control_nulls", pd.DataFrame(all_nulls)),
        ("regime_weak_control_edges", pd.DataFrame(all_edges)),
    ]:
        frame.to_parquet(root / f"{name}.parquet", index=False)
    return design
