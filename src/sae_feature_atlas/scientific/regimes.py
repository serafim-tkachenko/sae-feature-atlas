"""Held-out, support-aware comparisons of within-feature activation regimes.

Raw sparse membership, never a pruned edge table, defines partner observations.
The primary test conditions on document, position bin and total positive support.
It tests conditional exchangeability, not causality or discrete semantic identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp, rel_entr
from scipy.stats import spearmanr

from sae_feature_atlas.analysis.bimodality import compute_bimodality
from sae_feature_atlas.analysis.coactivation import canonical_pair
from sae_feature_atlas.analysis.populations import TOKEN_KEY, validate_sparse_activation_rows


@dataclass(frozen=True)
class RegimeConfig:
    seed: int = 20260907
    discovery_fraction: float = 0.5
    screen_features: int = 512
    min_discovery_support: int = 200
    min_discovery_documents: int = 30
    max_candidates: int = 24
    posterior_threshold: float = 0.9
    min_regime_support: int = 60
    min_regime_documents: int = 20
    min_partner_support: int = 10
    neighbor_k: int = 20
    permutations: int = 1999
    bootstraps: int = 300
    position_bin: int = 64
    support_bin: int = 16
    match_log_caliper: float = 0.35
    gmm_n_init: int = 5
    delta_bic_threshold: float = 10.0
    min_component_weight: float = 0.1
    min_separation: float = 2.0

    def __post_init__(self):
        if not 0.5 < self.posterior_threshold <= 1:
            raise ValueError("Posterior threshold must be in (0.5, 1].")
        if not 0 < self.discovery_fraction < 1:
            raise ValueError("Discovery fraction must be in (0, 1).")
        if self.bootstraps < 2:
            raise ValueError("At least two bootstrap replicates are required for a standard error.")
        for field in (
            "screen_features",
            "min_discovery_support",
            "min_discovery_documents",
            "max_candidates",
            "min_regime_support",
            "min_regime_documents",
            "min_partner_support",
            "neighbor_k",
            "permutations",
            "bootstraps",
            "position_bin",
            "support_bin",
            "gmm_n_init",
        ):
            if getattr(self, field) < 1:
                raise ValueError(f"{field} must be positive.")


def require_regime_config(root, cfg):
    observed = json.loads((Path(root) / "regime_design.json").read_text())["config"]
    if observed != asdict(cfg):
        raise ValueError("Regime configuration mismatch: rerun analysis or use a new run name.")


def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must be finite and in [0, 1].")
    order = np.argsort(p, kind="stable")
    out = np.empty(len(p))
    out[order] = np.minimum(
        1, np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    )
    return out


def assign_regimes(values, fit, threshold=0.9):
    """Apply the frozen discovery GMM; preserve ambiguous observations as -1."""
    if not 0.5 < threshold <= 1:
        raise ValueError("Threshold must be in (0.5, 1].")
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("GMM assignments require finite positive values.")
    x = np.log1p(values)
    means = np.array([fit["log_mean_low"], fit["log_mean_high"]])
    variances = np.array([fit["log_variance_low"], fit["log_variance_high"]])
    weights = np.array([fit["component_weight_low"], fit["component_weight_high"]])
    lp = np.log(weights) - 0.5 * (
        np.log(2 * np.pi * variances) + (x[:, None] - means) ** 2 / variances
    )
    probabilities = np.exp(lp - logsumexp(lp, axis=1, keepdims=True))
    label = np.where(
        probabilities[:, 0] >= threshold, 0, np.where(probabilities[:, 1] >= threshold, 1, -1)
    )
    return label, probabilities


def js_divergence(a, b):
    """Jensen-Shannon divergence in bits; NaN if either distribution is empty."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if (a < 0).any() or (b < 0).any():
        raise ValueError("Partner weights cannot be negative.")
    if a.sum() == 0 or b.sum() == 0:
        return float("nan")
    p, q = a / a.sum(), b / b.sum()
    m = (p + q) / 2
    return float((rel_entr(p, m).sum() + rel_entr(q, m).sum()) / (2 * np.log(2)))


def neighborhood_metrics(low, high, n_low, n_high, k):
    low, high = np.asarray(low, float), np.asarray(high, float)
    p, q = low / n_low, high / n_high
    # Zero-count edges are not neighbors. Stable feature-id ordering breaks ties.
    li = [i for i in np.argsort(-low, kind="stable") if low[i] > 0][:k]
    hi = [i for i in np.argsort(-high, kind="stable") if high[i] > 0][:k]
    shared = sorted(set(li) & set(hi))
    union = set(li) | set(hi)
    corr = float("nan")
    if len(shared) >= 3 and np.std(low[shared]) > 0 and np.std(high[shared]) > 0:
        corr = float(spearmanr(low[shared], high[shared]).statistic)
    denom = np.linalg.norm(p) * np.linalg.norm(q)
    return {
        "js_bits": js_divergence(low, high),
        "neighbor_jaccard": len(shared) / len(union) if union else np.nan,
        "shared_rank_spearman": corr,
        "shared_neighbors": len(shared),
        "cosine": float(p @ q / denom) if denom else np.nan,
        "conditional_l1": float(np.abs(p - q).sum()),
        "conditional_mass_change": float(q.sum() - p.sum()),
    }


def permute_labels(labels, strata, rng):
    """Exact regime counts within each stratum, including frozen strata."""
    out = np.array(labels, copy=True)
    for idx in strata:
        out[idx] = rng.permutation(out[idx])
    return out


def match_controls(fits, candidates, caliper):
    """Greedy, no-replacement nearest matching on discovery log support/documents.

    Frequency is redundant with support on this common token universe. A weak
    control has delta BIC < 10 and a converged fit, not merely poor separation.
    """
    controls = fits[(fits.fit_status == "ok") & (fits.delta_bic < 10)].copy()
    used, rows = set(), []
    for row in candidates.sort_values(["delta_bic", "feature_id"], ascending=[False, True]).to_dict(
        "records"
    ):
        pool = controls[~controls.feature_id.isin(used)].copy()
        pool["log_support_gap"] = np.log(pool.n_points / row["n_points"])
        pool["log_documents_gap"] = np.log(pool.n_documents / row["n_documents"])
        pool = pool[
            (pool.log_support_gap.abs() <= caliper) & (pool.log_documents_gap.abs() <= caliper)
        ]
        if pool.empty:
            rows.append(
                {
                    "feature_id": row["feature_id"],
                    "control_id": None,
                    "match_status": "no_control_within_caliper",
                }
            )
            continue
        pool["distance"] = np.hypot(pool.log_support_gap, pool.log_documents_gap)
        best = pool.sort_values(["distance", "feature_id"]).iloc[0]
        used.add(int(best.feature_id))
        rows.append(
            {
                "feature_id": row["feature_id"],
                "control_id": int(best.feature_id),
                "match_status": "matched",
                "match_distance": best.distance,
                "log_support_gap": best.log_support_gap,
                "log_documents_gap": best.log_documents_gap,
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "feature_id",
            "control_id",
            "match_status",
            "match_distance",
            "log_support_gap",
            "log_documents_gap",
        ],
    )


def _strata(frame, cfg, kind):
    f = frame.assign(
        position_bin=frame.token_pos // cfg.position_bin,
        support_bin=frame.n_positive_features // cfg.support_bin,
    )
    keys = ["text_id", "position_bin", "support_bin"]
    if kind == "token_identity":
        keys = ["token_id", "position_bin", "support_bin"]
    return list(f.groupby(keys, sort=True).indices.values())


def analyze_feature(frame, matrix, feature_id, cfg, *, target_sizes=None):
    """Evaluate a frozen assignment on held-out observations only."""
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(feature_id)]))
    selected = frame[frame.regime >= 0].copy()
    if target_sizes is not None:
        if any((selected.regime == r).sum() < n for r, n in enumerate(target_sizes)):
            return {"status": "insufficient_matched_regime_support"}, [], [], frame
        selected = pd.concat(
            [
                selected[selected.regime == r].sample(n=n, random_state=cfg.seed)
                for r, n in enumerate(target_sizes)
            ]
        )
    selected = selected.sort_values(TOKEN_KEY).reset_index(drop=True)
    labels = selected.regime.to_numpy()
    ns = [int((labels == r).sum()) for r in (0, 1)]
    nd = [int(selected.loc[labels == r, "text_id"].nunique()) for r in (0, 1)]
    base = {
        "n_low": ns[0],
        "n_high": ns[1],
        "documents_low": nd[0],
        "documents_high": nd[1],
        "assignment_rate": float((frame.regime >= 0).mean()),
    }
    if min(ns) < cfg.min_regime_support or min(nd) < cfg.min_regime_documents:
        return {**base, "status": "insufficient_regime_support"}, [], [], frame
    x = matrix[selected.token_index.to_numpy()]
    pooled = np.asarray(x.sum(axis=0)).ravel()
    partners = np.flatnonzero(
        (pooled >= cfg.min_partner_support) & (np.arange(matrix.shape[1]) != feature_id)
    )
    x = x[:, partners].astype(float)
    lo = np.asarray(x[labels == 0].sum(axis=0)).ravel()
    hi = np.asarray(x[labels == 1].sum(axis=0)).ravel()
    metrics = neighborhood_metrics(lo, hi, *ns, cfg.neighbor_k)
    if not np.isfinite(metrics["js_bits"]):
        return {**base, "status": "no_supported_partner_distribution"}, [], [], frame
    nulls = []
    for kind in ("document", "token_identity"):
        strata = _strata(selected, cfg, kind)
        movable = sum(len(i) for i in strata if len(np.unique(labels[i])) == 2)
        values = []
        # Sparse multiply amortizes permutations without dense token x feature arrays.
        for start in range(0, cfg.permutations, 64):
            batch = np.column_stack(
                [
                    permute_labels(labels, strata, rng) == 1
                    for _ in range(min(64, cfg.permutations - start))
                ]
            )
            high = np.asarray(x.T @ batch.astype(float))
            total = lo + hi
            for b in range(high.shape[1]):
                value = js_divergence(total - high[:, b], high[:, b])
                values.append(value)
                nulls.append({"null_kind": kind, "iteration": len(values) - 1, "js_bits": value})
        values = np.array(values)
        base.update(
            {
                f"{kind}_p": float(
                    (1 + (values >= metrics["js_bits"] - 1e-12).sum()) / (len(values) + 1)
                ),
                f"{kind}_null_mean": float(values.mean()),
                f"{kind}_null_p95": float(np.quantile(values, 0.95)),
                f"{kind}_movable_fraction": movable / len(selected),
            }
        )
    # Document bootstrap preserves token dependence; GMM and partner set fixed.
    docs = selected.text_id.unique()
    doc_idx = pd.Categorical(selected.text_id, categories=docs).codes
    incidence = sparse.csr_matrix(
        (np.ones(len(selected)), (doc_idx, np.arange(len(selected)))),
        shape=(len(docs), len(selected)),
    )
    low_by_doc = incidence @ x.multiply((labels == 0)[:, None])
    high_by_doc = incidence @ x.multiply((labels == 1)[:, None])
    ci_values = []
    for _ in range(cfg.bootstraps):
        counts = rng.multinomial(len(docs), np.ones(len(docs)) / len(docs))
        ci_values.append(
            js_divergence(
                np.asarray(counts @ low_by_doc).ravel(), np.asarray(counts @ high_by_doc).ravel()
            )
        )
    finite_ci = np.array(ci_values)[np.isfinite(ci_values)]
    ci = np.quantile(finite_ci, [0.025, 0.975]) if len(finite_ci) else [np.nan, np.nan]
    se = float(np.std(finite_ci, ddof=1)) if len(finite_ci) > 1 else np.nan
    base.update(
        js_ci_low=max(0.0, metrics["js_bits"] - 1.96 * se),
        js_ci_high=min(1.0, metrics["js_bits"] + 1.96 * se),
        js_bootstrap_se=se,
        js_bootstrap_bias=float(np.mean(finite_ci) - metrics["js_bits"]),
        js_bootstrap_percentile_low=ci[0],
        js_bootstrap_percentile_high=ci[1],
        bootstrap_valid=len(finite_ci),
        null_excess=metrics["js_bits"] - base["document_null_mean"],
    )
    edges = []
    for r, counts, n in [("low", lo, ns[0]), ("high", hi, ns[1])]:
        for j, c in zip(partners, counts):
            i0, j0 = canonical_pair(feature_id, j)
            edges.append(
                {
                    "regime": r,
                    "partner_id": int(j),
                    "feature_i": i0,
                    "feature_j": j0,
                    "count": int(c),
                    "regime_support": n,
                    "conditional_probability": c / n,
                    "edge_status": "observed_zero" if c == 0 else "observed",
                    "pooled_support_filter": cfg.min_partner_support,
                }
            )
    # Score all examples against each regime centroid, including ambiguous cases.
    all_x = matrix[frame.token_index.to_numpy()][:, partners]
    pl, ph = lo / ns[0], hi / ns[1]
    norms = np.sqrt(np.asarray(all_x.multiply(all_x).sum(axis=1)).ravel())
    sim_l = np.asarray(all_x @ pl).ravel() / np.maximum(norms * np.linalg.norm(pl), 1e-12)
    sim_h = np.asarray(all_x @ ph).ravel() / np.maximum(norms * np.linalg.norm(ph), 1e-12)
    frame = frame.assign(
        low_centroid_cosine=sim_l,
        high_centroid_cosine=sim_h,
        used_for_comparison=frame.token_index.isin(selected.token_index),
    )
    return {**base, **metrics, "status": "ok", "partner_count": len(partners)}, edges, nulls, frame


def run_regimes(acts, tokens, support, cfg, output_dir, *, storage_mode="positive"):
    """Full discovery/evaluation experiment; write per-feature recovery checkpoints."""
    validate_sparse_activation_rows(acts)
    if not np.isfinite(acts.activation).all() or (acts.activation <= 0).any():
        raise ValueError("Only finite positive analysis rows may enter this experiment.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokens = tokens.sort_values(TOKEN_KEY).reset_index(drop=True).copy()
    if tokens.duplicated(TOKEN_KEY).any():
        raise ValueError("Duplicate tokens.")
    tokens["token_index"] = np.arange(len(tokens))
    tokens = tokens.merge(
        support[TOKEN_KEY + ["n_positive_features"]], on=TOKEN_KEY, validate="one_to_one"
    )
    if len(tokens) == 0:
        raise ValueError("No eligible tokens.")
    acts = acts.merge(tokens[TOKEN_KEY + ["token_index"]], on=TOKEN_KEY, validate="many_to_one")
    docs = np.sort(tokens.text_id.unique())
    rng = np.random.default_rng(cfg.seed)
    discovery = set(rng.permutation(docs)[: int(len(docs) * cfg.discovery_fraction)])
    tokens["split"] = np.where(tokens.text_id.isin(discovery), "discovery", "evaluation")
    tokens.to_parquet(output / "regime_token_population.parquet", index=False)
    train = acts[acts.text_id.isin(discovery)]
    counts = train.groupby("feature_id").agg(n=("activation", "size"), nd=("text_id", "nunique"))
    eligible = counts[
        (counts.n >= cfg.min_discovery_support) & (counts.nd >= cfg.min_discovery_documents)
    ].index.to_numpy()
    ids = np.sort(rng.choice(eligible, min(len(eligible), cfg.screen_features), replace=False))
    fits = compute_bimodality(
        train,
        set(ids),
        min_points=cfg.min_discovery_support,
        delta_bic_threshold=cfg.delta_bic_threshold,
        min_component_weight=cfg.min_component_weight,
        min_separation=cfg.min_separation,
        n_init=cfg.gmm_n_init,
        random_seed=cfg.seed,
        storage_mode=storage_mode,
    ).evaluated_features
    if fits.empty:
        raise ValueError("No features have discovery support; increase corpus size.")
    fits["n_documents"] = fits.feature_id.map(counts.nd)
    fits["discovery_variance_log1p"] = fits.feature_id.map(
        train.assign(log_a=np.log1p(train.activation)).groupby("feature_id").log_a.var()
    )
    candidates = (
        fits[fits.is_bimodal_candidate]
        .sort_values(["delta_bic", "feature_id"], ascending=[False, True])
        .head(cfg.max_candidates)
    )
    fits["selected_candidate"] = fits.feature_id.isin(candidates.feature_id)
    fits.to_parquet(output / "regime_feature_summary.parquet", index=False)
    matches = match_controls(fits, candidates, cfg.match_log_caliper)
    matches.to_parquet(output / "regime_matched_controls.parquet", index=False)
    info = {
        "config": asdict(cfg),
        "eligible_discovery_features": len(eligible),
        "screened_features": len(fits),
        "selected_candidates": len(candidates),
        "discovery_documents": len(discovery),
        "evaluation_documents": len(docs) - len(discovery),
        "eligible_tokens": len(tokens),
        "storage_mode": storage_mode,
        "screen_selection": "uniform seeded subset of discovery-supported features",
        "primary_statistic": "Jensen-Shannon divergence in bits",
        "primary_null": "within-document x position-bin x positive-support-bin label permutation",
        "secondary_null": "within-token-identity x position-bin x positive-support-bin permutation",
        "ineligible_candidates_in_fdr": "p=1",
        "multiple_testing": "BH and BY over selected candidates",
        "partner_population": "all raw stored latents with pooled confident evaluation count >= threshold",
        "control_population": "converged discovery delta BIC <10; frozen discovery quantile tails",
    }
    (output / "regime_design.json").write_text(json.dumps(info, indent=2))
    width = int(acts.feature_id.max()) + 1
    matrix = sparse.csr_matrix(
        (
            np.ones(len(acts), dtype=np.float32),
            (acts.token_index.to_numpy(), acts.feature_id.to_numpy()),
        ),
        shape=(len(tokens), width),
    )
    eval_acts = acts[~acts.text_id.isin(discovery)]
    groups = {
        int(f): g
        for f, g in eval_acts[
            eval_acts.feature_id.isin(
                set(candidates.feature_id) | set(matches.control_id.dropna().astype(int))
            )
        ].groupby("feature_id")
    }
    fit_map = fits.set_index("feature_id").to_dict("index")
    comparisons, assignments, edges_all, nulls_all = [], [], [], []
    for candidate in candidates.to_dict("records"):
        fid = int(candidate["feature_id"])
        jobs = [(fid, "candidate")]
        match = matches[matches.feature_id == fid].iloc[0]
        if pd.notna(match.control_id):
            jobs.append((int(match.control_id), "control"))
        target_sizes = None
        discovery_labels, _ = assign_regimes(
            train.loc[train.feature_id == fid, "activation"], candidate, cfg.posterior_threshold
        )
        tail_rates = [(discovery_labels == r).mean() for r in (0, 1)]
        for current_id, role in jobs:
            frame = groups.get(current_id, eval_acts.iloc[:0]).copy()
            frame = frame.merge(
                tokens[["token_index", "token_id", "n_positive_features"]],
                on="token_index",
                validate="many_to_one",
            )
            label, post = assign_regimes(
                frame.activation, fit_map[current_id], cfg.posterior_threshold
            )
            if role == "control":
                values = train.loc[train.feature_id == current_id, "activation"].to_numpy()
                low_cut, high_cut = np.quantile(values, [tail_rates[0], 1 - tail_rates[1]])
                label = np.where(
                    frame.activation < low_cut, 0, np.where(frame.activation > high_cut, 1, -1)
                )
                frame["control_low_cut"] = low_cut
                frame["control_high_cut"] = high_cut
            frame["regime"] = label
            frame["posterior_low"], frame["posterior_high"] = post[:, 0], post[:, 1]
            frame["assignment_method"] = (
                "frozen_gmm" if role == "candidate" else "frozen_quantile_tails"
            )
            result, edges, nulls, frame = analyze_feature(
                frame,
                matrix,
                current_id,
                cfg,
                target_sizes=target_sizes if role == "control" else None,
            )
            result.update(
                feature_id=current_id,
                candidate_id=fid,
                role=role,
                discovery_delta_bic=fit_map[current_id].get("delta_bic"),
            )
            comparisons.append(result)
            frame["role"], frame["candidate_id"] = role, fid
            assignments.append(frame)
            edges_all.extend(
                {"feature_id": current_id, "candidate_id": fid, "role": role, **e} for e in edges
            )
            nulls_all.extend(
                {"feature_id": current_id, "candidate_id": fid, "role": role, **n} for n in nulls
            )
            checkpoint = output / "regime_checkpoints"
            checkpoint.mkdir(exist_ok=True)
            (checkpoint / f"{current_id}.json").write_text(pd.Series(result).to_json(indent=2))
            frame.to_parquet(checkpoint / f"{current_id}.assignments.parquet", index=False)
            pd.DataFrame(nulls).to_parquet(checkpoint / f"{current_id}.null.parquet", index=False)
            pd.DataFrame(edges).to_parquet(checkpoint / f"{current_id}.edges.parquet", index=False)
            print("REGIME", current_id, role, result["status"], result.get("js_bits"), flush=True)
            if role == "candidate":
                if result["status"] != "ok":
                    break
                target_sizes = [result["n_low"], result["n_high"]]
    comp = pd.DataFrame(
        comparisons,
        columns=None
        if comparisons
        else ["feature_id", "candidate_id", "role", "status", "document_p"],
    )
    if not comp.empty:
        for kind in ("document", "token_identity"):
            if f"{kind}_p" not in comp:
                comp[f"{kind}_p"] = np.nan
            mask = comp.role == "candidate"
            q = bh_adjust(comp.loc[mask, f"{kind}_p"].fillna(1))
            comp.loc[mask, f"{kind}_q_bh"] = q
            comp.loc[mask, f"{kind}_q_by"] = np.minimum(1, q * sum(1 / np.arange(1, len(q) + 1)))
    for name, table in [
        ("regime_neighborhood_comparison", comp),
        (
            "regime_assignments",
            pd.concat(assignments, ignore_index=True)
            if assignments
            else pd.DataFrame(
                columns=[
                    "feature_id",
                    "text_id",
                    "token_pos",
                    "regime",
                    "posterior_low",
                    "posterior_high",
                ]
            ),
        ),
        (
            "regime_coactivation",
            pd.DataFrame(
                edges_all,
                columns=None
                if edges_all
                else ["feature_id", "partner_id", "regime", "count", "conditional_probability"],
            ),
        ),
        (
            "regime_null_results",
            pd.DataFrame(
                nulls_all,
                columns=None if nulls_all else ["feature_id", "null_kind", "iteration", "js_bits"],
            ),
        ),
    ]:
        table.to_parquet(output / f"{name}.parquet", index=False)
    return info
