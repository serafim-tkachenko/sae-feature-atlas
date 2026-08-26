"""Data-driven scientific report, evidence cards, and publication-quality figures."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from sae_feature_atlas.inspection.context import ContextRenderer
from sae_feature_atlas.runtime.loaders import tokenizer_decoder


def table(frame, columns, n=24):
    columns = [c for c in columns if c in frame]
    if frame.empty:
        return "No eligible observations."

    def fmt(v):
        if isinstance(v, float):
            return "NA" if not np.isfinite(v) else f"{v:.4g}"
        return str(v).replace("|", "/").replace("\n", " ")

    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    rows += [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in frame[columns].head(n).itertuples(index=False, name=None)
    ]
    return "\n".join(rows)


def examples(cfg, comp):
    root = cfg.run_data_dir
    assigned = pd.read_parquet(root / "regime_assignments.parquet")
    tokens = pd.read_parquet(cfg.token_metadata_path)
    provenance = json.loads((root / "collection_provenance.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_name, revision=provenance["model_revision"]
    )
    renderer = ContextRenderer(tokens, tokenizer_decoder(tokenizer))
    raw = {
        r["text_id"]: r["text"]
        for r in map(json.loads, cfg.source_texts_path.read_text().splitlines())
    }
    acts = pd.read_parquet(cfg.sae_activations_path)
    target_keys = assigned[["text_id", "token_pos"]].drop_duplicates()
    observed = acts.merge(target_keys, on=["text_id", "token_pos"])
    neighbors = {
        key: g.sort_values(["activation", "feature_id"], ascending=[False, True])
        for key, g in observed.groupby(["text_id", "token_pos"])
    }
    rows = []
    for fid in comp.loc[comp.status == "ok", "feature_id"].astype(int):
        frame = assigned[assigned.feature_id == fid].copy()
        if "low_centroid_cosine" not in frame:
            continue
        selected = set()
        for regime in (0, 1, -1):
            group = frame[frame.regime == regime].copy()
            if group.empty:
                continue
            own = "low_centroid_cosine" if regime == 0 else "high_centroid_cosine"
            other = "high_centroid_cosine" if regime == 0 else "low_centroid_cosine"
            group["representativeness"] = (
                group[own] if regime != -1 else -(group.posterior_low - 0.5).abs()
            )
            methods = (
                ["centroid_representative", "counterexample"] if regime >= 0 else ["ambiguous"]
            )
            for method in methods:
                group["order"] = (
                    group[other] - group[own]
                    if method == "counterexample"
                    else group.representativeness
                )
                ordered = group.sort_values(
                    ["order", "text_id", "token_pos"], ascending=[False, True, True]
                )
                ordered = (
                    ordered[~ordered.token_index.isin(selected)].drop_duplicates("text_id").head(3)
                )
                for row in ordered.to_dict("records"):
                    selected.add(row["token_index"])
                    context = renderer.render(int(row["text_id"]), int(row["token_pos"]), 20)
                    co = neighbors[(row["text_id"], row["token_pos"])]
                    co = co[co.feature_id != fid].head(20)
                    rows.append(
                        {
                            **row,
                            **context,
                            "selection_method": method,
                            "regime_name": {0: "low", 1: "high", -1: "ambiguous"}[regime],
                            "raw_text": raw[int(row["text_id"])],
                            "coactivating_features_json": co[["feature_id", "activation"]].to_json(
                                orient="records"
                            ),
                            "counterexample_meaning": "closest to opposite centroid; not a validated semantic contradiction",
                        }
                    )
    out = pd.DataFrame(
        rows, columns=None if rows else ["feature_id", "regime_name", "display_context"]
    )
    out.to_parquet(root / "regime_context_examples.parquet", index=False)
    return out


def enrich_edges(cfg):
    root = cfg.run_data_dir
    edge = pd.read_parquet(root / "regime_coactivation.parquet")
    if edge.empty:
        return
    population = pd.read_parquet(root / "regime_token_population.parquet")
    evaluation = population[population.split == "evaluation"][["text_id", "token_pos"]]
    acts = pd.read_parquet(cfg.sae_activations_path, columns=["text_id", "token_pos", "feature_id"])
    acts = acts.merge(evaluation, on=["text_id", "token_pos"])
    counts = acts.groupby("feature_id").size()
    edge["evaluation_tokens"] = len(evaluation)
    edge["partner_evaluation_count"] = edge.partner_id.map(counts)
    edge["regime_event_jaccard"] = edge["count"] / (
        edge.regime_support + edge.partner_evaluation_count - edge["count"]
    )
    threshold = json.loads((root / "regime_design.json").read_text())["config"][
        "min_partner_support"
    ]
    edge["pmi_nats"] = np.where(
        edge["count"] >= threshold,
        np.log(
            np.maximum(edge["count"], 1)
            * len(evaluation)
            / (edge.regime_support * edge.partner_evaluation_count)
        ),
        np.nan,
    )
    edge["pmi_status"] = np.where(
        edge["count"] >= threshold, "supported", "insufficient_regime_pair_count"
    )
    edge.to_parquet(root / "regime_coactivation.parquet", index=False)


def figures(root, report_dir, comp, pairs, fits):
    plots = report_dir / "figures"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "font.family": "DejaVu Sans",
        }
    )
    c = comp[(comp.role == "candidate") & (comp.status == "ok")]
    paths = []
    if c.empty:
        return paths
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    a = ax.ravel()
    a[0].scatter(c.discovery_delta_bic, c.js_bits, c="#176b87", s=30)
    a[0].set(
        xlabel="Discovery delta BIC",
        ylabel="Held-out JS divergence (bits)",
        title="A  Mixture evidence vs relational effect",
    )
    a[1].scatter(c.n_low + c.n_high, c.null_excess, c="#176b87", s=30)
    a[1].axhline(0, color="gray", lw=0.8)
    a[1].set(
        xlabel="Confident held-out observations",
        ylabel="JS minus mean document null (bits)",
        title="B  Support and excess divergence",
    )
    a[2].scatter(c.document_null_mean, c.js_bits, c="#176b87", s=30)
    limit = max(c.document_null_mean.max(), c.js_bits.max()) * 1.05
    a[2].plot([0, limit], [0, limit], "--", c="gray")
    a[2].set(
        xlabel="Mean document-null JS (bits)",
        ylabel="Observed JS (bits)",
        title="C  Observed vs conditional null",
    )
    if not pairs.empty:
        for row in pairs.itertuples():
            a[3].plot([0, 1], [row.candidate_js, row.control_js], color="gray", alpha=0.45)
        a[3].scatter(np.zeros(len(pairs)), pairs.candidate_js, c="#176b87")
        a[3].scatter(np.ones(len(pairs)), pairs.control_js, c="#ce7745")
    a[3].set(
        xticks=[0, 1],
        xticklabels=["Candidate", "Matched control"],
        ylabel="JS divergence (bits)",
        title=f"D  Equal regime counts ({len(pairs)} pairs)",
    )
    a[4].hist(c.document_q_bh, bins=np.linspace(0, 1, 11), color="#176b87", alpha=0.8, label="BH")
    a[4].axvline(0.05, color="#ce7745", linestyle="--")
    a[4].set(
        xlabel="Primary BH-adjusted q",
        ylabel="Candidate count",
        title="E  Multiple-testing-aware evidence",
    )
    a[5].hist(c.neighbor_jaccard, bins=np.linspace(0, 1, 11), color="#176b87")
    a[5].set(
        xlabel="Top-20 neighbor Jaccard", ylabel="Candidate count", title="F  Neighborhood overlap"
    )
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"aggregate.{ext}")
    plt.close(fig)
    paths.append("figures/aggregate.png")
    ranked = c.sort_values(
        ["document_q_by", "null_excess", "feature_id"], ascending=[True, False, True]
    ).head(12)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for i, row in enumerate(ranked.itertuples()):
        ax.plot([row.js_ci_low, row.js_ci_high], [i, i], c="#176b87", lw=2)
        ax.scatter(row.js_bits, i, c="#176b87", s=35)
        ax.scatter(row.document_null_mean, i, c="#ce7745", marker="x", s=45)
    ax.set(
        yticks=np.arange(len(ranked)),
        yticklabels=ranked.feature_id.astype(str),
        xlabel="Jensen-Shannon divergence (bits)",
        ylabel="SAE latent ID",
        title="Observed effects and document-bootstrap intervals\nBlue: observed / 95% interval; orange: mean conditional null",
    )
    ax.invert_yaxis()
    for ext in ("png", "pdf"):
        fig.savefig(plots / f"candidate_effects.{ext}")
    plt.close(fig)
    paths.append("figures/candidate_effects.png")
    assignments = pd.read_parquet(root / "regime_assignments.parquet")
    for fid in ranked.head(3).feature_id.astype(int):
        f = fits[fits.feature_id == fid].iloc[0]
        obs = assignments[assignments.feature_id == fid]
        fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
        for r, name, color in [
            (0, "Low", "#176b87"),
            (1, "High", "#ce7745"),
            (-1, "Ambiguous", "#aaaaaa"),
        ]:
            ax.hist(
                np.log1p(obs.loc[obs.regime == r, "activation"]),
                bins=40,
                alpha=0.6,
                label=name,
                color=color,
            )
        ax.set(
            xlabel="log(1 + activation)",
            ylabel="Held-out observations",
            title=f"Latent {fid}: frozen discovery GMM assignments (delta BIC {f.delta_bic:.1f})",
        )
        ax.legend(frameon=False)
        for ext in ("png", "pdf"):
            fig.savefig(plots / f"feature_{fid}_activation.{ext}")
        plt.close(fig)
        paths.append(f"figures/feature_{fid}_activation.png")
    return paths


def report(cfg, rcfg):
    root, dest = cfg.run_data_dir, cfg.run_reports_dir
    from sae_feature_atlas.scientific.regimes import require_regime_config

    require_regime_config(root, rcfg)
    dest.mkdir(parents=True, exist_ok=True)
    comp = pd.read_parquet(root / "regime_neighborhood_comparison.parquet")
    for column in (
        "js_bits",
        "null_excess",
        "document_q_bh",
        "document_q_by",
        "token_identity_q_by",
    ):
        if column not in comp:
            comp[column] = np.nan
    fits = pd.read_parquet(root / "regime_feature_summary.parquet")
    matches = pd.read_parquet(root / "regime_matched_controls.parquet")
    summary = json.loads((root / "regime_run_summary.json").read_text())
    prov = json.loads((root / "collection_provenance.json").read_text())
    candidates = comp[comp.role == "candidate"].copy()
    valid = candidates[candidates.status == "ok"].copy()
    controls = comp[(comp.role == "control") & (comp.status == "ok")]
    supplemental = root / "regime_weak_control_comparison.parquet"
    control_family = "strict weak-BIC"
    if supplemental.exists():
        controls = pd.read_parquet(supplemental)
        controls = controls[controls.status == "ok"]
        for column in ("js_bits", "null_excess"):
            if column not in controls:
                controls[column] = np.nan
        control_family = "exploratory weak-separation"
    pairs = (
        valid[["feature_id", "js_bits", "null_excess"]]
        .rename(
            columns={
                "feature_id": "candidate_id",
                "js_bits": "candidate_js",
                "null_excess": "candidate_excess",
            }
        )
        .merge(
            controls[["candidate_id", "feature_id", "js_bits", "null_excess"]].rename(
                columns={
                    "feature_id": "control_id",
                    "js_bits": "control_js",
                    "null_excess": "control_excess",
                }
            ),
            on="candidate_id",
        )
    )
    pairs["paired_js_difference"] = pairs.candidate_js - pairs.control_js
    pairs["paired_excess_difference"] = pairs.candidate_excess - pairs.control_excess
    pairs.to_parquet(root / "regime_control_comparison.parquet", index=False)
    enrich_edges(cfg)
    context = examples(cfg, comp)
    figures(root, dest, comp, pairs, fits)
    tables = dest / "tables"
    tables.mkdir(exist_ok=True)
    for name in (
        "regime_feature_summary",
        "regime_neighborhood_comparison",
        "regime_matched_controls",
        "regime_control_comparison",
    ):
        pd.read_parquet(root / f"{name}.parquet").to_csv(tables / f"{name}.csv", index=False)
    if not context.empty:
        # Blind review file strips regime and feature identity. Key remains separate.
        blind = context.sample(frac=1, random_state=rcfg.seed).reset_index(drop=True)
        blind["example_id"] = ["E%05d" % i for i in range(len(blind))]
        blind[["example_id", "feature_id", "regime_name", "selection_method"]].to_csv(
            tables / "annotation_key.csv", index=False
        )
        blind[["example_id", "display_context", "center_token"]].assign(
            contextual_category="", annotator="", confidence="", notes=""
        ).to_csv(tables / "blind_annotation.csv", index=False)
    n_bh = int((valid.get("document_q_bh", pd.Series(dtype=float)) <= 0.05).sum())
    n_by = int((valid.get("document_q_by", pd.Series(dtype=float)) <= 0.05).sum())
    n_token = int((valid.get("token_identity_q_by", pd.Series(dtype=float)) <= 0.05).sum())
    robustness_path = root / "regime_one_per_document.parquet"
    robustness_text = "The one-observation-per-document sensitivity has not been run."
    if robustness_path.exists():
        robustness = pd.read_parquet(robustness_path)
        robustness_text = (
            f"A sensitivity analysis retained one uniformly sampled confident observation per document and feature, "
            f"without consulting the regime or partner values during sampling. "
            f"{int((robustness.status == 'ok').sum())}/{len(robustness)} features retained minimum regime support; "
            f"{int((robustness.q_by <= 0.05).sum())}/{len(robustness)} passed BY q <=0.05 under the token-identity/position/support null, "
            "counting support failures as p=1. This reduces within-document pseudoreplication but remains conditional on "
            "cross-document exchangeability and coarse support matching. It is an exploratory sensitivity, not independent replication."
        )
    median = float(valid.js_bits.median()) if not valid.empty else np.nan
    excess = float(valid.null_excess.median()) if not valid.empty else np.nan
    conclusion = (
        f"We find evidence that activation regimes differ in conditional coactivation structure for {n_by} of {len(candidates)} selected latents under the document-stratified null (BY q <= 0.05)."
        if n_by
        else "We do not find multiplicity-corrected evidence against the document-stratified regime-exchangeability null in this pilot."
    )
    control_text = (
        f"{len(pairs)} {control_family} support-matched pairs were evaluable with identical low/high sample sizes. "
        f"The median candidate-minus-control JS difference was {pairs.paired_js_difference.median():.4f} bits; "
        f"{int((pairs.paired_js_difference > 0).sum())}/{len(pairs)} candidates exceeded their controls. "
        f"The median difference in null-excess JS was {pairs.paired_excess_difference.median():.4f} bits. "
        "These are descriptive contrasts across selected, dependent features, without an independence-based population test."
        if len(pairs)
        else "No matched pair met all evaluation support requirements. The claim that mixture-like features exceed comparable weak-mixture controls is untested in this run."
    )
    text = f"""# Activation magnitude and SAE neighborhood structure

## A document-held-out pilot with Gemma 3 1B and Gemma Scope 2

Research report | {cfg.collection.run_name} | Real-model GPU experiment | Exploratory, not peer reviewed

## Abstract

We tested whether low- and high-activation observations of the same sparse-autoencoder latent have different same-token coactivation neighborhoods. We collected all positive activations from {prov["text_count"]:,} unique documents ({summary["stored_tokens"]:,} tokens; {summary["analysis_tokens"]:,} eligible tokens) using google/gemma-3-1b-pt and a 16,384-latent, target-L0-60 Gemma Scope 2 SAE at native layer 13. Discovery documents determined mixture fits and candidate selection; independent evaluation documents supplied relational measurements. Of {summary["screened_features"]} screened latents, {int(fits.is_bimodal_candidate.sum())} met mixture qualification, {len(candidates)} were selected, and {len(valid)} met held-out support criteria. Median observed Jensen-Shannon divergence was {median:.4f} bits, with median excess above the conditional null of {excess:.4f} bits. {n_bh} candidates passed BH and {n_by} passed the more conservative BY adjustment at 0.05. {control_text} {conclusion} This does not establish semantic polysemy, discrete contextual identities, or causal feature interactions.

## Research question and estimands

For a fixed latent f, let A_f(t)>0 denote its activation on an eligible token t and R_f(t) its posterior-confident low/high regime under a discovery-fitted mixture. The central estimand is P(A_j(t)>0 | A_f(t)>0, R_f(t)=r, token eligible), for each partner j. The JS statistic compares normalized vectors of these conditional probabilities. It weights partner occurrences rather than treating the partner set as a mutually exclusive categorical token label. It therefore describes relational composition, while L1 and signed conditional-mass change separately measure total membership differences.

The population is the selected, deduplicated Pile-10k document sample, truncated to the first {cfg.collection.max_seq_len} tokenizer tokens per document, with the repository's explicit clean-target policy. The inference target is the selected discovery candidates, not every SAE feature or all natural language. The GMM estimates the distribution of log(1+A) among eligible positive observations; zeros are outside that population.

## Hypotheses

The distributional screen asks whether two Gaussian components describe positive log1p activations better by BIC than one component. This is mixture-likeness, not a formal proof of two density modes. The primary relational null states that regime labels are exchangeable within document, position-bin, and positive-support-bin strata, conditional on the frozen confident observation set and stratum regime counts. Its alternative is excess JS divergence. A secondary null conditions on token identity, position, and support instead of document. Neither null makes all tokens independent or controls all contextual factors. The exploratory control question is whether mixture-qualified latents change more than weak-mixture latents of similar discovery support and document frequency.

## Experimental setup

The model was loaded at revision `{prov["model_revision"]}`. The SAE repository was `{prov["sae_repo"]}` at revision `{prov["sae_revision"]}`, path `resid_post/layer_13_width_16k_l0_medium`. Its native configuration identifies `model.layers.13.output`, width 16,384, JumpReLU architecture and target L0=60. We used native Hugging Face residual outputs in bfloat16 and SAE encoding in float32; all native SAE parameter tensors were checked against the pinned checkpoint. Collection ran on {prov["gpu"]}, PyTorch {prov["torch"]}, CUDA {prov["cuda"]}. No top-k cap was applied.

Corpus: `{prov["dataset"]}`, revision `{prov["dataset_revision"]}`. A seeded permutation selected unique raw documents of at least 300 characters; raw strings were preserved, without heuristic text cleanup. Collection seed: {cfg.collection.random_seed}. Discovery/evaluation seed: {rcfg.seed}; split: {summary["discovery_documents"]}/{summary["evaluation_documents"]} documents, with no document crossing the boundary. The repository target-token filter excludes initial position 0, empty/known mojibake strings and whitespace, special, punctuation, quote, symbol, control and mojibake target classes. Surrounding context remains available. All-positive collection passed finite-value and dimension checks. The recorded single-document reconstruction check is a compatibility diagnostic, not a corpus-wide reconstruction-quality estimate.

## Methods

### Discovery, mixtures and assignment

Latents required at least {rcfg.min_discovery_support} eligible discovery activations in {rcfg.min_discovery_documents} documents. We uniformly sampled up to {rcfg.screen_features} of these latents using a fixed seed ({summary["eligible_discovery_features"]} were eligible). One- and two-component Gaussian mixtures used log(1+A), full scalar covariance, scikit-learn default covariance regularization 1e-6, tolerance 1e-3, max_iter=100, k-means initialization and {rcfg.gmm_n_init} starts. BIC is -2 log-likelihood + k log(n); delta BIC = BIC(1)-BIC(2). Qualification requires convergence, delta BIC >= {rcfg.delta_bic_threshold}, both component weights >= {rcfg.min_component_weight}, and separation (mu_high-mu_low)/sqrt((variance_low+variance_high)/2) >= {rcfg.min_separation}. Components are ordered by their log1p means. Highest discovery delta BIC, then latent ID, determines the cap of {rcfg.max_candidates}; there is no composite scientific score.

Frozen discovery parameters assign evaluation observations to low or high only when that component's posterior is >= {rcfg.posterior_threshold}. Others remain ambiguous and are excluded from relational tests but retained as evidence. Each regime requires {rcfg.min_regime_support} observations in {rcfg.min_regime_documents} documents. Failures remain in the selected testing family with p=1. GMM diagnostics and downstream effect estimates use different documents; held-out observations never refit the GMM or rank discovery candidates.

### Neighborhoods and effect sizes

The partner universe is every SAE latent except the focal latent with at least {rcfg.min_partner_support} occurrences pooled across the two confident evaluation regimes. This filter is invariant to all subsequent label permutations. It is constructed directly from complete sparse activations, with no global graph cap. For partner j and regime r, c_rj is the joint count and n_r the regime size, so p_rj=c_rj/n_r. JS uses q_rj=c_rj/sum_j(c_rj), m=(q_low+q_high)/2 and JS=0.5 sum_j[q_low,j log2(q_low,j/m_j)+q_high,j log2(q_high,j/m_j)]. Its range is 0 to 1 bit; an empty distribution is undefined, never silently zero.

Other metrics are cosine between p vectors; sum_j |p_high,j-p_low,j|; signed sum_j(p_high,j-p_low,j); Jaccard of the top-{rcfg.neighbor_k} strictly nonzero neighbor sets; and Spearman correlation of counts restricted to shared top neighbors (undefined for insufficient or constant ranks). Ties are resolved by latent ID. The exported regime-event Jaccard is c_rj/(n_r+n_j-c_rj) on the complete eligible evaluation-token universe. PMI=ln(c_rj*N/(n_r*n_j)) is emitted only if the regime pair has adequate support. These event metrics are descriptive. Raw-count zeros within the saved partner universe are observed zeros; absent partners were filtered and must not be imputed into that universe as zeros.

### Nulls, uncertainty and multiple comparisons

Each feature uses {rcfg.permutations} independently seeded label shuffles, preserving feature identity and exact regime counts within strata. Primary strata are document x floor(position/{rcfg.position_bin}) x floor(number of positive latents/{rcfg.support_bin}); secondary strata replace document with exact token ID. Single-regime strata remain fixed. Movable fractions quantify how much the null can randomize. Empirical p=(1+number of null JS >= observed JS)/(B+1), with resolution {1 / (rcfg.permutations + 1):.5f}. This plus-one estimator follows Phipson and Smyth [2]. Null means, 95th percentiles, all replicates, and observed-minus-null effects are saved.

BH q-values are reported for the selected candidate family, with BY q-values as a conservative arbitrary-dependence sensitivity [3]. Secondary-null adjustments form a separate, explicitly secondary family; selecting the better of the two does not constitute a corrected primary test. Approximate pointwise 95% confidence intervals are observed JS +/-1.96 document-bootstrap standard errors, clipped to [0,1], from {rcfg.bootstraps} document resamples of evaluation observations. Resampling preserves within-document token clusters and fixes the GMM and partner set. Raw bootstrap percentile ranges and bootstrap bias are separately exported as diagnostics; divergence has positive finite-sample bias. Normal-approximation intervals are descriptive, may have poor coverage near boundaries, and do not account for corpus selection, GMM fitting, candidate selection or multiple-feature interval coverage.

### Matched controls and contextual review

Controls must have a converged discovery fit and delta BIC <10. Greedy nearest matching, without replacement, uses Euclidean distance in log discovery activation count and log distinct-document count, with absolute per-coordinate caliper {rcfg.match_log_caliper}. Frequency is redundant with support on a common eligible-token denominator. Variance is recorded for balance diagnostics, not matched. Each control's low/high tails are fixed by discovery activation quantiles matching the candidate's discovery-confident fractions; these are magnitude-based controls, not claimed mixture regimes. Evaluation control tails must contain enough observations to downsample without replacement to exactly the candidate's two regime sizes. No extra control is substituted after evaluation failure; exclusions are reported.

Context examples are selected algorithmically by cosine to their own regime's partner centroid, allowing at most one example per document per selection category. Counterexamples maximize opposite-minus-own centroid similarity and are structural challenges, not validated semantic contradictions. Ambiguous examples have posterior nearest 0.5. Raw text, contiguous tokenizer-decoded spans, target ID/string, position, activation, both posteriors and up to 20 coactivating features accompany each example. Contexts include a right-hand window for reading; future tokens are not causal inputs to the target activation. A randomized blind annotation sheet and separate key support subsequent human review. No annotation has yet been completed and no automatic semantic label is treated as ground truth.

## Results

{table(valid.sort_values("null_excess", ascending=False) if not valid.empty else valid, ["feature_id", "n_low", "n_high", "js_bits", "null_excess", "document_q_bh", "document_q_by", "neighbor_jaccard"])}

Figure 1 shows mixture evidence, support, the conditional null, matched controls, adjusted evidence and neighbor overlap. These plots describe the selected candidate family.

![Aggregate results](figures/aggregate.png)

## Aggregate analysis

{conclusion} Median JS among supported candidates was {median:.4f} bits and median null excess was {excess:.4f} bits. Under the secondary token-identity-stratified null, {n_token} selected candidates passed BY q <= 0.05. These counts cannot establish how prevalent the phenomenon is across the full SAE dictionary: candidates were selected for strong discovery mixture evidence and evaluation eligibility varies by feature. The top-24 cap and initial supported-feature sampling further delimit the population. There is no justified single population-level semantic conclusion.

## Controls and nulls

{control_text} Discovery matching succeeded for {int((matches.match_status == "matched").sum())} of {len(matches)} selected candidates; remaining matches were explicitly unavailable within the caliper. A failure to obtain controls weakens the specificity claim, even when within-feature permutation evidence is strong.

{table(pairs, ["candidate_id", "control_id", "candidate_js", "control_js", "paired_js_difference"])}

![Candidate effects and null baselines](figures/candidate_effects.png)

## Candidate-level analysis

The following cases are ordered lexicographically by primary BY q, then null-excess JS, then feature ID. This is a transparent review order, not a calibrated scientific score. Text examples are preselected by the algorithm described above and have not been semantically adjudicated.
"""
    ordered = (
        valid.sort_values(
            ["document_q_by", "null_excess", "feature_id"], ascending=[True, False, True]
        )
        if not valid.empty
        else valid
    )
    for row in ordered.head(3).itertuples():
        text += f"\n### Latent {row.feature_id}\n\nObserved JS={row.js_bits:.4f} bits, null excess={row.null_excess:.4f}; primary p={row.document_p:.4g}, BY q={row.document_q_by:.4g}. Regime supports: {int(row.n_low)}/{int(row.n_high)} observations across {int(row.documents_low)}/{int(row.documents_high)} documents. Assignment rate={row.assignment_rate:.1%}; primary movable fraction={row.document_movable_fraction:.1%}.\n\n"
        text += f"![Activation assignments](figures/feature_{row.feature_id}_activation.png)\n\n"
        for regime in ["low", "high", "ambiguous"]:
            group = context[
                (context.feature_id == row.feature_id) & (context.regime_name == regime)
            ]
            for ex in group.head(2).itertuples():
                text += (
                    f"**{regime.capitalize()} / {ex.selection_method}**, document {ex.text_id}, token {ex.token_pos}, "
                    f"activation {ex.activation:.3f}, P(low)={ex.posterior_low:.3f}, P(high)={ex.posterior_high:.3f}; "
                    f"target `{ex.center_token}`.\n\n> "
                    + ex.display_context.replace("\n", " ").replace(">", "")
                    + "\n\n"
                )
        counter = context[
            (context.feature_id == row.feature_id) & (context.selection_method == "counterexample")
        ].head(1)
        if not counter.empty:
            ex = counter.iloc[0]
            text += f"**Structural counterexample ({ex.regime_name})**, document {ex.text_id}, token {ex.token_pos}:\n\n> {ex.display_context.replace(chr(10), ' ')}\n\n"
    text += f"""## Dependence sensitivity

{robustness_text}

## Exploratory control amendment

The strict delta-BIC<10 control arm is preserved in regime_matched_controls.parquet. It produced no support-matched controls in the initial run. After this failure, we added an explicitly exploratory arm: up to 2,048 additional discovery-supported latents were sampled using seed+1, fitted only on discovery documents, and pooled with the original screen. Controls in this arm require convergence and standardized component separation below 2. The original support/document calipers and exact evaluation regime-size requirement remain unchanged. Fits, matches and null replicates are exported separately with the weak_control prefix. Weak separation does not imply a one-Gaussian distribution, so this amendment tests specificity relative to weakly separated mixtures only. It must not be described as a successful strict-BIC control analysis. This is a post-hoc exploratory amendment and needs prospective replication.

## Interpretation

Activation-conditioned relational change is compatible with several explanations: continuous contextual specificity, lexical identity, changing activation support, topic variation, or multiple contextual uses. A mixture fit does not distinguish these explanations. Even a candidate that exceeds its random-split null can behave like ordinary weak-mixture controls. Semantically distinct regimes require blinded annotations with reliability checks and replication, and causal claims require interventions. The strongest current conclusion is restricted to empirical conditional coactivation structure under the stated exchangeability assumptions.

## Threats to validity and skeptical review

1. **Magnitude alone.** Regime membership is itself a function of activation magnitude. Null rejection detects association, not a discontinuity or a special two-state mechanism. Quantile-tail controls are essential; a smooth activation-to-context relationship remains a viable explanation.
2. **Top-k censoring.** The primary collection stores all positive activations, removing rank competition from this estimand. The historical 100-document top-32 artifacts lack current lineage and were not reused as scientific evidence. Generalization to rank-censored collections is untested.
3. **Corpus dependence.** Pile-10k is a small, preselected corpus sample. Deduplication here removes exact repeated raw texts only; near duplicates and shared sources can still correlate documents. First-window truncation favors document openings.
4. **Tokenization and support.** Target filtering changes the population. Subwords, lexical identity and position remain plausible drivers. The secondary lexical null and primary document null control different confounders, not all simultaneously. Positive-support bins are coarse, so within-bin magnitude/support confounding can remain.
5. **Exchangeability.** Shuffling within document/position/support does not guarantee exchangeability of naturally ordered tokens. Adjacent tokens can remain dependent. The cluster bootstrap improves uncertainty treatment but cannot validate the permutation null. Very low movable fractions indicate conditional tests with little power. Therefore p/q values are conditional-model evidence, not assumption-free proof.
6. **Missing edges.** The raw membership matrix avoids confusing graph truncation with zero. The pooled support threshold still changes the analyzed partner population; instability below the threshold is not estimated. Different candidates can have different partner sets.
7. **GMM misspecification.** A skewed or heavy-tailed unimodal distribution may favor two Gaussians. Separation and weight screens do not certify density bimodality. Parameters are on log1p scale, not log scale, and are conditional on positivity. GMM selection uncertainty is not included in the reported CIs.
8. **Multiple comparisons and selection.** Held-out evaluation separates fitting from testing; BH and BY include selected support failures. Feature tests share tokens and partners. BY handles arbitrary dependence only when each p-value is valid under its null; it cannot repair exchangeability violations. Candidate rankings remain exploratory and interval coverage is pointwise.
9. **Controls.** Matching is on support and document frequency, not every activation-distribution property. Control failures and discovery calipers limit comparability. A raw divergence difference is not a causal treatment effect of mixture-likeness.
10. **SAE/model specificity.** One pretrained model, one layer, width and target L0 were tested. No SAE-seed, layer, model or corpus replication is claimed. Decoder geometry and historical PCA/UMAP are descriptive and were not used as semantic validation.
11. **Context interpretation.** Representative and contrary contexts were chosen computationally. Human readers may nevertheless invent post-hoc themes. Blinded independent annotation, negative examples and a held-out semantic validation set remain necessary. Coactivation is descriptive and is not causal interaction.

## Scientific conclusion

{conclusion} {control_text} These results support continued investigation of activation-conditioned relational structure. They do not establish that SAE latents lack a single semantic identity, or that distinct semantic concepts occupy the fitted components.

## Next experiments

Use the Colab larger configuration for more documents and a wider discovery screen, preserving the discovery/evaluation boundary. Replicate on a distinct corpus and later document windows. Compare against flexible smooth activation-response models to distinguish continuous specificity from discrete regimes. Repeat with target L0 variants, widths, layers and model sizes. Obtain two blinded annotators per context, define categories without exposing activation regime, measure inter-rater agreement, then test regime/category association on a new semantic holdout. Finally, use activation patching or carefully designed interventions to test causal consequences. Larger compute alone does not resolve these identification problems.

## Reproducibility and artifacts

The machine-readable experiment manifest records Git SHA and dirty state, code hashes, exact model/SAE/corpus revisions, checkpoint checksum, runtime versions, all filtering policies, seeds, GMM settings, confidence/support thresholds, null counts and control parameters. Per-25-document collection chunks and per-feature analysis checkpoints preserve completed work. Legacy lineage is untouched; this fresh run has its own collection/analysis lineage. Source texts and raw positive sparse activations are retained for independent analysis. Full posterior assignments, fits, regime edges, null replicates, controls, context examples and CSV tables accompany this report. The final bundle includes the source used to generate the experiment so uncommitted code is not lost.

## References

[1] Google DeepMind. Gemma Scope 2 technical report and model release. [Official model card](https://huggingface.co/google/gemma-scope-2-1b-pt) and [technical report](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/gemma-scope-2-helping-the-ai-safety-community-deepen-understanding-of-complex-language-model-behavior/Gemma_Scope_2_Technical_Paper.pdf).

[2] Phipson, B. and Smyth, G. K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. [Author manuscript](https://gksmyth.github.io/pubs/PermPValuesPreprint.pdf).

[3] Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. Annals of Statistics 29(4), 1165-1188. [Author manuscript](https://www.math.tau.ac.il/~ybenja/depApr27.pdf).
"""
    if valid.empty:
        text = text.replace(
            "![Aggregate results](figures/aggregate.png)",
            "No supported candidate figure is available.",
        )
        text = text.replace(
            "![Candidate effects and null baselines](figures/candidate_effects.png)", ""
        )
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"
    (dest / "scientific_report.md").write_text(text, encoding="utf-8")
    return dest / "scientific_report.md"
