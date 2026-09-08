"""Publication figures from measured foundation outputs; illustrative panels labelled."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

COLORS = {17: "#235c7c", 22: "#b46a32"}
SOURCES = ["fineweb-edu-sample", "wikimedia-en"]
LABELS = ["FineWeb-Edu", "Wikipedia"]


def figures(destination):
    out = Path(destination) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "savefig.facecolor": "white",
        }
    )
    arms = {}
    for layer in (17, 22):
        root = Path(f"data/processed/gemma4b_foundation_v1_l{layer}")
        arms[layer] = {"root": root, "g": root / "geometry"}
        for key, path in {
            "fits": "regime_feature_summary.parquet",
            "stats": "feature_stats.parquet",
            "source": "source_confirmation/source_results.parquet",
            "conj": "source_confirmation/conjunction.parquet",
            "orth": "geometry/orthogonal_source_results.parquet",
            "orthq": "geometry/orthogonal_conjunction.parquet",
            "metrics": "geometry/decoder_metrics.parquet",
            "reconstruction": "geometry/pc_reconstruction.parquet",
        }.items():
            arms[layer][key] = pd.read_parquet(root / path)
        arms[layer]["pca"] = np.load(root / "geometry/residual_pca.npz")
        arms[layer]["coverage"] = json.loads((root / "geometry/coverage_complete.json").read_text())

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=240, bbox_inches="tight")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 2.7))
    rng = np.random.default_rng(8)
    for ax, context, title in zip(
        axes, [False, True], ["Amplitude alone", "Amplitude with contextual displacement"]
    ):
        for x, y, c, label in [
            (1, 0, "#235c7c", "Weak"),
            (3, 1.2 if context else 0, "#b46a32", "Strong"),
        ]:
            cloud = rng.normal(size=(35, 2)) * 0.17 + [x, y]
            ax.scatter(*cloud.T, s=9, c=c, alpha=0.6, label=label)
        ax.annotate("", xy=(4, 0), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "#555"})
        ax.set(
            xlim=(0, 4.1),
            ylim=(-0.6, 1.8),
            title=title,
            xlabel="Feature decoder direction",
            ylabel="Orthogonal context",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Hypothesis illustration - these points are not experimental data", fontsize=10, y=1.04
    )
    save(fig, "01_hypothesis")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3))
    x = np.arange(2)
    axes[0].bar(x, [2965, 3054], color="#235c7c", label="Discovery")
    axes[0].bar(x, [3035, 2946], bottom=[2965, 3054], color="#b8ced9", label="Evaluation")
    axes[0].set(xticks=x, xticklabels=LABELS, ylabel="Documents", title="Frozen corpus allocation")
    axes[0].legend(frameon=False, fontsize=8)
    for layer in (17, 22):
        a = arms[layer]
        eligible = int(a["stats"].analysis_token_denominator.iloc[0])
        stored = int(a["stats"].stored_token_denominator.iloc[0])
        axes[1].bar(str(layer), eligible / 1e6, color=COLORS[layer])
        axes[1].bar(str(layer), (stored - eligible) / 1e6, bottom=eligible / 1e6, color="#ddd")
    axes[1].set(
        ylabel="Million token positions",
        xlabel="Layer",
        title="Analysis population (color); excluded (gray)",
    )
    save(fig, "02_design")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.1))
    for layer in (17, 22):
        s = arms[layer]["stats"]
        freq = (
            s.set_index("feature_id")
            .analysis_token_frequency.reindex(range(65536), fill_value=0)
            .to_numpy()
        )
        axes[0].plot(
            np.arange(1, len(freq) + 1),
            np.sort(freq)[::-1],
            label=f"Layer {layer}",
            color=COLORS[layer],
        )
        axes[1].hist(
            np.log10(s.loc[s.analysis_activation_count > 0, "analysis_activation_count"]),
            bins=50,
            histtype="step",
            color=COLORS[layer],
            label=f"Layer {layer}",
        )
    axes[0].set(
        yscale="log",
        xlabel="Feature rank",
        ylabel="Positive activation frequency",
        title="Highly unequal feature usage",
    )
    axes[1].set(
        xlabel="log10 eligible token support",
        ylabel="Features",
        title="Observed support distribution",
    )
    axes[0].legend(frameon=False)
    save(fig, "03_frequency")
    a = arms[17]
    examples = pd.read_parquet(a["g"] / "discovery_activation_examples.parquet")
    cases = (
        a["fits"][a["fits"].selected_candidate]
        .sort_values(["delta_bic", "feature_id"], ascending=[False, True])
        .head(6)
    )
    fig, axes = plt.subplots(2, 3, figsize=(8.3, 4.6))
    for ax, (_, f) in zip(axes.flat, cases.iterrows()):
        values = np.log1p(examples.loc[examples.feature_id == f.feature_id, "activation"])
        ax.hist(values, bins=40, density=True, color="#cadbe3", edgecolor="none")
        xx = np.linspace(values.min(), values.max(), 300)
        for r, color in [("low", "#235c7c"), ("high", "#b46a32")]:
            ax.plot(
                xx,
                f[f"component_weight_{r}"]
                * norm.pdf(xx, f[f"log_mean_{r}"], np.sqrt(f[f"log_variance_{r}"])),
                color=color,
            )
        ax.set(
            title=f"Feature {int(f.feature_id)}",
            xlabel="log(1 + positive activation)",
            ylabel="Density",
        )
    save(fig, "04_distributions")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3))
    for ax, layer in zip(axes, (17, 22)):
        a = arms[layer]
        v = a["source"]
        v = (
            v[v.status == "ok"]
            .pivot(index="feature_id", columns="source", values="null_excess")
            .reindex(columns=SOURCES)
            .dropna()
        )
        qs = a["conj"].set_index("feature_id").q_by.reindex(v.index)
        ax.scatter(
            v.iloc[:, 0],
            v.iloc[:, 1],
            c=np.where(qs <= 0.05, COLORS[layer], "#bbb"),
            s=36,
            edgecolors="white",
            linewidths=0.4,
        )
        ax.axhline(0, color="#888", lw=0.6)
        ax.axvline(0, color="#888", lw=0.6)
        ax.set(
            title=f"Layer {layer}: {int((qs <= 0.05).sum())}/{len(a['conj'])} pass BY",
            xlabel="FineWeb-Edu: JS excess (bits)",
            ylabel="Wikipedia: JS excess (bits)",
        )
    save(fig, "05_source_replication")
    fig, ax = plt.subplots(figsize=(8.3, 6.5))
    a = arms[17]
    order = a["conj"].sort_values(["q_by", "feature_id"]).feature_id.to_numpy()
    for offset, source, label, c in zip([-0.15, 0.15], SOURCES, LABELS, ["#235c7c", "#b46a32"]):
        table = a["source"].set_index(["feature_id", "source"])
        for n, fid in enumerate(order):
            row = table.loc[(fid, source)]
            if row.status == "ok":
                ax.plot([row.js_ci_low, row.js_ci_high], [n + offset, n + offset], c=c, lw=1)
                ax.scatter(row.js_bits, n + offset, s=18, c=c, label=label if n == 0 else None)
                ax.scatter(row.token_identity_null_mean, n + offset, s=15, marker="|", c="black")
            else:
                ax.text(0.002, n + offset, "insufficient support", fontsize=6, color=c, va="center")
    ax.set(
        yticks=np.arange(len(order)),
        yticklabels=[str(x) for x in order],
        xlabel="JS divergence (bits); black ticks: conditional-null mean",
        ylabel="Feature ID",
        title="Layer 17: effect sizes and pointwise bootstrap intervals",
    )
    ax.invert_yaxis()
    ax.legend(frameon=False)
    save(fig, "06_effect_intervals")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3))
    for ax, layer in zip(axes, (17, 22)):
        a = arms[layer]
        o = a["orth"]
        o = o[o.status == "ok"].copy()
        o["fraction"] = o.orthogonal_squared / o.total_squared
        q = a["orthq"].set_index("feature_id").q_by
        for source, marker in zip(SOURCES, ["o", "^"]):
            v = o[o.source == source]
            ax.scatter(
                v.fraction,
                (v.orthogonal_squared - v.null_mean) / v.null_mean,
                marker=marker,
                c=np.where(v.feature_id.map(q) <= 0.05, COLORS[layer], "#bbb"),
                s=28,
                label=LABELS[SOURCES.index(source)],
            )
        ax.axhline(0, color="#888", lw=0.6)
        ax.set(
            title=f"Layer {layer}: orthogonal context",
            xlabel="Fraction of squared shift orthogonal to decoder",
            ylabel="Relative squared displacement above null mean",
        )
        ax.set_yscale("symlog", linthresh=0.1)
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "07_orthogonal")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for layer in (17, 22):
        eig = arms[layer]["pca"]["eigenvalues"]
        xx = np.arange(1, len(eig) + 1)
        axes[0].loglog(xx, eig / eig.sum(), color=COLORS[layer], label=f"Layer {layer}")
        axes[1].plot(xx, np.cumsum(eig) / eig.sum(), color=COLORS[layer])
    axes[0].set(
        xlabel="Principal component",
        ylabel="Fraction of residual variance",
        title="Full discovery residual spectrum",
    )
    axes[1].set(
        xlabel="Number of principal components",
        ylabel="Cumulative variance fraction",
        title="How much variance does a projection retain?",
    )
    axes[0].legend(frameon=False)
    save(fig, "08_pca_spectrum")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    random_vectors = np.random.default_rng(20261031).normal(size=(4096, 2560))
    random_mass = random_vectors**2
    random_mass /= random_mass.sum(axis=1, keepdims=True)
    random_pr = 1 / (random_mass**2).sum(axis=1)
    pd.DataFrame({"isotropic_participation_ratio": random_pr}).to_csv(
        out.parent / "isotropic_reference.csv", index=False
    )
    for ax, layer in zip(axes, (17, 22)):
        m = arms[layer]["metrics"]
        for col, label, c in [
            ("raw_pr", "Raw coordinates", "#999"),
            ("pca_pr", "PCA basis", "#235c7c"),
            ("white_0.001_pr", "Whitened PCA", "#b46a32"),
        ]:
            values = np.sort(m[col])
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=label, color=c)
        ax.plot(
            np.sort(random_pr),
            np.arange(1, len(random_pr) + 1) / len(random_pr),
            "k--",
            lw=0.8,
            label="Isotropic reference (simulation)",
        )
        ax.set(
            xlabel="Effective number of components (participation ratio)",
            ylabel="Fraction of decoder directions",
            title=f"Layer {layer}: basis-dependent participation",
        )
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "09_participation")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for ax, layer in zip(axes, (17, 22)):
        m = arms[layer]["metrics"]
        ax.hist(m.pc_head_64, bins=60, color=COLORS[layer], alpha=0.8)
        ax.axvline(
            64 / len(arms[layer]["pca"]["eigenvalues"]),
            c="#222",
            ls="--",
            label="Uniform directional mass",
        )
        ax.set(
            xlabel="Squared decoder mass in the first 64 residual PCs",
            ylabel="Decoder directions",
            title=f"Layer {layer}: directional alignment",
        )
    axes[0].legend(frameon=False, fontsize=7)
    save(fig, "10_alignment")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.3))
    for ax, layer in zip(axes, (17, 22)):
        pairs = pd.read_parquet(arms[layer]["g"] / "screened_pair_geometry.parquet").sample(
            n=100000, random_state=123
        )
        h = ax.hexbin(
            pairs.decoder_cosine,
            pairs.token_jaccard,
            gridsize=45,
            bins="log",
            mincnt=1,
            cmap="Blues",
        )
        ax.set(
            xlabel="Decoder cosine",
            ylabel="Token Jaccard overlap",
            title=f"Layer {layer}: geometry and co-occurrence",
        )
        fig.colorbar(h, ax=ax, label="Pair count (log color scale)")
    save(fig, "11_pair_geometry")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for ax, layer in zip(axes, (17, 22)):
        r = arms[layer]["reconstruction"]
        ax.plot(r.pc, r.r_squared, color=COLORS[layer], lw=0.7)
        ax.axhline(0, c="#888", lw=0.6)
        ax.set(
            xlabel="Discovery principal component",
            ylabel="Held-out reconstruction R-squared",
            yscale="symlog",
            title=f"Layer {layer}: reconstruction across the spectrum",
        )
    save(fig, "12_reconstruction")
    a = arms[17]
    case = int(a["conj"].sort_values(["q_by", "feature_id"]).feature_id.iloc[0])
    target = pd.read_parquet(a["g"] / "targets.parquet")
    projection = pd.read_parquet(a["g"] / "native_projections.parquet")
    frame = target[(target.purpose == "primary") & (target.feature_id == case)].merge(
        projection[["text_id", "token_pos", "pc1", "pc2"]], on=["text_id", "token_pos"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    ratio = a["pca"]["eigenvalues"][:2].sum() / a["pca"]["eigenvalues"].sum()
    for ax, source, label in zip(axes, SOURCES, LABELS):
        for level, c, name in [(0, "#235c7c", "Weak"), (1, "#b46a32", "Strong")]:
            f = frame[(frame.source == source) & (frame.level == level)]
            ax.scatter(f.pc1, f.pc2, s=7, c=c, alpha=0.35, label=name)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"{label}: feature {case}")
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(
        f"Discovery-fitted projection retains {ratio:.1%} of reference variance",
        fontsize=10,
        y=1.03,
    )
    save(fig, "13_native_projection")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for ax, layer in zip(axes, (17, 22)):
        q = arms[layer]["orthq"]
        colors = np.where(q.q_by <= 0.05, COLORS[layer], "#bbb")
        ax.scatter(np.arange(len(q)), q.cross_source_cosine, c=colors, s=24)
        ax.axhline(0, c="#888", lw=0.6)
        ax.set(
            ylim=(-1.05, 1.05),
            xlabel="Selected feature (ID order)",
            ylabel="Cosine between orthogonal mean shifts",
            title=f"Layer {layer}: agreement between sources",
        )
    save(fig, "14_shift_agreement")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for ax, layer in zip(axes, (17, 22)):
        q = pd.read_parquet(arms[layer]["g"] / "quantile_geometry.parquet")
        for fid, f in q.groupby("feature_id"):
            v = f.groupby("level").pc1_mean.mean().reindex(range(4))
            if v.notna().all():
                ax.plot(
                    range(1, 5),
                    (v - v.mean()) / np.sqrt(arms[layer]["pca"]["eigenvalues"][0]),
                    color=COLORS[layer],
                    alpha=0.25,
                    lw=0.7,
                )
        ax.set(
            xticks=range(1, 5),
            xlabel="Frozen positive-activation quartile",
            ylabel="Centered mean PC1 / discovery SD",
            title=f"Layer {layer}: 64 randomly sampled features",
        )
    save(fig, "15_quantile_context")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for ax, layer in zip(axes, (17, 22)):
        original = arms[layer]["orthq"].set_index("feature_id")
        control = pd.read_parquet(
            arms[layer]["g"] / "encoder_control_conjunction.parquet"
        ).set_index("feature_id")
        merged = original[["q_by"]].join(control[["q_by"]], rsuffix="_control")
        ax.scatter(
            -np.log10(merged.q_by), -np.log10(merged.q_by_control), color=COLORS[layer], s=25
        )
        ax.axhline(-np.log10(0.05), c="#888", ls="--", lw=0.7)
        ax.axvline(-np.log10(0.05), c="#888", ls="--", lw=0.7)
        ax.set(
            xlabel="Original orthogonal test: -log10(q)",
            ylabel="Encoder-covariance diagnostic: -log10(q)",
            title=f"Layer {layer}: exploratory sensitivity",
        )
    save(fig, "16_encoder_control")
    a = arms[17]
    joint = sorted(
        set(a["conj"].loc[a["conj"].q_by <= 0.05, "feature_id"])
        & set(a["orthq"].loc[a["orthq"].q_by <= 0.05, "feature_id"])
    )
    fig, axes = plt.subplots(1, len(joint), figsize=(8.3, 4.5), squeeze=False)
    for ax, fid in zip(axes.flat, joint):
        cols = []
        for source in SOURCES:
            edges = pd.read_parquet(a["root"] / f"source_confirmation/{fid}_{source}.edges.parquet")
            wide = edges.pivot(
                index="partner_id", columns="regime", values="conditional_probability"
            )
            cols.append((wide.high - wide.low).rename(source))
        changes = pd.concat(cols, axis=1).fillna(0)
        # Only partners present in both declared supported universes are compared.
        common = set(cols[0].index) & set(cols[1].index)
        changes = changes.loc[sorted(common)]
        chosen = changes.abs().max(axis=1).nlargest(20).index
        changes = changes.loc[chosen]
        limit = max(float(changes.abs().to_numpy().max()), 0.01)
        im = ax.imshow(changes.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        ax.set(
            xticks=[0, 1],
            xticklabels=["Web", "Wiki"],
            yticks=np.arange(len(chosen)),
            yticklabels=[str(x) for x in chosen],
            title=f"Feature {fid}",
            ylabel="Partner feature ID",
        )
        ax.tick_params(axis="y", labelsize=6)
        fig.colorbar(im, ax=ax, label="P(partner | strong) - P(partner | weak)", shrink=0.75)
    save(fig, "17_partner_changes")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.4))
    for ax, layer in zip(axes, (17, 22)):
        root = arms[layer]["root"]
        matches = pd.read_parquet(root / "regime_weak_matched_controls.parquet")
        controls = pd.read_parquet(root / "regime_weak_control_comparison.parquet")
        candidates = pd.read_parquet(root / "regime_neighborhood_comparison.parquet")
        candidates = candidates[candidates.role == "candidate"].set_index("feature_id")
        paired = []
        for row in controls[controls.status == "ok"].itertuples():
            candidate = candidates.loc[row.candidate_id]
            assert candidate.status == "ok"
            assert candidate.n_low == row.n_low and candidate.n_high == row.n_high
            difference = candidate.js_bits - row.js_bits
            # Cauchy-Schwarz bounds the standard error of a difference by the
            # sum of marginal standard errors, without independence assumptions.
            radius = 1.96 * (candidate.js_bootstrap_se + row.js_bootstrap_se)
            paired.append(
                {
                    "candidate_id": row.candidate_id,
                    "control_id": row.feature_id,
                    "candidate_js": candidate.js_bits,
                    "control_js": row.js_bits,
                    "difference": difference,
                    "approx_low": difference - radius,
                    "approx_high": difference + radius,
                    "candidate_document_excess": candidate.null_excess,
                    "control_document_excess": row.null_excess,
                    "control_document_p": row.document_p,
                    "control_token_identity_p": row.token_identity_p,
                }
            )
        paired = pd.DataFrame(paired)
        arms[layer]["weak_pairs"] = paired
        arms[layer]["weak_counts"] = [
            layer,
            len(matches),
            int(matches.control_id.notna().sum()),
            len(paired),
        ]
        y = np.arange(len(paired))
        if len(paired):
            ax.errorbar(
                paired.difference,
                y,
                xerr=np.stack(
                    [paired.difference - paired.approx_low, paired.approx_high - paired.difference]
                ),
                fmt="o",
                color=COLORS[layer],
                markersize=4,
            )
            ax.set_yticks(
                y, [f"{int(r.candidate_id)} / {int(r.control_id)}" for r in paired.itertuples()]
            )
        else:
            ax.text(0.5, 0.5, "No supported pairs", ha="center", transform=ax.transAxes)
            ax.set_yticks([])
        ax.axvline(0, color="#888", lw=0.7)
        ax.set(
            title=f"Layer {layer}: {len(paired)} supported pairs",
            xlabel="Candidate minus control JS (bits)",
            ylabel="Candidate / control feature",
        )
    save(fig, "18_weak_controls")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for layer in (17, 22):
        frame = pd.read_parquet(arms[layer]["g"] / "decoder_context_consistency.parquet")
        arms[layer]["consistency"] = frame
        axes[0].scatter(
            frame.binary_native_cosine,
            frame.binary_native_orthogonal_cosine,
            color=COLORS[layer],
            alpha=0.65,
            s=15,
            label=f"Layer {layer}",
        )
        axes[1].scatter(
            frame.decoder_low_neighbor_jaccard,
            frame.decoder_high_neighbor_jaccard,
            color=COLORS[layer],
            alpha=0.65,
            s=15,
            label=f"Layer {layer}",
        )
    axes[0].set(
        xlabel="Binary partner sum vs native shift: cosine",
        ylabel="Same comparison after removing focal axis",
        title="Do partner changes track native context?",
    )
    axes[1].set(
        xlabel="Decoder / weak-context top-20 Jaccard",
        ylabel="Decoder / strong-context top-20 Jaccard",
        title="Direction neighbors and context neighbors",
    )
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    save(fig, "19_context_consistency")
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.2))
    for ax, layer in zip(axes, (17, 22)):
        p = pd.read_parquet(arms[layer]["g"] / "decoder_pca.parquet")
        record = json.loads((arms[layer]["g"] / "consistency_complete.json").read_text())
        im = ax.hexbin(p.pc1, p.pc2, gridsize=45, mincnt=1, bins="log", cmap="Blues")
        ax.set(
            xlabel="Decoder PC1",
            ylabel="Decoder PC2",
            title=f"Layer {layer}: {record['decoder_pca_variance_2d']:.1%} variance retained",
        )
        fig.colorbar(im, ax=ax, label="Dictionary directions (log count)")
    save(fig, "20_decoder_pca")
    return arms


if __name__ == "__main__":
    figures("reports/gemma4b_foundation_v1")
