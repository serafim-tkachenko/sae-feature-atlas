"""Development-only paired controls, prediction checks and scientific pilot figures."""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sae_feature_atlas.scientific.collect import sha256
from sae_feature_atlas.scientific.intervention_math import paired_bootstrap
from sae_feature_atlas.util.io import write_json


def prediction_checks(frame):
    """Calibrate only on fit documents; check documents remain old development data."""
    rows = []
    base = [
        "encoder",
        "norm",
        "token_pos",
        "support",
        "trailing_digit_count",
        "trailing_letter_count",
    ]
    for (fid, dose), data in frame.groupby(["feature_id", "dose"]):
        for train_source in ["pooled", *sorted(data.source.unique())]:
            train = data[data.split == "fit"]
            test = data[data.split == "check"]
            if train_source != "pooled":
                train = train[train.source == train_source]
                test = test[test.source != train_source]
            if len(train) < 4 or len(test) < 2:
                continue
            y_train = np.stack(train.effect)
            y_test = np.stack(test.effect)
            for name, numeric in [
                ("nuisance", base),
                ("context", base + ["context"]),
                ("pc8_context", base + ["context"] + [f"pc{i}" for i in range(8)]),
                ("scalar_gain", base + ["context"]),
            ]:
                transform = ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), numeric),
                        (
                            "categorical",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ["token_id", "source"],
                        ),
                    ]
                )
                model = make_pipeline(transform, Ridge(alpha=1.0))
                if name == "scalar_gain":
                    mean = y_train.mean(0)
                    if mean @ mean < 1e-12:
                        continue
                    gains = y_train @ mean / (mean @ mean)
                    model.fit(train, gains)
                    pred = model.predict(test)[:, None] * mean
                else:
                    model.fit(train, y_train)
                    pred = model.predict(test)
                for (_, sample), expected, predicted in zip(test.iterrows(), y_test, pred):
                    rows.append(
                        dict(
                            feature_id=fid,
                            dose=dose,
                            train_source=train_source,
                            model=name,
                            text_id=sample.text_id,
                            duplicate_group=sample.duplicate_group,
                            source=sample.source,
                            mse=float(np.mean((expected - predicted) ** 2)),
                        )
                    )
    return pd.DataFrame(rows)


def analyze(bundle, results, out):
    plan = json.loads((bundle / "plan.json").read_text())
    vectors = np.load(bundle / "vectors.npz")
    complete = json.loads((results / "complete.json").read_text())
    if json.loads((results / "provenance.json").read_text())["plan_sha256"] != sha256(
        bundle / "plan.json"
    ):
        raise ValueError("Results and preparation differ")
    for name, digest in complete["files"].items():
        if sha256(results / name) != digest:
            raise ValueError("Corrupt results")
    out.mkdir(parents=True, exist_ok=True)
    rows, predictions, trajectories, exclusions = [], [], [], []
    for name in complete["files"]:
        result = json.loads((results / name).read_text())
        p = result["prompt"]
        common = {k: p[k] for k in ["feature_id", "text_id", "duplicate_group", "source", "split"]}
        common["baseline_active"] = bool(
            result["baseline"]["encoder"] > float(vectors[f"{p['feature_id']}_threshold"])
        )
        doses = set()
        for record in result["records"]:
            if record["status"] != "ok":
                exclusions.append({**common, **record})
                continue
            cells = np.asarray(record["cells"])
            e0, e1 = cells[1] - cells[0], cells[3] - cells[2]
            residual = e1 - e0 * (e0 @ e1) / (e0 @ e0) if e0 @ e0 > 1e-8 else None
            row = {
                **common,
                **{k: record[k] for k in ["direction", "fraction", "sign", "dose"]},
                "interaction_norm": float(np.linalg.norm(record["interaction"])),
                "normalization_only_norm": float(
                    np.linalg.norm(record["normalization_only_interaction"])
                ),
                "pre_norm_interaction_norm": float(np.linalg.norm(record["pre_norm_interaction"])),
                "beyond_gain_norm": float(np.linalg.norm(residual))
                if residual is not None
                else np.nan,
                "max_constraint_error": max(record["audit"].values()),
            }
            losses = record.get("next_token_nll", [None] * 4)
            row["combined_nll_change"] = losses[3] - losses[0] if losses[0] is not None else np.nan
            row["context_nll_change"] = losses[2] - losses[0] if losses[0] is not None else np.nan
            rows.append(row)
            for layer, value in record["layer_interaction_norm"].items():
                trajectories.append({**row, "layer": int(layer), "value": value})
            if record["dose"] not in doses:
                doses.add(record["dose"])
                word = re.search(r"[A-Za-z]+$", p.get("excerpt", ""))
                predictions.append(
                    {
                        **common,
                        "dose": record["dose"],
                        "effect": e0,
                        **{k: result["baseline"][k] for k in ["encoder", "norm", "context"]},
                        **{f"pc{i}": v for i, v in enumerate(result["baseline"]["pc8"])},
                        "token_pos": p["token_pos"],
                        "token_id": str(p["token_id"]),
                        "support": np.log1p(p["n_positive_features"]),
                        "trailing_digit_count": p["trailing_digit_count"],
                        "trailing_letter_count": len(word.group()) if word else 0,
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No successful interventions")
    frame.to_csv(out / "interventions.csv", index=False)
    write_json(out / "exclusions.json", exclusions)
    forecasts = prediction_checks(pd.DataFrame(predictions))
    forecasts.to_csv(out / "prediction_errors.csv", index=False)
    predictive_summary = []
    if not forecasts.empty:
        paired = forecasts.pivot(
            index=["feature_id", "dose", "train_source", "duplicate_group", "text_id"],
            columns="model",
            values="mse",
        ).reset_index()
        for (fid, training), group in paired.groupby(["feature_id", "train_source"]):
            if "context" in group and "nuisance" in group:
                estimate = paired_bootstrap(group.nuisance - group.context, group.duplicate_group)
                predictive_summary.append(
                    dict(
                        feature_id=fid,
                        train_source=training,
                        endpoint="nuisance_mse_minus_context_mse",
                        **estimate,
                    )
                )
        pd.DataFrame(predictive_summary).to_csv(out / "prediction_improvement.csv", index=False)
    key = [
        "feature_id",
        "text_id",
        "duplicate_group",
        "source",
        "split",
        "fraction",
        "sign",
        "dose",
    ]
    random = (
        frame[frame.direction.str.startswith("random_")]
        .groupby(key)[["interaction_norm", "beyond_gain_norm"]]
        .mean()
        .reset_index()
    )
    learned = frame[frame.direction == "learned"].merge(
        random, on=key, suffixes=("", "_random"), validate="one_to_one"
    )
    summaries = []
    for (fid, source, split, dose, fraction), group in learned.groupby(
        ["feature_id", "source", "split", "dose", "fraction"]
    ):
        for endpoint in ["interaction_norm", "beyond_gain_norm"]:
            usable = group.dropna(subset=[endpoint, endpoint + "_random"])
            if not len(usable):
                continue
            estimate = paired_bootstrap(
                usable[endpoint] - usable[endpoint + "_random"], usable.duplicate_group
            )
            summaries.append(
                dict(
                    feature_id=fid,
                    source=source,
                    split=split,
                    dose=dose,
                    fraction=fraction,
                    endpoint=endpoint,
                    **estimate,
                )
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "paired_control_summary.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figures = []

    def save(fig, name):
        fig.tight_layout()
        for suffix in ["png", "svg"]:
            fig.savefig(out / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
        plt.close(fig)
        figures.append(name)

    ids = sorted(frame.feature_id.unique())
    fig, axes = plt.subplots(len(ids), 1, figsize=(8, 2.6 * len(ids)), squeeze=False)
    for ax, fid in zip(axes[:, 0], ids):
        groups = list(frame[frame.feature_id == fid].groupby("direction"))
        for i, (direction, group) in enumerate(groups):
            estimate = paired_bootstrap(group.interaction_norm, group.duplicate_group)
            ax.errorbar(
                estimate["mean"],
                i,
                xerr=[[estimate["mean"] - estimate["low"]], [estimate["high"] - estimate["mean"]]],
                fmt="o",
                color="#255c86" if direction == "learned" else "#7d8b92",
            )
        ax.set_yticks(range(len(groups)), [d.replace("_", " ") for d, _ in groups])
        ax.set_title(f"Feature {fid}: development direction controls")
        ax.set_xlabel("Mean interaction norm (document-bootstrap interval)")
    save(fig, "01_direction_controls")
    fig, ax = plt.subplots(figsize=(6, 5))
    for fid, group in frame[frame.direction == "learned"].groupby("feature_id"):
        ax.scatter(
            group.normalization_only_norm, group.interaction_norm, s=15, alpha=0.5, label=str(fid)
        )
    ax.set_xlabel("Normalization-only surrogate interaction norm")
    ax.set_ylabel("Actual downstream interaction norm")
    ax.set_title("Surrogate diagnostic; not a subtraction-based correction")
    ax.legend(title="Feature")
    save(fig, "02_normalization_diagnostic")
    fig, ax = plt.subplots(figsize=(7, 4))
    trajectory = pd.DataFrame(trajectories)
    for (fid, direction), group in trajectory.groupby(["feature_id", "direction"]):
        if direction not in ["learned", "random_0"]:
            continue
        curve = group.groupby("layer").value.median()
        ax.plot(curve.index, curve.values, label=f"{fid}: {direction}")
    ax.set_xlabel("Native decoder block index")
    ax.set_ylabel("Median residual interaction norm")
    ax.set_title("Where interaction develops; descriptive localization")
    ax.legend(fontsize=8)
    save(fig, "03_layer_trajectory")
    if not forecasts.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        view = (
            forecasts[forecasts.train_source == "pooled"]
            .groupby(["feature_id", "model"])
            .mse.mean()
            .unstack()
        )
        view.plot.bar(ax=ax)
        ax.set_ylabel("Prediction MSE (development check)")
        ax.set_xlabel("Feature")
        ax.legend([name.replace("_", " ") for name in view.columns], title="Predictor")
        ax.set_title("Fit-only calibration; no fresh confirmation")
        save(fig, "04_prediction_check")
    if frame.combined_nll_change.notna().any():
        fig, ax = plt.subplots(figsize=(7, 4))
        selected = frame[frame.direction == "learned"]
        for fid, group in selected.groupby("feature_id"):
            ax.scatter(
                group.combined_nll_change, group.interaction_norm, alpha=0.4, s=15, label=str(fid)
            )
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlabel("Combined edit minus baseline next-token NLL (nats)")
        ax.set_ylabel("Interaction norm in diagnostic contrasts")
        ax.set_title("Prediction-quality diagnostic; positive means worse next-token loss")
        ax.legend(title="Feature")
        save(fig, "05_prediction_quality")
    fig, ax = plt.subplots(figsize=(7, 4))
    check = learned[learned.split == "check"]
    comparisons = []
    for fid, group in check.groupby("feature_id"):
        for endpoint in ["interaction_norm", "beyond_gain_norm"]:
            usable = group.dropna(subset=[endpoint, endpoint + "_random"])
            if not len(usable):
                continue
            estimate = paired_bootstrap(
                usable[endpoint] - usable[endpoint + "_random"], usable.duplicate_group
            )
            comparisons.append(dict(feature_id=fid, endpoint=endpoint, **estimate))
    for i, row in enumerate(comparisons):
        ax.errorbar(
            row["mean"], i, xerr=[[row["mean"] - row["low"]], [row["high"] - row["mean"]]], fmt="o"
        )
    ax.set_yticks(
        range(len(comparisons)),
        [f"{r['feature_id']}: {r['endpoint'].replace('_', ' ')}" for r in comparisons],
    )
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Learned minus paired random controls")
    ax.set_title("Development-check documents: descriptive paired comparison")
    save(fig, "06_paired_control_check")
    pd.DataFrame(comparisons).to_csv(out / "pooled_check_controls.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 1.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.72,
        r"$I=Y(h+\delta+\alpha u)-Y(h+\delta)-Y(h+\alpha u)+Y(h)$",
        ha="center",
        fontsize=14,
    )
    ax.text(
        0.5,
        0.2,
        r"$w^{\top}\delta=0,\quad u^{\top}\delta=0,\quad \|h+\delta\|_2=\|h\|_2$",
        ha="center",
        fontsize=14,
    )
    fig.savefig(out / "equations.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    max_error = float(frame.max_constraint_error.max())
    lines = [
        "# Context intervention development pilot",
        "",
        f"Completed {complete['prompts']} feature-prompt cases from {frame.duplicate_group.nunique()} duplicate groups, with {len(frame)} factorial comparisons. Repeated comparisons are not independent sample units.",
        "",
        "These are development diagnostics on previously collected documents. The readouts describe numerical and word-boundary formatting, not validated semantic behaviors. There are no confirmatory discoveries or causal claims about natural text semantics in this report.",
        "",
        "## Research question and methods",
        "",
        "The model is Gemma 3 4B at native block 17, using the 65,536-feature Gemma Scope 2 dictionary. Each supported feature contributes eight fit and eight check documents per source (FineWeb-Edu and English Wikipedia). A seeded duplicate-group split prevents fit/check overlap across features. Both subsets come from the old foundation corpus.",
        "",
        "Can a direction learned from ordinary SAE activation contexts predict and selectively modify a fixed decoder intervention? The proposed contribution is a transferable causal explanation; this experiment tests its first development diagnostics.",
        "",
        "An SAE (sparse autoencoder) represents a layer activation through a sparse set of learned dictionary entries, here called pseudo-concepts. The decoder vector u gives an intervention direction; the encoder vector w measures the focal feature score. Context perturbations delta preserve both coordinates and residual norm. Alpha sets the fixed decoder dose. Y denotes a specified output logit contrast.",
        "",
        "![Factorial interaction and context constraints](equations.png)",
        "",
        "The first equation compares the decoder effect with and without a context edit. The constraints in the second line hold in real arithmetic and are audited after float32 rounding. A nonzero interaction can arise from generic downstream nonlinearity; it is insufficient evidence of a special mechanism.",
        "",
        "PCA (principal component analysis) orders orthogonal directions by activation variance. Eight foundation PCs provide a richer context baseline, while the first PC supplies a perturbation control. Ridge regression predicts intervention effects with a penalty on large coefficients. The scalar-gain model allows context to scale one training-mean effect direction, testing whether predictive value needs more than amplification.",
        "",
        f"Maximum representable-state relative constraint error: {max_error:.3g}.",
        "",
        "The model runs in float32 with TF32 disabled; the original collection was bfloat16. Context directions were frozen before intervention outputs. All four cells recompute the full identical prefix. The decoder dose is fixed within a comparison; the context edit is calculated once from its baseline.",
        "",
        "Prediction uses fixed ridge regularization and fit-only scaling/encoding. Token identity, source, encoder score, norm, position, support and trailing digit/Latin-letter counts are nuisance inputs. Models add the learned context score, eight reference PCs, or constrain prediction to scalar gain. The PC basis comes from the original foundation discovery data, not a new confirmation set. The check split is development data, and these intervals are descriptive without multiplicity correction or calibration-fit uncertainty.",
        "",
        "Control intervals resample duplicate groups and weight documents equally. Small/zero baseline effects are excluded only from the beyond-gain projection at squared norm <=1e-8; their raw interactions remain. The spectrum-matched control is approximate after enforcing constraints. The normalization-only surrogate is a diagnostic, not a complete model of transformer normalization.",
        "",
    ]
    raw_positive = [
        str(r["feature_id"])
        for r in comparisons
        if r["endpoint"] == "interaction_norm" and r["low"] > 0
    ]
    gain_positive = [
        str(r["feature_id"])
        for r in comparisons
        if r["endpoint"] == "beyond_gain_norm" and r["low"] > 0
    ]
    prediction_positive = [
        str(r["feature_id"])
        for r in predictive_summary
        if r["train_source"] == "pooled" and r["low"] > 0
    ]
    lines += [
        "## What this pilot establishes",
        "",
        "Features with a pooled development-check interval entirely above zero for learned-minus-random raw interaction: "
        + (", ".join(raw_positive) or "none")
        + ".",
        "",
        "Features passing the analogous beyond-gain comparison: "
        + (", ".join(gain_positive) or "none")
        + ". Features with a pooled prediction-improvement interval entirely above zero: "
        + (", ".join(prediction_positive) or "none")
        + ".",
        "",
        "These are descriptive interval summaries, not corrected discoveries. A small pilot cannot establish absence of an effect. Nevertheless, raw modulation alone does not meet the proposed novelty criterion: prediction, selective function beyond gain, and a validated downstream mechanism must be connected before making that claim.",
        "",
    ]
    if not gain_positive or not prediction_positive:
        lines += [
            "The current evidence does not justify presenting the strong hypothesis as a positive result or scaling this exact design directly into confirmation. The next scientific task is to improve and validate the behavioral endpoints and determine whether the observed modulation is mostly formatting or gain. Any revised hypothesis remains exploratory until separately frozen and tested on fresh data.",
            "",
        ]
    if predictive_summary:
        lines += [
            "## Development prediction check",
            "",
            "Positive improvement means lower error after adding the learned context score. Intervals resample check documents with the fitted predictor held fixed.",
            "",
            "| Feature | Calibration source | MSE improvement | 95% descriptive interval |",
            "| --- | --- | ---: | ---: |",
        ]
        for row in predictive_summary:
            lines.append(
                f"| {row['feature_id']} | {row['train_source']} | {row['mean']:.4g} | [{row['low']:.4g}, {row['high']:.4g}] |"
            )
        lines.append("")
    lines += [
        "## Candidate support",
        "",
        "| Feature | Preparation status | Within-stratum observations |",
        "| --- | --- | ---: |",
    ]
    for feature in plan["features"]:
        lines.append(
            f"| {feature['feature_id']} | {feature['status']} | {feature.get('overlap', feature.get('support', 0))} |"
        )
    inactive = frame.loc[~frame.baseline_active, ["feature_id", "text_id"]].drop_duplicates()
    lines += [
        "",
        f"Inactive feature-prompt cases under the float32 focal encoder, despite selection from positive bfloat16 collection activations: {len(inactive)}. All remain reported; no outcome-dependent exclusion is applied.",
        "",
    ]
    captions = {
        "01_direction_controls": "Direction controls, pooled over development cases and fixed doses. Intervals resample documents; separate overlapping intervals are not a paired comparison.",
        "02_normalization_diagnostic": "Actual and normalization-only interactions for learned context directions. The surrogate does not remove all normalization mechanisms in the full transformer.",
        "03_layer_trajectory": "Layerwise residual interaction norms for the learned direction and one seeded random example. Curves show where interactions develop, not which pathway causes them.",
        "04_prediction_check": "Prediction error on development-check documents after pooled fit-only calibration. Lower is better; scalar gain constrains the output-effect direction.",
        "05_prediction_quality": "Interaction versus change in loss on the observed next token. Positive horizontal values indicate worse prediction. This is a local quality diagnostic.",
        "06_paired_control_check": "Paired learned-minus-mean-random comparisons on development-check documents. Beyond-gain values remove the best scalar rescaling of the baseline two-contrast effect. Intervals are descriptive and uncorrected.",
    }
    figure_order = [
        "06_paired_control_check",
        "01_direction_controls",
        "02_normalization_diagnostic",
        "03_layer_trajectory",
        "04_prediction_check",
        "05_prediction_quality",
    ]
    for number, name in enumerate([name for name in figure_order if name in figures], 1):
        lines += [
            f"![Figure {number}. {captions[name]}]({name}.png)",
            "",
            f"Figure {number}. {captions[name]}",
            "",
        ]
    lines += [
        "## Remaining requirements",
        "",
        "Review meaningful behavior contrasts; calibrate a minimally useful effect and sample size; freeze a new protocol; collect deduplicated fresh prompts; test pathway ablation/rescue and an independent SAE. Neither this pilot nor more plots can substitute for those steps.",
        "",
        "## Relation to prior work",
        "",
        "FEGA already characterizes context-dependent clouds of SAE intervention effects [1]. Steering side-effect prediction from feature statistics is also established [2], and the cylindrical representation hypothesis studies how normal-plane context modulates steering [3]. Component/bypass interaction and its relation to downstream curvature are prior results [4]. This pilot makes no claim of novelty from an effect plot or nonzero factorial interaction alone.",
        "",
        "## References",
        "",
        "[1] Hoang et al. (2026). [Sparse Autoencoders Encode Both Concepts and Functions: The Downstream Geometry of Feature Effects](https://arxiv.org/abs/2607.24645).",
        "",
        "[2] Duan (2026). [Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects](https://arxiv.org/abs/2606.08365).",
        "",
        "[3] Gao et al. (2026). [The Cylindrical Representation Hypothesis for Language Model Steering](https://arxiv.org/abs/2605.01844).",
        "",
        "[4] Vaidyanathan et al. (2026). [The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching](https://arxiv.org/abs/2606.27510).",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(
        out / "analysis_manifest.json",
        dict(
            status="development only",
            prompts=complete["prompts"],
            factorial_comparisons=len(frame),
            max_constraint_error=max_error,
            source_complete_sha256=sha256(results / "complete.json"),
            analysis_sha256=sha256(__file__),
        ),
    )
    print(out / "report.md", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.bundle, args.results, args.out)
