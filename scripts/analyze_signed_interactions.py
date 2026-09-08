"""Exploratory signed diagnostics from saved factorial cells; no model execution."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sae_feature_atlas.scientific.intervention_math import paired_bootstrap
from sae_feature_atlas.scientific.intervention_analysis import prediction_checks


def parity_components(interactions):
    """Hadamard coefficients; mapping keys are (decoder sign, context sign)."""
    if set(interactions) != {(-1, -1), (-1, 1), (1, -1), (1, 1)}:
        raise ValueError("Require all four unique sign combinations")
    return {
        (a, b): sum(s**a * t**b * np.asarray(v) for (s, t), v in interactions.items()) / 4
        for a in (0, 1)
        for b in (0, 1)
    }


def estimate(values, groups):
    result = paired_bootstrap(values, groups)
    x = pd.DataFrame({"v": values, "g": groups}).groupby("g").v.mean().to_numpy()
    loo = (x.sum() - x) / (len(x) - 1) if len(x) > 1 else x
    return {**result, "leave_one_out_min": float(loo.min()), "leave_one_out_max": float(loo.max())}


def run(results, prior, output):
    output.mkdir(parents=True, exist_ok=True)
    rows, hashes, prediction_rows = [], {}, []
    for path in sorted(results.glob("prompt_*.json")):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        prompt = data["prompt"]
        seen_doses = set()
        groups = {}
        for record in data["records"]:
            if record["status"] != "ok":
                raise ValueError("Incomplete case")
            if record["dose"] not in seen_doses:
                seen_doses.add(record["dose"])
                cells = np.asarray(record["cells"])
                word = re.search(r"[A-Za-z]+$", prompt.get("excerpt", ""))
                prediction_rows.append(
                    {
                        **{
                            k: prompt[k]
                            for k in ("feature_id", "text_id", "duplicate_group", "source", "split")
                        },
                        "dose": record["dose"],
                        "effect": cells[1] - cells[0],
                        **{k: data["baseline"][k] for k in ("encoder", "norm", "context")},
                        **{f"pc{i}": v for i, v in enumerate(data["baseline"]["pc8"])},
                        "token_pos": prompt["token_pos"],
                        "token_id": str(prompt["token_id"]),
                        "support": np.log1p(prompt["n_positive_features"]),
                        "trailing_digit_count": prompt["trailing_digit_count"],
                        "trailing_letter_count": len(word.group()) if word else 0,
                    }
                )
            groups.setdefault((record["direction"], record["fraction"]), []).append(record)
        for (direction, fraction), records in groups.items():
            if len(records) != 4:
                raise ValueError("Unexpected dose/sign grid")
            interactions = {}
            for r in records:
                cells = np.asarray(r["cells"])
                value = cells[3] - cells[2] - cells[1] + cells[0]
                np.testing.assert_allclose(value, r["interaction"], atol=1e-10)
                interactions[(int(np.sign(r["dose"])), r["sign"])] = value
            p = parity_components(interactions)
            total = sum(v @ v for v in p.values())
            np.testing.assert_allclose(total, np.mean([v @ v for v in interactions.values()]))
            coefficient = p[1, 1] / (abs(records[0]["dose"]) * fraction)
            native = p[1, 1] / (abs(records[0]["alpha"]) * records[0]["length"])
            row = {
                k: prompt[k]
                for k in ("feature_id", "text_id", "duplicate_group", "source", "split")
            }
            row.update(
                direction=direction,
                fraction=fraction,
                energy=total,
                odd_fraction=float(p[1, 1] @ p[1, 1] / total) if total > 0 else np.nan,
                k_digit=coefficient[0],
                k_word=coefficient[1],
                native_k_digit=native[0],
                native_k_word=native[1],
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    if len(hashes) != 96 or len(frame) != 1344:
        raise ValueError("Expected complete 96-case pilot")
    frame.to_csv(output / "signed_coefficients.csv", index=False)
    summaries = []
    for (fid, direction, split), group in frame.groupby(["feature_id", "direction", "split"]):
        for endpoint in ["odd_fraction", "k_digit", "k_word"]:
            summaries.append(
                dict(
                    feature_id=fid,
                    direction=direction,
                    split=split,
                    endpoint=endpoint,
                    **estimate(group[endpoint], group.duplicate_group),
                )
            )
    pd.DataFrame(summaries).to_csv(output / "signed_summary.csv", index=False)

    index = ["feature_id", "direction", "split", "source", "duplicate_group", "text_id"]
    small = frame[frame.fraction == 0.005]
    large = frame[frame.fraction == 0.01]
    lengths = small.merge(large, on=index, suffixes=("_small", "_large"), validate="one_to_one")
    a = lengths[["k_digit_small", "k_word_small"]].to_numpy()
    b = lengths[["k_digit_large", "k_word_large"]].to_numpy()
    denominator = np.linalg.norm(a, axis=1) + np.linalg.norm(b, axis=1)
    lengths["relative_discrepancy"] = np.divide(
        2 * np.linalg.norm(a - b, axis=1),
        denominator,
        out=np.full(len(a), np.nan),
        where=denominator > 0,
    )
    lengths.to_csv(output / "length_consistency.csv", index=False)

    # Average lengths within each group before fitting or evaluating a constant vector.
    doc = frame.groupby(index)[["k_digit", "k_word"]].mean().reset_index()
    transfer = []
    errors = []
    for (fid, direction), group in doc.groupby(["feature_id", "direction"]):
        for source in ["pooled", "fineweb-edu-sample", "wikimedia-en"]:
            fit, check = group[group.split == "fit"], group[group.split == "check"]
            if source != "pooled":
                fit, check = fit[fit.source == source], check[check.source != source]
            mean = fit[["k_digit", "k_word"]].mean().to_numpy()
            target = check[["k_digit", "k_word"]].to_numpy()
            zero_error = np.mean(target**2, axis=1)
            fitted_error = np.mean((target - mean) ** 2, axis=1)
            improvement = zero_error - fitted_error
            transfer.append(
                dict(
                    feature_id=fid,
                    direction=direction,
                    calibration=source,
                    fit_documents=len(fit),
                    fit_digit=mean[0],
                    fit_word=mean[1],
                    **estimate(improvement, check.duplicate_group),
                )
            )
            for (_, r), e0, e1 in zip(check.iterrows(), zero_error, fitted_error):
                errors.append(
                    dict(
                        feature_id=fid,
                        direction=direction,
                        calibration=source,
                        duplicate_group=r.duplicate_group,
                        source=r.source,
                        zero_mse=e0,
                        fitted_mse=e1,
                    )
                )
    transfer = pd.DataFrame(transfer)
    transfer.to_csv(output / "signed_transfer.csv", index=False)
    pd.DataFrame(errors).to_csv(output / "signed_transfer_errors.csv", index=False)

    old = pd.read_csv(prior / "interventions.csv")
    # Preserve the original JSON identifier type and bootstrap ordering.
    old["duplicate_group"] = old.duplicate_group.astype(str)
    keys = [
        "feature_id",
        "text_id",
        "duplicate_group",
        "source",
        "split",
        "fraction",
        "sign",
        "dose",
    ]
    endpoints = ["interaction_norm", "beyond_gain_norm"]
    learned = old[(old.direction == "learned") & (old.split == "check")]
    controls = []
    for control in ["mean_random", "leading_pc", "approx_spectrum_matched", "shuffled_labels"]:
        other = (
            old[old.direction.str.startswith("random_")]
            if control == "mean_random"
            else old[old.direction == control]
        )
        other = other.groupby(keys)[endpoints].mean().reset_index()
        paired = learned.merge(other, on=keys, suffixes=("", "_control"), validate="one_to_one")
        for fid, group in paired.groupby("feature_id"):
            for endpoint in endpoints:
                valid = group.dropna(subset=[endpoint, endpoint + "_control"])
                controls.append(
                    dict(
                        feature_id=fid,
                        control=control,
                        endpoint=endpoint,
                        **estimate(
                            valid[endpoint] - valid[endpoint + "_control"], valid.duplicate_group
                        ),
                    )
                )
    controls = pd.DataFrame(controls)
    controls.to_csv(output / "specificity_controls.csv", index=False)
    forecasts = prediction_checks(pd.DataFrame(prediction_rows), include_pc_only=True)
    original = pd.read_csv(prior / "prediction_errors.csv")
    original["duplicate_group"] = original.duplicate_group.astype(str)
    prediction_keys = ["feature_id", "dose", "train_source", "model", "text_id", "duplicate_group"]
    reproduced = original.merge(
        forecasts, on=prediction_keys, suffixes=("_old", "_new"), validate="one_to_one"
    )
    if len(reproduced) != len(original):
        raise ValueError("Original prediction rows were not all reproduced")
    prediction_difference = float(np.max(np.abs(reproduced.mse_old - reproduced.mse_new)))
    if prediction_difference > 1e-12:
        raise ValueError("Original prediction errors changed")
    forecasts.to_csv(output / "incremental_prediction_errors.csv", index=False)
    paired_predictions = forecasts.pivot(
        index=["feature_id", "dose", "train_source", "duplicate_group", "text_id"],
        columns="model",
        values="mse",
    ).reset_index()
    incremental = []
    for (fid, source), g in paired_predictions.groupby(["feature_id", "train_source"]):
        incremental.append(
            dict(
                feature_id=fid,
                calibration=source,
                **estimate(g.pc8_only - g.pc8_context, g.duplicate_group),
            )
        )
    incremental = pd.DataFrame(incremental)
    incremental.to_csv(output / "incremental_prediction.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    ids = sorted(frame.feature_id.unique())
    colors = {1645: "#176b87", 2966: "#b17934", 28027: "#7860a0"}

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    for ax, fid in zip(axes, ids):
        group = frame[(frame.feature_id == fid) & (frame.split == "check")]
        order = [
            "learned",
            "leading_pc",
            "approx_spectrum_matched",
            "shuffled_labels",
            "random_0",
            "random_1",
            "random_2",
        ]
        for i, direction in enumerate(order):
            g = group[group.direction == direction].groupby("duplicate_group").odd_fraction.mean()
            ax.scatter(g, np.full(len(g), i), alpha=0.45, s=18, color=colors[fid])
            ax.plot(g.mean(), i, "k|")
        ax.set(title=f"Pseudo-concept {fid}", xlim=(0, 1.02), xlabel="Odd-odd energy fraction")
        ax.set_yticks(
            range(7),
            [
                d.replace("approx_spectrum_matched", "spectrum")
                .replace("shuffled_labels", "shuffled")
                .replace("_", " ")
                for d in order
            ]
            if ax == axes[0]
            else [],
        )
    save(fig, "07_sign_parity")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3))
    for ax, fid in zip(axes, ids):
        g = lengths[
            (lengths.feature_id == fid)
            & (lengths.direction == "learned")
            & (lengths.split == "check")
        ]
        for col, marker in [("k_digit", "o"), ("k_word", "^")]:
            ax.scatter(
                g[col + "_small"],
                g[col + "_large"],
                s=23,
                marker=marker,
                alpha=0.8,
                label=col.removeprefix("k_"),
            )
        lim = (
            np.max(
                np.abs(
                    g[["k_digit_small", "k_digit_large", "k_word_small", "k_word_large"]].to_numpy()
                )
            )
            * 1.08
        )
        ax.plot([-lim, lim], [-lim, lim], color="gray", ls="--", lw=1)
        ax.set(
            title=f"Pseudo-concept {fid}",
            xlabel="Coefficient at 0.5% chord",
            ylabel="Coefficient at 1% chord",
        )
        ax.legend(fontsize=8)
    save(fig, "08_length_consistency")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, fid in zip(axes, ids):
        g = doc[(doc.feature_id == fid) & (doc.direction == "learned")]
        for (source, split), subset in g.groupby(["source", "split"]):
            ax.scatter(
                subset.k_digit,
                subset.k_word,
                marker="o" if split == "check" else "x",
                color="#176b87" if source == "fineweb-edu-sample" else "#b17934",
                alpha=0.8,
                label=("Web" if source == "fineweb-edu-sample" else "Wiki") + " " + split,
            )
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set(
            title=f"Pseudo-concept {fid}",
            xlabel="Signed digit coefficient",
            ylabel="Signed word coefficient",
        )
        if ax == axes[0]:
            ax.legend(fontsize=7)
    save(fig, "09_signed_response")

    fig, axes = plt.subplots(3, 2, figsize=(9, 7.5))
    for i, fid in enumerate(ids):
        for j, endpoint in enumerate(endpoints):
            ax = axes[i, j]
            g = controls[
                (controls.feature_id == fid) & (controls.endpoint == endpoint)
            ].reset_index(drop=True)
            ax.errorbar(
                g["mean"] * 1000,
                range(len(g)),
                xerr=[(g["mean"] - g.low) * 1000, (g.high - g["mean"]) * 1000],
                fmt="o",
                color=colors[fid],
            )
            ax.axvline(0, color="gray", ls="--", lw=1)
            ax.set_yticks(
                range(len(g)),
                g.control.str.replace("approx_spectrum_matched", "spectrum").str.replace("_", " "),
            )
            ax.set_title(f"{fid}: " + ("raw interaction" if j == 0 else "beyond scalar gain"))
            if i == 2:
                ax.set_xlabel("Learned minus control\n(0.001 logit contrast norm)")
    save(fig, "10_specificity_controls")

    fig, axes = plt.subplots(1, 3, figsize=(10, 4.5))
    for ax, fid in zip(axes, ids):
        g = transfer[(transfer.feature_id == fid) & (transfer.calibration == "pooled")].reset_index(
            drop=True
        )
        ax.errorbar(
            g["mean"],
            range(len(g)),
            xerr=[g["mean"] - g.low, g.high - g["mean"]],
            fmt="o",
            color=colors[fid],
        )
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_yticks(
            range(len(g)),
            g.direction.str.replace("approx_spectrum_matched", "spectrum")
            .str.replace("shuffled_labels", "shuffled")
            .str.replace("_", " ")
            if ax == axes[0]
            else [],
        )
        ax.set(title=f"Pseudo-concept {fid}", xlabel="Zero MSE minus fitted-mean MSE")
    save(fig, "11_signed_transfer")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, fid in zip(axes, ids):
        g = incremental[incremental.feature_id == fid].reset_index(drop=True)
        ax.errorbar(
            g["mean"] * 1e6,
            range(len(g)),
            xerr=[(g["mean"] - g.low) * 1e6, (g.high - g["mean"]) * 1e6],
            fmt="o",
            color=colors[fid],
        )
        ax.axvline(0, color="gray", ls="--", lw=1)
        ax.set_yticks(
            range(len(g)),
            g.calibration.map(
                {
                    "fineweb-edu-sample": "Web to Wiki",
                    "wikimedia-en": "Wiki to Web",
                    "pooled": "Pooled",
                }
            )
            if ax == axes[0]
            else [],
        )
        ax.set(
            title=f"Pseudo-concept {fid}",
            xlabel="Incremental MSE improvement\n(0.000001 squared logit units)",
        )
    save(fig, "12_incremental_prediction")
    (output / "analysis_manifest.json").write_text(
        json.dumps(
            {
                "status": "post-result exploratory reanalysis; no new model execution",
                "prompt_files": hashes,
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "cases": 96,
                "coefficient_rows": len(frame),
                "bootstrap_draws": 2000,
                "original_prediction_max_absolute_difference": prediction_difference,
                "plan": "docs/signed_interaction_analysis.md",
            },
            indent=2,
        )
        + "\n"
    )
    print(controls.to_string(index=False))
    print(transfer[transfer.direction == "learned"].to_string(index=False))
    print(incremental.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("outputs/context_download_v3/full_v3"))
    parser.add_argument("--prior", type=Path, default=Path("reports/context_intervention_pilot_v1"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/context_intervention_pilot_v1/signed_analysis")
    )
    args = parser.parse_args()
    run(args.results, args.prior, args.output)
