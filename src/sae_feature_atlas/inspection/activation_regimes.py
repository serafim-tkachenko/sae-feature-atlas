
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def _fit_two_component_log_gmm(values: np.ndarray) -> GaussianMixture | None:
    values = values[np.isfinite(values)]
    values = values[values > 0]
    if len(values) < 2:
        return None
    x = np.log1p(values).reshape(-1, 1)
    try:
        return GaussianMixture(n_components=2, random_state=0).fit(x)
    except Exception:
        return None


def _context_rows(feature_rows: pd.DataFrame, token_meta: pd.DataFrame, context_window: int) -> pd.DataFrame:
    """Attach token context to activation rows.

    This intentionally mirrors top-example style context so bimodal low/high regimes
    can be inspected by humans rather than treated as self-explanatory metrics.
    """
    if feature_rows.empty:
        return pd.DataFrame()
    meta = token_meta[["text_id", "token_pos", "source", "token_str"]].copy()
    rows: list[dict] = []
    grouped_meta = {int(tid): group.sort_values("token_pos") for tid, group in meta.groupby("text_id")}
    for row in feature_rows.to_dict("records"):
        text_id = int(row["text_id"])
        token_pos = int(row["token_pos"])
        group = grouped_meta.get(text_id)
        if group is None:
            continue
        left = group[(group["token_pos"] >= token_pos - context_window) & (group["token_pos"] < token_pos)]
        center = group[group["token_pos"] == token_pos]
        right = group[(group["token_pos"] > token_pos) & (group["token_pos"] <= token_pos + context_window)]
        rows.append(
            {
                **row,
                "source": row.get("source", center["source"].iloc[0] if not center.empty else ""),
                "left_context": "".join(left["token_str"].astype(str).tolist()),
                "center_token": center["token_str"].iloc[0] if not center.empty else "",
                "right_context": "".join(right["token_str"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def build_bimodal_peak_examples(
    acts: pd.DataFrame,
    token_meta: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    top_features: int = 50,
    examples_per_peak: int = 8,
    context_window: int = 20,
) -> pd.DataFrame:
    """Build low/high activation-regime examples for ranked bimodal features.

    The output addresses the analysis question behind bimodality: not only
    whether a feature has two activation-strength modes, but which token contexts
    belong to the low and high modes.
    """
    if acts.empty or candidates.empty:
        return pd.DataFrame()

    needed = {"feature_id", "text_id", "token_pos", "activation"}
    missing = needed - set(acts.columns)
    if missing:
        raise ValueError(f"Activation table is missing columns needed for regime examples: {sorted(missing)}")

    ranked_ids = candidates.head(top_features)["feature_id"].astype(int).tolist()
    bimodality_score_by_feature = dict(
        zip(
            candidates["feature_id"].astype(int),
            candidates.get("bimodality_score", pd.Series([np.nan] * len(candidates))),
        )
    )
    rows: list[pd.DataFrame] = []

    for feature_id in tqdm(ranked_ids, desc="Bimodal low/high examples"):
        feature_rows = acts[acts["feature_id"].astype(int).eq(feature_id)].copy()
        if feature_rows.empty:
            continue
        values = feature_rows["activation"].to_numpy(dtype=np.float64)
        gmm = _fit_two_component_log_gmm(values)
        if gmm is None:
            continue

        x = np.log1p(feature_rows["activation"].to_numpy(dtype=np.float64)).reshape(-1, 1)
        labels = gmm.predict(x)
        means = gmm.means_.reshape(-1)
        low_component = int(np.argmin(means))
        high_component = int(np.argmax(means))
        feature_rows["log_activation"] = x.reshape(-1)
        feature_rows["peak_label"] = np.where(labels == low_component, "low", "high")
        feature_rows["peak_log_mean_low"] = float(means[low_component])
        feature_rows["peak_log_mean_high"] = float(means[high_component])
        feature_rows["bimodality_score"] = float(bimodality_score_by_feature.get(feature_id, np.nan))

        low = feature_rows[feature_rows["peak_label"].eq("low")].sort_values("activation", ascending=False).head(examples_per_peak)
        high = feature_rows[feature_rows["peak_label"].eq("high")].sort_values("activation", ascending=False).head(examples_per_peak)
        examples = pd.concat([low, high], ignore_index=True)
        if examples.empty:
            continue
        rows.append(_context_rows(examples, token_meta, context_window=context_window))

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    keep = [
        "feature_id",
        "peak_label",
        "activation",
        "log_activation",
        "bimodality_score",
        "peak_log_mean_low",
        "peak_log_mean_high",
        "text_id",
        "token_pos",
        "source",
        "left_context",
        "center_token",
        "right_context"]
    keep = [col for col in keep if col in out.columns]
    return out[keep].sort_values(["bimodality_score", "feature_id", "peak_label", "activation"], ascending=[False, True, True, False]).reset_index(drop=True)


def peak_examples_json(examples: pd.DataFrame, feature_id: int, peak_label: str, n: int = 5) -> str:
    if examples.empty:
        return "[]"
    subset = examples[
        examples["feature_id"].astype(int).eq(int(feature_id)) & examples["peak_label"].astype(str).eq(peak_label)
    ]
    cols = [
        "activation",
        "source",
        "text_id",
        "token_pos",
        "left_context",
        "center_token",
        "right_context"]
    cols = [col for col in cols if col in subset.columns]
    records = subset.sort_values("activation", ascending=False).head(n)[cols].to_dict("records")
    return json.dumps(records, ensure_ascii=False)
