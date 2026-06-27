from __future__ import annotations

import pandas as pd
from tqdm import tqdm


def compute_feature_stats(
    populations,
    activation_mode: str,
) -> pd.DataFrame:
    """Compute stored and analysis statistics with explicit token denominators.

    Under top-k collection, stored frequencies mean retained top-k membership per
    collected token. They are not true positive SAE activation frequencies.
    """
    stored = populations.all_stored_activations
    analysis = populations.analysis_activations
    n_stored_tokens = populations.stored_tokens[["text_id", "token_pos"]].drop_duplicates().shape[0]
    n_analysis_tokens = populations.analysis_tokens[["text_id", "token_pos"]].drop_duplicates().shape[0]
    if n_stored_tokens == 0 or n_analysis_tokens == 0:
        raise ValueError("Stored and analysis token populations must both be non-empty.")

    stored_stats = (
        stored.groupby("feature_id")
        .agg(
            stored_activation_count=("activation", "size"),
            stored_text_count=("text_id", "nunique"),
        )
        .reset_index()
    )
    analysis_stats = (
        analysis.groupby("feature_id")
        .agg(
            analysis_activation_count=("activation", "size"),
            analysis_text_count=("text_id", "nunique"),
            mean_activation=("activation", "mean"),
            max_activation=("activation", "max"),
            p50_activation=("activation", "median"),
            p95_activation=("activation", lambda x: x.quantile(0.95)),
            p99_activation=("activation", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )
    feature_stats = stored_stats.merge(analysis_stats, on="feature_id", how="outer")
    count_columns = [
        "stored_activation_count",
        "stored_text_count",
        "analysis_activation_count",
        "analysis_text_count",
    ]
    feature_stats[count_columns] = feature_stats[count_columns].fillna(0).astype(int)
    feature_stats["stored_token_frequency"] = (
        feature_stats["stored_activation_count"] / n_stored_tokens
    )
    feature_stats["analysis_token_frequency"] = (
        feature_stats["analysis_activation_count"] / n_analysis_tokens
    )
    feature_stats["analysis_to_stored_support_ratio"] = (
        feature_stats["analysis_activation_count"]
        / feature_stats["stored_activation_count"].clip(lower=1)
    )
    feature_stats["stored_token_denominator"] = int(n_stored_tokens)
    feature_stats["analysis_token_denominator"] = int(n_analysis_tokens)
    feature_stats["activation_storage_mode"] = str(activation_mode)
    feature_stats["stored_frequency_semantics"] = (
        "retained_topk_membership_per_collected_token"
        if activation_mode == "topk"
        else "stored_positive_membership_per_collected_token"
    )

    # Compatibility aliases retain old consumers while keeping population meaning explicit.
    feature_stats["n_token_activations"] = feature_stats["analysis_activation_count"]
    feature_stats["n_texts"] = feature_stats["analysis_text_count"]
    feature_stats["token_frequency"] = feature_stats["analysis_token_frequency"]
    feature_stats["text_frequency"] = (
        feature_stats["analysis_text_count"]
        / max(1, populations.analysis_tokens["text_id"].nunique())
    )
    return feature_stats.sort_values("analysis_activation_count", ascending=False)


def build_top_examples(acts: pd.DataFrame, token_meta: pd.DataFrame, top_n: int = 20, context_window: int = 20) -> pd.DataFrame:
    rows: list[dict] = []
    token_groups = {
        text_id: group.sort_values("token_pos")["token_str"].tolist()
        for text_id, group in token_meta.groupby("text_id")
    }
    for feature_id, group in tqdm(acts.groupby("feature_id"), desc="Building top examples"):
        for rank, row in enumerate(group.sort_values("activation", ascending=False).head(top_n).itertuples(index=False), start=1):
            toks = token_groups[row.text_id]
            pos = int(row.token_pos)
            rows.append({
                "feature_id": int(feature_id),
                "rank": int(rank),
                "activation": float(row.activation),
                "text_id": int(row.text_id),
                "source": row.source,
                "token_pos": int(pos),
                "token_str": row.token_str,
                "left_context": "".join(toks[max(0, pos - context_window):pos]),
                "center_token": toks[pos] if pos < len(toks) else row.token_str,
                "right_context": "".join(toks[pos + 1 : pos + 1 + context_window]),
            })
    return pd.DataFrame(rows)
