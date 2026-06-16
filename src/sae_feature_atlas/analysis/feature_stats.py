from __future__ import annotations

import pandas as pd
from tqdm import tqdm


def compute_feature_stats(acts: pd.DataFrame, token_meta: pd.DataFrame) -> pd.DataFrame:
    relevant_positions = acts[["text_id", "token_pos"]].drop_duplicates()
    token_meta_relevant = token_meta.merge(relevant_positions, on=["text_id", "token_pos"], how="inner")
    n_total_tokens = token_meta_relevant[["text_id", "token_pos"]].drop_duplicates().shape[0]
    n_total_texts = token_meta_relevant["text_id"].nunique()
    if n_total_tokens == 0 or n_total_texts == 0:
        raise ValueError("No tokens left after activation-row filtering.")
    feature_stats = (
        acts.groupby("feature_id")
        .agg(
            n_token_activations=("activation", "size"),
            n_texts=("text_id", "nunique"),
            mean_activation=("activation", "mean"),
            max_activation=("activation", "max"),
            p50_activation=("activation", "median"),
            p95_activation=("activation", lambda x: x.quantile(0.95)),
            p99_activation=("activation", lambda x: x.quantile(0.99)),
        )
        .reset_index()
    )
    feature_stats["token_frequency"] = feature_stats["n_token_activations"] / n_total_tokens
    feature_stats["text_frequency"] = feature_stats["n_texts"] / n_total_texts
    return feature_stats.sort_values("n_token_activations", ascending=False)


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
