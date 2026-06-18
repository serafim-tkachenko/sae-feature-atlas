from __future__ import annotations

from pathlib import Path

import pandas as pd

from sae_feature_atlas.config.schema import ExperimentConfig


CONTEXT_WIDTH = 20


def _load_parquet(path: Path, description: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return pd.read_parquet(path)


def _format_context(row) -> str:
    return f"{row.left_context}[{row.center_token}]{row.right_context}"


def print_feature_examples(cfg: ExperimentConfig, feature_id: int, n: int = 10) -> None:
    """Print the top activation examples for one feature from a completed run."""
    top_examples = _load_parquet(cfg.top_examples_path, "top feature examples")
    rows = (
        top_examples[top_examples["feature_id"].astype(int).eq(int(feature_id))]
        .sort_values("activation", ascending=False)
        .head(n)
    )
    if rows.empty:
        print(f"No examples found for feature {feature_id} in {cfg.top_examples_path}")
        return

    for row in rows.itertuples(index=False):
        print("=" * 120)
        print(f"feature={row.feature_id} rank={row.rank} activation={row.activation:.4f}")
        print(f"source={row.source} text_id={row.text_id} token_pos={row.token_pos}")
        print(_format_context(row))


def _print_one_feature_block(top_examples: pd.DataFrame, feature_id: int, n: int) -> None:
    rows = (
        top_examples[top_examples["feature_id"].astype(int).eq(int(feature_id))]
        .sort_values("activation", ascending=False)
        .head(n)
    )
    print("\n" + "=" * 120)
    print(f"FEATURE {feature_id}")
    print("=" * 120)
    if rows.empty:
        print("No examples found.")
        return
    for row in rows.itertuples(index=False):
        print("-" * 120)
        print(f"rank={row.rank} activation={row.activation:.4f} source={row.source}")
        print(f"text_id={row.text_id} token_pos={row.token_pos}")
        print(_format_context(row))


def print_pair_examples(cfg: ExperimentConfig, feature_i: int, feature_j: int, n: int = 10) -> None:
    """Print diagnostics and top examples for a feature pair."""
    top_examples = _load_parquet(cfg.top_examples_path, "top feature examples")

    for path, label in [
        (cfg.geometry_vs_coactivation_path, "GEOMETRY / COACTIVATION ROW"),
        (cfg.coactivation_pairs_path, "COACTIVATION ROW"),
    ]:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not {"feature_i", "feature_j"}.issubset(df.columns):
            continue
        direct = df[
            (df["feature_i"].astype(int).eq(int(feature_i)))
            & (df["feature_j"].astype(int).eq(int(feature_j)))
        ]
        reverse = df[
            (df["feature_i"].astype(int).eq(int(feature_j)))
            & (df["feature_j"].astype(int).eq(int(feature_i)))
        ]
        row = pd.concat([direct, reverse], ignore_index=True).head(1)
        if not row.empty:
            print("\n" + label)
            print(row.to_string(index=False))

    _print_one_feature_block(top_examples, feature_i, n)
    _print_one_feature_block(top_examples, feature_j, n)


def _activation_context_rows(feature_rows: pd.DataFrame, token_meta: pd.DataFrame) -> pd.DataFrame:
    token_groups = {
        int(text_id): group.sort_values("token_pos")["token_str"].astype(str).tolist()
        for text_id, group in token_meta.groupby("text_id")
    }
    rows: list[dict] = []
    for row in feature_rows.itertuples(index=False):
        text_id = int(row.text_id)
        pos = int(row.token_pos)
        toks = token_groups.get(text_id, [])
        center = toks[pos] if 0 <= pos < len(toks) else getattr(row, "token_str", "")
        rows.append(
            {
                "activation": row.activation,
                "source": getattr(row, "source", ""),
                "text_id": text_id,
                "token_pos": pos,
                "left_context": "".join(toks[max(0, pos - CONTEXT_WIDTH) : pos]),
                "center_token": center,
                "right_context": "".join(toks[pos + 1 : pos + CONTEXT_WIDTH + 1]),
            }
        )
    return pd.DataFrame(rows)


def print_bimodal_feature_examples(cfg: ExperimentConfig, feature_id: int, n: int = 6) -> None:
    """Print low/median/high activation examples for one feature."""
    acts = _load_parquet(cfg.sae_activations_path, "SAE activations")
    token_meta = _load_parquet(cfg.token_metadata_path, "token metadata")
    feature_acts = acts[acts["feature_id"].astype(int).eq(int(feature_id))].copy()
    if feature_acts.empty:
        print(f"No activations found for feature {feature_id} in {cfg.sae_activations_path}")
        return

    print(f"Feature {feature_id}")
    print(feature_acts["activation"].describe())

    low = feature_acts.sort_values("activation").head(n)
    median_order = (feature_acts["activation"] - feature_acts["activation"].median()).abs().sort_values().index
    mid = feature_acts.loc[median_order].head(n)
    high = feature_acts.sort_values("activation", ascending=False).head(n)

    for title, df in [
        ("LOW ACTIVATION EXAMPLES", _activation_context_rows(low, token_meta)),
        ("MEDIAN ACTIVATION EXAMPLES", _activation_context_rows(mid, token_meta)),
        ("HIGH ACTIVATION EXAMPLES", _activation_context_rows(high, token_meta)),
    ]:
        print("\n" + "=" * 120)
        print(title)
        print("=" * 120)
        for row in df.itertuples(index=False):
            print("-" * 120)
            print(
                f"activation={row.activation:.4f} source={row.source} "
                f"text_id={row.text_id} token_pos={row.token_pos}"
            )
            print(_format_context(row))
