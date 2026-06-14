from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-id", type=int, required=True)
    parser.add_argument("--n", type=int, default=8)
    return parser.parse_args()


def format_examples(rows: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)

    for row in rows.itertuples(index=False):
        print("-" * 120)
        print(f"activation={row.activation:.4f} source={row.source}")
        print(f"text_id={row.text_id} token_pos={row.token_pos}")
        print(f"{row.left_context}[{row.center_token}]{row.right_context}")


def main() -> None:
    args = parse_args()
    base = Path("data/processed") / args.run_name

    acts = pd.read_parquet(base / "sae_activations_topk.parquet")
    token_meta = pd.read_parquet(base / "token_metadata.parquet")

    feature_acts = acts[acts["feature_id"] == args.feature_id].copy()
    if feature_acts.empty:
        print(f"No activations found for feature {args.feature_id}")
        return

    token_groups = {
        text_id: group.sort_values("token_pos")["token_str"].tolist()
        for text_id, group in token_meta.groupby("text_id")
    }

    def add_context(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in df.itertuples(index=False):
            toks = token_groups[row.text_id]
            pos = int(row.token_pos)
            left = "".join(toks[max(0, pos - 20):pos])
            center = toks[pos] if pos < len(toks) else row.token_str
            right = "".join(toks[pos + 1 : pos + 21])
            rows.append(
                {
                    "activation": row.activation,
                    "source": row.source,
                    "text_id": row.text_id,
                    "token_pos": row.token_pos,
                    "left_context": left,
                    "center_token": center,
                    "right_context": right,
                }
            )
        return pd.DataFrame(rows)

    low = feature_acts.sort_values("activation", ascending=True).head(args.n)
    mid = feature_acts.iloc[
        (feature_acts["activation"] - feature_acts["activation"].median()).abs().argsort()
    ].head(args.n)
    high = feature_acts.sort_values("activation", ascending=False).head(args.n)

    print(f"Feature {args.feature_id}")
    print(feature_acts["activation"].describe())

    format_examples(add_context(low), "LOW ACTIVATION EXAMPLES")
    format_examples(add_context(mid), "MEDIAN ACTIVATION EXAMPLES")
    format_examples(add_context(high), "HIGH ACTIVATION EXAMPLES")


if __name__ == "__main__":
    main()