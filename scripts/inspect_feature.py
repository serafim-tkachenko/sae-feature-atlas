from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-id", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    base = Path("data/processed") / args.run_name
    top_examples = pd.read_parquet(base / "top_feature_examples.parquet")
    rows = top_examples[top_examples["feature_id"] == args.feature_id].sort_values("activation", ascending=False).head(args.n)
    if rows.empty:
        print(f"No examples found for feature {args.feature_id}")
        return
    for row in rows.itertuples(index=False):
        print("=" * 120)
        print(f"feature={row.feature_id} rank={row.rank} activation={row.activation:.4f}")
        print(f"source={row.source} text_id={row.text_id} token_pos={row.token_pos}")
        print(f"{row.left_context}[{row.center_token}]{row.right_context}")


if __name__ == "__main__":
    main()
