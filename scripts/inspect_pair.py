from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def print_feature_examples(top_examples: pd.DataFrame, feature_id: int, n: int) -> None:
    rows = top_examples[top_examples["feature_id"] == feature_id].sort_values("activation", ascending=False).head(n)
    print("\n" + "=" * 120)
    print(f"FEATURE {feature_id}")
    print("=" * 120)
    for row in rows.itertuples(index=False):
        print("-" * 120)
        print(f"rank={row.rank} activation={row.activation:.4f} source={row.source}")
        print(f"text_id={row.text_id} token_pos={row.token_pos}")
        print(f"{row.left_context}[{row.center_token}]{row.right_context}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-i", type=int, required=True)
    parser.add_argument("--feature-j", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()
    base = Path("data/processed") / args.run_name
    top_examples = pd.read_parquet(base / "top_feature_examples.parquet")
    for filename, label in [("geometry_vs_coactivation.parquet", "GEOMETRY / COACTIVATION ROW"), ("coactivation_pairs.parquet", "COACTIVATION ROW")]:
        path = base / filename
        if path.exists():
            df = pd.read_parquet(path)
            row = df[(df["feature_i"] == args.feature_i) & (df["feature_j"] == args.feature_j)]
            if not row.empty:
                print("\n" + label)
                print(row.head(1).to_string(index=False))
    print_feature_examples(top_examples, args.feature_i, args.n)
    print_feature_examples(top_examples, args.feature_j, args.n)


if __name__ == "__main__":
    main()
