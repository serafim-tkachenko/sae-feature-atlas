from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--feature-i", type=int, required=True)
    parser.add_argument("--feature-j", type=int, required=True)
    parser.add_argument("--n", type=int, default=10)
    return parser.parse_args()


def print_feature_examples(top_examples: pd.DataFrame, feature_id: int, n: int) -> None:
    rows = (
        top_examples[top_examples["feature_id"] == feature_id]
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
        print(f"{row.left_context}[{row.center_token}]{row.right_context}")


def main() -> None:
    args = parse_args()
    base = Path("data/processed") / args.run_name

    top_examples = pd.read_parquet(base / "top_feature_examples.parquet")

    geom_path = base / "geometry_vs_coactivation.parquet"
    coact_path = base / "coactivation_pairs.parquet"

    if geom_path.exists():
        geom = pd.read_parquet(geom_path)
        row = geom[
            (geom["feature_i"] == args.feature_i)
            & (geom["feature_j"] == args.feature_j)
        ]
        if not row.empty:
            print("\nGEOMETRY / COACTIVATION ROW")
            print(row.head(1).to_string(index=False))

    if coact_path.exists():
        coact = pd.read_parquet(coact_path)
        row = coact[
            (coact["feature_i"] == args.feature_i)
            & (coact["feature_j"] == args.feature_j)
        ]
        if not row.empty:
            print("\nCOACTIVATION ROW")
            print(row.head(1).to_string(index=False))

    print_feature_examples(top_examples, args.feature_i, args.n)
    print_feature_examples(top_examples, args.feature_j, args.n)


if __name__ == "__main__":
    main()