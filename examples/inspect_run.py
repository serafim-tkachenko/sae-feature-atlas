"""Inspect saved Atlas tables without loading model weights."""
import argparse

from sae_feature_atlas.storage import AtlasRun


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Existing data/processed/<run_name> directory")
    parser.add_argument("--feature", type=int, help="Dictionary entry to inspect")
    parser.add_argument("--examples", type=int, default=5)
    args = parser.parse_args()
    run = AtlasRun.from_dir(args.run_dir)
    print(run.artifact_status().to_string(index=False))
    if args.feature is not None:
        card = run.feature_card(args.feature)
        examples = run.feature_examples(args.feature, n=args.examples)
        print(f"\nFeature {args.feature}")
        print(card.to_string() if not card.empty else "No feature card saved for this entry.")
        print(examples.to_string(index=False) if not examples.empty else "No examples saved for this entry.")


if __name__ == "__main__":
    main()
