"""Regenerate analysis in a new run directory using an explicit local tokenizer."""

import argparse
import json
from pathlib import Path

from sae_feature_atlas.pipeline.reanalysis import reanalyze_saved_evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path, help="Local tokenizer.json")
    args = parser.parse_args()
    result = reanalyze_saved_evidence(args.run_dir, args.output_dir, args.manifest, args.tokenizer)
    print(json.dumps(result["counts"], indent=2))
