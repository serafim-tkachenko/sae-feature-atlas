from __future__ import annotations

import json

import pandas as pd

from sae_feature_atlas.config import CFG
from sae_feature_atlas.features import build_and_save_feature_outputs
from sae_feature_atlas.io_utils import ensure_project_dirs
from sae_feature_atlas.manifest import build_run_manifest, write_run_manifest


def main() -> None:
    ensure_project_dirs()

    print(f"Loading {CFG.sae_activations_path}")
    acts = pd.read_parquet(CFG.sae_activations_path)

    print(f"Loading {CFG.token_metadata_path}")
    token_meta = pd.read_parquet(CFG.token_metadata_path)

    metrics = build_and_save_feature_outputs(acts, token_meta, CFG)

    manifest = build_run_manifest(CFG, metrics, stage="build_feature_cards")
    write_run_manifest(CFG, manifest)

    print("Feature-card outputs written")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
