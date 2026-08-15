from __future__ import annotations

from dataclasses import replace

import pytest

from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.config.schema import PathsConfig
from sae_feature_atlas.pipeline.lineage import (
    StaleArtifactError,
    analysis_fingerprint_payload,
    require_compatible_lineage,
    write_lineage,
)


def test_token_quality_policy_change_prevents_silent_reuse(tmp_path) -> None:
    cfg = replace(
        make_config(run_name="lineage-test"),
        paths=PathsConfig(
            raw_texts_path=tmp_path / "raw.jsonl",
            data_root=tmp_path / "data",
            reports_root=tmp_path / "reports",
        ),
    )
    cfg.run_data_dir.mkdir(parents=True)
    write_lineage(cfg, stage="collect")
    require_compatible_lineage(cfg)

    changed_filter = replace(
        cfg.activation_filter,
        exclude_token_quality_kinds=("space", "special"),
    )
    changed = replace(cfg, activation_filter=changed_filter)
    with pytest.raises(StaleArtifactError, match="Analysis fingerprint mismatch"):
        require_compatible_lineage(changed)



def test_token_quality_implementation_is_part_of_analysis_fingerprint(tmp_path) -> None:
    cfg = replace(
        make_config(run_name="lineage-policy-test"),
        paths=PathsConfig(
            raw_texts_path=tmp_path / "raw.jsonl",
            data_root=tmp_path / "data",
            reports_root=tmp_path / "reports",
        ),
    )
    digest = analysis_fingerprint_payload(cfg)["token_quality_policy_digest"]

    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)
