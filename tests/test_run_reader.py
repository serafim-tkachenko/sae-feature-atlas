import json

import pandas as pd
import pytest

from sae_feature_atlas.storage.run import AtlasRun


def test_reader_rejects_file_as_run_directory(tmp_path):
    path = tmp_path / "file"
    path.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        AtlasRun.from_dir(path)


def test_reader_requires_explicit_storage_when_both_formats_exist(tmp_path):
    for mode, value in [("topk", 1), ("positive", 2)]:
        pd.DataFrame({"feature_id": [value]}).to_parquet(tmp_path / f"sae_activations_{mode}.parquet")
    run = AtlasRun.from_dir(tmp_path)
    with pytest.raises(ValueError, match="Both activation"):
        run.sae_activations()
    assert run.sae_activations(mode="positive").feature_id.tolist() == [2]
    (tmp_path / "lineage.json").write_text(json.dumps({
        "fingerprint_payloads": {"collection": {"collection": {"activation_mode": "positive"}}},
    }))
    assert run.sae_activations().feature_id.tolist() == [2]


def test_reader_rejects_corrupt_identity_instead_of_guessing(tmp_path):
    pd.DataFrame({"feature_id": [1]}).to_parquet(tmp_path / "sae_activations_topk.parquet")
    (tmp_path / "lineage.json").write_text("{")
    with pytest.raises(ValueError, match="Cannot read activation storage identity"):
        AtlasRun.from_dir(tmp_path).sae_activations()


def test_reader_does_not_substitute_a_different_recorded_storage_mode(tmp_path):
    pd.DataFrame({"feature_id": [1]}).to_parquet(tmp_path / "sae_activations_topk.parquet")
    (tmp_path / "lineage.json").write_text(json.dumps({
        "fingerprint_payloads": {"collection": {"collection": {"activation_mode": "positive"}}},
    }))
    with pytest.raises(FileNotFoundError, match="positive"):
        AtlasRun.from_dir(tmp_path).sae_activations()
