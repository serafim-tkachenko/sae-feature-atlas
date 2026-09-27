from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from sae_feature_atlas.app import build_parser


def _inspect(*args: str) -> None:
    parsed = build_parser().parse_args(["inspect-run", *args])
    parsed.func(parsed)


def test_inventory_does_not_read_tables_or_write_files(tmp_path, capsys) -> None:
    # Inventory should work even for artifacts whose contents cannot be read.
    artifact = tmp_path / "feature_cards.parquet"
    artifact.write_bytes(b"not a parquet file")

    _inspect(str(tmp_path))

    output = capsys.readouterr().out
    assert "feature_cards.parquet" in output
    assert "True" in output
    assert "False" in output
    assert list(tmp_path.iterdir()) == [artifact]
    assert artifact.read_bytes() == b"not a parquet file"


def test_feature_inspection_selects_card_and_limits_sorted_examples(tmp_path, capsys) -> None:
    pd.DataFrame(
        [{"feature_id": 7, "label": "selected-card"}, {"feature_id": 8, "label": "other-card"}]
    ).to_parquet(tmp_path / "feature_cards.parquet")
    pd.DataFrame(
        [
            {"feature_id": 7, "activation": 1.0, "center_token": "lowest-example"},
            {"feature_id": 8, "activation": 100.0, "center_token": "other-feature"},
            {"feature_id": 7, "activation": 3.0, "center_token": "highest-example"},
            {"feature_id": 7, "activation": 2.0, "center_token": "middle-example"},
        ]
    ).to_parquet(tmp_path / "top_feature_examples.parquet")
    original = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    _inspect(str(tmp_path), "--feature", "7", "--examples", "2")

    output = capsys.readouterr().out
    assert "Feature 7" in output
    assert "selected-card" in output
    assert output.index("highest-example") < output.index("middle-example")
    for excluded in ("other-card", "lowest-example", "other-feature"):
        assert excluded not in output
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_missing_optional_artifacts_are_explained(tmp_path, capsys) -> None:
    _inspect(str(tmp_path), "--feature", "0")

    output = capsys.readouterr().out
    assert "Feature 0" in output
    assert "No feature card saved for this entry." in output
    assert "No examples saved for this entry." in output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["missing", "file"])
def test_invalid_run_directory_is_a_cli_error(tmp_path, capsys, kind) -> None:
    path = tmp_path / "run"
    if kind == "file":
        path.write_text("not a directory")

    with pytest.raises(SystemExit) as error:
        _inspect(str(path))

    assert error.value.code == 2
    assert "not an existing run directory" in capsys.readouterr().err


@pytest.mark.parametrize("count", ["0", "-1", "bad"])
def test_invalid_example_count_is_a_cli_error(tmp_path, capsys, count) -> None:
    with pytest.raises(SystemExit) as error:
        _inspect(str(tmp_path), "--examples", count)

    assert error.value.code == 2
    assert "must be a positive integer" in capsys.readouterr().err


def test_inspection_does_not_import_model_or_analysis_dependencies(tmp_path) -> None:
    script = """
import sys
for name in ("torch", "transformer_lens", "sae_lens", "datasets", "matplotlib", "sklearn"):
    sys.modules[name] = None
from sae_feature_atlas.app import main
main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "inspect-run", str(tmp_path), "--feature", "0"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Feature 0" in result.stdout
