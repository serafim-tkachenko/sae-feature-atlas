import hashlib
import json
from dataclasses import asdict

import pandas as pd
import pytest
from tokenizers import Tokenizer, models

from sae_feature_atlas.config.registry import make_config
from sae_feature_atlas.pipeline.reanalysis import reanalyze_saved_evidence
from sae_feature_atlas.report.explorer import build_explorer_payload


@pytest.fixture
def legacy_evidence(tmp_path):
    source = tmp_path / "legacy"
    source.mkdir()
    surfaces = ["<bos>", "Here", "’", ".", "words"]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(surfaces)}))
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    tokens = pd.DataFrame([
        dict(text_id=0, token_pos=i, token_id=i, token_str=word, source="synthetic")
        for i, word in enumerate(surfaces)
    ])
    tokens.to_parquet(source / "token_metadata.parquet", index=False)
    acts = tokens.drop(columns="token_id").assign(feature_id=7, activation=[1., 2., 9000., 8000., 3.])
    acts.to_parquet(source / "sae_activations_topk.parquet", index=False)
    # Stale top examples must not limit the replacement to already saved targets.
    acts.iloc[[2]].assign(center_token="’").to_parquet(source / "top_feature_examples.parquet")
    cfg = make_config(run_name="legacy")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "model": asdict(cfg.model), "collection": asdict(cfg.collection),
        "feature_filter": {"min_feature_token_count": 1, "min_feature_text_count": 1,
                           "max_feature_token_frequency": 1.0},
        "activation_row_filter": {"exclude_token_positions": [0]},
    }))
    return source, tmp_path / "filtered", manifest, tokenizer_path


def test_reanalysis_replaces_quote_dominated_examples_and_metrics_together(legacy_evidence):
    source, output, manifest, tokenizer = legacy_evidence
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    record = reanalyze_saved_evidence(source, output, manifest, tokenizer)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    assert not (output / "lineage.json").exists()
    assert record["counts"]["analysis_token_count"] == 2
    examples = pd.read_parquet(output / "top_feature_examples.parquet")
    assert examples.center_token.tolist() == ["words", "Here"]
    assert examples.target_quality.eq("clean").all()
    assert examples.display_context.str.contains("’", regex=False).all()
    payload = build_explorer_payload(output)
    assert payload["run"]["provenance"] == "legacy"
    assert payload["run"]["reanalysis_source"] == "legacy"
    assert payload["run"]["collected_texts"] == 1
    feature = payload["features"][0]
    assert feature["support"] == 2 and feature["denominator"] == 2
    assert feature["p99"] == pytest.approx(2.99)
    assert feature["histogram"]["count"] == 5
    assert feature["histogram"]["edges"][-1] == 9000
    assert not any("formatting targets" in warning for warning in payload["warnings"])


def test_reanalysis_rejects_wrong_tokenizer_before_writing(legacy_evidence):
    source, output, manifest, tokenizer = legacy_evidence
    Tokenizer(models.WordLevel({"incorrect": 0})).save(str(tokenizer))
    with pytest.raises(ValueError, match="Tokenizer does not reproduce"):
        reanalyze_saved_evidence(source, output, manifest, tokenizer)
    assert not output.exists()


def test_reanalysis_refuses_overwriting_source_or_existing_destination(legacy_evidence):
    source, output, manifest, tokenizer = legacy_evidence
    for destination in [source, manifest.parent]:
        with pytest.raises(FileExistsError, match="new output directory"):
            reanalyze_saved_evidence(source, destination, manifest, tokenizer)
    with pytest.raises(ValueError, match="outside the original"):
        reanalyze_saved_evidence(source, source / "nested", manifest, tokenizer)


def test_reanalysis_rejects_inconsistent_token_evidence(legacy_evidence):
    source, output, manifest, tokenizer = legacy_evidence
    acts = pd.read_parquet(source / "sae_activations_topk.parquet")
    acts.loc[1, "token_str"] = "different"
    acts.to_parquet(source / "sae_activations_topk.parquet")
    with pytest.raises(ValueError, match="disagree with token metadata"):
        reanalyze_saved_evidence(source, output, manifest, tokenizer)
    assert not output.exists()
