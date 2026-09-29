"""Explicit, separate reanalysis of saved evidence without claiming new collection lineage."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from tokenizers import Tokenizer

from sae_feature_atlas.analysis import token_quality
from sae_feature_atlas.analysis.bimodality import compute_bimodality
from sae_feature_atlas.analysis.coactivation import compute_same_token_coactivation
from sae_feature_atlas.analysis.populations import build_activation_populations
from sae_feature_atlas.config.schema import (
    CollectionConfig, ExperimentConfig, FeatureFilterConfig, ModelConfig, PathsConfig,
)
from sae_feature_atlas.inspection.activation_regimes import build_bimodal_peak_examples
from sae_feature_atlas.inspection.context import ContextRenderer
from sae_feature_atlas.inspection.feature_cards import (
    build_and_save_feature_outputs, enrich_feature_cards,
)
from sae_feature_atlas.inspection.inspection import (
    feature_summaries_to_frame, summarize_features_batch,
)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def reanalyze_saved_evidence(
    source: Path, output: Path, manifest_path: Path, tokenizer_path: Path,
) -> dict:
    """Apply current analysis defaults to saved activations in a new directory.

    The manifest supplies collection identity and feature selection, not the old
    token filter. A local tokenizer must reproduce every saved token string.
    Source artifacts remain untouched. This does not recreate collection lineage
    or certify that the original model inference was performed correctly.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest_path, tokenizer_path = Path(manifest_path), Path(tokenizer_path)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if output.is_relative_to(source):
        raise ValueError("Output must be outside the original run directory")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    collection = CollectionConfig(**manifest["collection"])
    if collection.activation_mode not in {"topk", "positive"}:
        raise ValueError("Manifest must identify topk or positive storage")
    cfg = ExperimentConfig(
        model=ModelConfig(**manifest["model"]),
        collection=replace(collection, run_name=output.name),
        feature_filter=FeatureFilterConfig(**manifest["feature_filter"]),
        paths=PathsConfig(data_root=output.parent),
    )
    activation_name = cfg.sae_activations_path.name
    acts = pd.read_parquet(source / activation_name)
    tokens = pd.read_parquet(source / "token_metadata.parquet")
    keys = ["text_id", "token_pos"]
    if tokens.duplicated(keys).any():
        raise ValueError("Duplicate token positions in saved metadata")
    joined = acts.merge(tokens[keys + ["token_str", "source"]], on=keys,
                        how="left", suffixes=("", "_metadata"), validate="many_to_one",
                        indicator=True)
    if (not joined["_merge"].eq("both").all()
            or not joined.token_str.eq(joined.token_str_metadata).all()
            or not joined.source.eq(joined.source_metadata).all()):
        raise ValueError("Saved activations disagree with token metadata")
    if not (np.isfinite(acts.activation) & acts.activation.gt(0)).all():
        raise ValueError("Expected finite positive sparse activations")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    decode = lambda ids: tokenizer.decode(ids, skip_special_tokens=False)
    for row in tokens[["token_id", "token_str"]].drop_duplicates().itertuples(index=False):
        if decode([int(row.token_id)]) != row.token_str:
            raise ValueError(f"Tokenizer does not reproduce saved token ID {row.token_id}")
    populations = build_activation_populations(acts, tokens, cfg.activation_filter)
    renderer = ContextRenderer(tokens, decode)
    renderer.render = lru_cache(maxsize=None)(renderer.render)

    inputs = {name: _sha256(source / name) for name in (activation_name, "token_metadata.parquet")}
    output.mkdir(parents=True)
    for name in inputs:
        shutil.copy2(source / name, output / name)
    counts = build_and_save_feature_outputs(populations, renderer, cfg)
    selected = pd.read_parquet(cfg.analysis_features_path)
    ids = set(selected.feature_id.astype(int))
    top = pd.read_parquet(cfg.top_examples_path)
    pairs = compute_same_token_coactivation(
        populations.analysis_activations, ids, populations.analysis_tokens,
        storage_mode=collection.activation_mode,
    )
    pairs.pairs.to_parquet(cfg.coactivation_pairs_path, index=False)
    cfg.coactivation_metadata_path.write_text(json.dumps(pairs.metadata, indent=2), encoding="utf-8")
    summaries = summarize_features_batch(
        populations.analysis_activations, top, sorted(ids), max_seq_len=collection.max_seq_len,
    )
    feature_summaries_to_frame(summaries).to_parquet(cfg.inspection_feature_summaries_path, index=False)
    bimodality = compute_bimodality(populations.analysis_activations, ids,
                                   storage_mode=collection.activation_mode)
    bimodality.evaluated_features.to_parquet(cfg.bimodality_evaluated_path, index=False)
    bimodality.candidates.to_parquet(cfg.bimodal_candidates_path, index=False)
    regimes = build_bimodal_peak_examples(populations.analysis_activations, renderer,
                                         bimodality.candidates)
    regimes.to_parquet(cfg.bimodal_peak_examples_path, index=False)
    notes = [
        "Examples, activation summaries, frequencies, coactivation, inspection and activation regimes "
        "were recomputed with the recorded token filter. Original collection remains unverified.",
        "Histograms retain all stored activations, including excluded target tokens. "
        "Punctuation in surrounding context is preserved.",
        "PCA, residual alignment and graph alignment were not regenerated; old values were not reused.",
    ]
    decoder_path = source / "decoder_neighbors.parquet"
    if decoder_path.exists():
        decoder = pd.read_parquet(decoder_path)
        if not decoder.empty:
            decoder = decoder[decoder.feature_i.isin(ids)]
        decoder.to_parquet(cfg.decoder_neighbors_path, index=False)
        inputs[decoder_path.name] = _sha256(decoder_path)
        notes.append("Decoder cosines reuse the original saved candidate edges; nearest neighbors "
                     "were not recomputed from model weights. Newly selected features may lack them.")
    enrich_feature_cards(cfg)
    record = {
        "schema_version": 1,
        "kind": "saved_evidence_reanalysis",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run": source.name,
        "source_sha256": inputs,
        "manifest_sha256": _sha256(manifest_path),
        "tokenizer_sha256": _sha256(tokenizer_path),
        "token_quality_sha256": _sha256(Path(token_quality.__file__)),
        "reanalysis_code_sha256": _sha256(Path(__file__)),
        "analysis_config": asdict(cfg.analysis),
        "activation_filter": asdict(cfg.activation_filter),
        "feature_filter": asdict(cfg.feature_filter),
        "counts": {**counts, "collected_texts": int(tokens.text_id.nunique()),
                   "tokenizer_mappings_checked": len(tokens[["token_id", "token_str"]].drop_duplicates())},
        "notes": notes,
    }
    record["output_sha256"] = {p.name: _sha256(p) for p in sorted(output.glob("*.parquet"))}
    (output / "reanalysis.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
