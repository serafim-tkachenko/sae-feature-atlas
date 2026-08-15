# CLI guide

```bash
uv run sae-atlas <command>
```

Discovery and validation:

```bash
uv run sae-atlas list-models
uv run sae-atlas list-sites
uv run sae-atlas list-corpora
uv run sae-atlas list-presets
uv run sae-atlas resolve-sae --model gemma-3-1b-pt --layer 13 --site resid_post
uv run sae-atlas plan --preset research --model gemma-3-1b-pt --layer 13 --max-texts 100
uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
```

Run a preset or selected steps:

```bash
uv run sae-atlas run --preset core --max-texts 50
uv run sae-atlas run --preset atlas --max-texts 500
uv run sae-atlas run --preset research --max-texts 1500
uv run sae-atlas run --steps features,coactivation,geometry
```

`coverage` remains the CLI step name for compatibility but produces the diagnostic `decoder_residual_pc_alignment.parquet` artifact.

Non-collection steps fail when lineage is missing or incompatible. After an analysis-policy change, rerun `features` before downstream steps so all population-dependent outputs share the new fingerprint.
