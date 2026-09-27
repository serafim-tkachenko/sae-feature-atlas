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

Inspect a saved run by its data directory:

```bash
uv run sae-atlas inspect-run data/processed/<run_name>
uv run sae-atlas inspect-run data/processed/<run_name> --feature 1645 --examples 5
```

`inspect-run` is read-only and does not load model weights or import the model runtime.
It lists artifact availability; `--feature` also prints the saved feature card and
highest-activation examples. Choose a feature ID from your run. `--examples` must
be a positive integer (default: 5). Missing optional tables are reported without
failing. Either activation file may be absent, depending on the collection mode.
Use an absolute directory path to inspect a run from outside the project.
