# CLI reference

The CLI entrypoint is:

```bash
uv run sae-atlas <command>
```

## Discovery commands

### `list-models`

Lists supported model aliases.

```bash
uv run sae-atlas list-models
```

Use this before choosing `--model`.

### `list-presets`

Lists pipeline presets.

```bash
uv run sae-atlas list-presets
```

Typical presets:

```text
core       collect reusable activation artifacts
atlas      build feature atlas and report
research   atlas + residual coverage, graph alignment, features for manual follow-up
paper      larger research-style run
```

### `list-corpora`

Lists supported corpora.

```bash
uv run sae-atlas list-corpora
```

Use this to choose between small/debug corpora and broader research corpora.

## SAE/model checks

### `resolve-sae`

Shows which Gemma Scope SAE will be used for a model/layer/site selection.

```bash
uv run sae-atlas resolve-sae \
  --model gemma-3-1b-pt \
  --layer 13
```

### `smoke-test`

Loads model/SAE and checks basic compatibility.

```bash
uv run sae-atlas smoke-test \
  --model gemma-3-1b-pt \
  --layer 13
```

Run this before a large experiment.

## Planning command

### `plan`

Dry-run command for checking an experiment before execution.

```bash
uv run sae-atlas plan \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-broad \
  --max-texts 300 \
  --max-seq-len 256 \
  --activation-mode topk \
  --top-k 32
```

`plan` should show:

- model alias and resolved model name;
- layer and hook;
- SAE release/id;
- corpus;
- activation mode;
- selected pipeline steps;
- output directories;
- analysis-workflow coverage.

This command should not run model inference.

## Main run command

### `run`

```bash
uv run sae-atlas run \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-broad \
  --max-texts 300 \
  --max-seq-len 256 \
  --activation-mode topk \
  --top-k 32
```

Important options:

```text
--preset             core | atlas | research
--model              model alias, e.g. gemma-3-1b-pt
--layer              model layer
--site               residual site / hook family, default usually resid_post
--corpus             corpus id or jsonl:/path/file.jsonl
--max-texts          number of texts to process
--max-seq-len        token length per text
--activation-mode    topk | positive
--top-k              number of saved SAE features per token in topk mode
--steps              optional explicit comma-separated step list
```

## Report command

### `report`

Regenerates reports/plots from existing artifacts when supported.

```bash
uv run sae-atlas report \
  --run-name <run_name>
```

## Preset guidance

Use `core` when you only need activation artifacts.

Use `atlas` when you want feature statistics, coactivation, geometry, feature
cards, and report.

Use `research` when you want the full analysis-workflow workflow.

Use `research` with larger settings only after a smaller `research` run succeeds.

## Recommended workflow

```bash
uv run sae-atlas list-models
uv run sae-atlas list-corpora

uv run sae-atlas plan \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-broad \
  --max-texts 50

uv run sae-atlas smoke-test \
  --model gemma-3-1b-pt \
  --layer 13

uv run sae-atlas run \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-broad \
  --max-texts 50
```
