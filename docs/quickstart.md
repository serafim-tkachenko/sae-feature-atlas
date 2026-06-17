# Quickstart

This page shows the fastest path from a clean checkout to a small research run.

## 1. Install

```bash
git clone https://github.com/serafim-tkachenko/sae-feature-atlas.git
cd sae-feature-atlas
uv sync
uv run hf auth login
```

## 2. Check available options

```bash
uv run sae-atlas list-models
uv run sae-atlas list-presets
uv run sae-atlas list-corpora
```

## 3. Resolve the selected SAE

```bash
uv run sae-atlas resolve-sae \
  --model gemma-3-1b-pt \
  --layer 13
```

## 4. Smoke-test model/SAE compatibility

```bash
uv run sae-atlas smoke-test \
  --model gemma-3-1b-pt \
  --layer 13
```

## 5. Preview the run

```bash
uv run sae-atlas plan \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-broad \
  --max-texts 50 \
  --max-seq-len 128 \
  --activation-mode topk \
  --top-k 32
```

`plan` does not run the experiment. Use it to verify the selected model, layer,
SAE, corpus, steps, and output directories.

## 6. Run a small pilot

```bash
uv run sae-atlas run \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-broad \
  --max-texts 50 \
  --max-seq-len 128 \
  --activation-mode topk \
  --top-k 32
```

## 7. Inspect outputs

The run should produce:

```text
data/processed/<run_name>/
reports/<run_name>/
```

Start with:

```text
reports/<run_name>/summary.md
data/processed/<run_name>/feature_cards.parquet
data/processed/<run_name>/bimodal_peak_examples.parquet
```

## 8. Load from Python

```python
from sae_feature_atlas.storage import AtlasRun

run = AtlasRun.from_dir("data/processed/<run_name>")
cards = run.feature_cards()
bimodal_examples = run.bimodal_peak_examples()
```

## Recommended scaling path

1. `mixed-broad`, `max_texts=50`, `max_seq_len=128`, `top_k=32`
2. `mixed-broad`, `max_texts=300`, `max_seq_len=256`, `top_k=32`
3. `mixed-large`, `max_texts=2000`, `max_seq_len=512`, `top_k=64`
4. Consider `positive` activation mode only after the top-k workflow is stable.

## Common issues

### Hugging Face access

If model or dataset loading fails, check:

```bash
uv run hf auth login
```

Some models or datasets may also require accepting terms on Hugging Face.

### Memory pressure

Reduce:

```text
--max-texts
--max-seq-len
--top-k
```

Start small and scale after the first successful run.

### Interpreting top-k results

In `topk` mode, feature frequency means “appears among saved top-K features,” not
true positive activation frequency. Use `positive` mode for more faithful
frequency estimates when feasible.
