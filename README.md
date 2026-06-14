# SAE Feature Atlas

`sae_feature_atlas` is a research toolkit for building SAE feature atlases over Gemma Scope models

It is designed for the workflow we want:

```text
choose model + layer
-> automatically resolve the corresponding Gemma Scope SAE
-> choose corpus and activation storage mode
-> run one configurable pipeline
-> get feature cards, analysis tables, plots, and a readable report
-> build custom research on top of the generated feature-card dataset
```

The library is intentionally focused on Gemma + Gemma Scope 2 today, but the structure is meant to be extensible

## Main user workflow

Run a complete feature-atlas pipeline:

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus pile-10k \
  --max-texts 1500 \
  --max-seq-len 256 \
  --activation-mode topk \
  --top-k 64 \
  --steps all
```

The library automatically resolves:

```text
model=gemma-3-1b-pt
layer=13
site=resid_post
width=16k
l0=medium
```

into:

```text
model_name=google/gemma-3-1b-pt
sae_release=gemma-scope-2-1b-pt-res
sae_id=layer_13_width_16k_l0_medium
hook_name=blocks.13.hook_resid_post
```

## Quick start

```bash
git clone https://github.com/serafim-tkachenko/sae-feature-atlas.git
cd sae-feature-atlas
uv sync
uv run huggingface-cli login
```

Smoke test:

```bash
uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
```

Run a small debug pipeline:

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --max-texts 100 \
  --top-k 32 \
  --steps collect,features,report
```

Run a stronger pilot:

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --max-texts 1500 \
  --top-k 64 \
  --steps all
```

Try a 4B pilot:

```bash
uv run sae-atlas smoke-test \
  --model gemma-3-4b-pt \
  --layer 12 \
  --l0 small
```

Then:

```bash
uv run sae-atlas run \
  --model gemma-3-4b-pt \
  --layer 12 \
  --l0 small \
  --max-texts 200 \
  --top-k 32 \
  --steps collect,features,report
```

## Colab workflow

```python
!git clone https://github.com/serafim-tkachenko/sae-feature-atlas.git
%cd sae-feature-atlas

!curl -LsSf https://astral.sh/uv/install.sh | sh
!~/.local/bin/uv sync
```

Log in to Hugging Face:

```python
from huggingface_hub import notebook_login
notebook_login()
```

Run:

```python
!~/.local/bin/uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
!~/.local/bin/uv run sae-atlas run --model gemma-3-1b-pt --layer 13 --steps all
```

If Colab has environment conflicts:

```python
!~/.local/bin/uv pip install --system -e .
!sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
```

## Activation storage modes

### `topk`

```bash
--activation-mode topk --top-k 64
```

Stores only the strongest K SAE features per token

Pros:

- bounded storage,
- fast,
- good for first-pass feature cards and co-activation

Caveat:

- feature frequency means "frequency of appearing in top-K", not true positive activation frequency

### `positive`

```bash
--activation-mode positive
```

Stores all features with positive activation

Pros:

- more faithful for frequency/co-activation

Caveat:

- larger files,
- slower,
- more RAM/storage pressure

## Generated outputs

A run writes to:

```text
data/processed/<run_name>/
reports/<run_name>/
```

Typical data artifacts:

```text
token_metadata.parquet
sae_activations_topk.parquet
token_activation_summary.parquet
residual_vectors_sample.npy
residual_vectors_metadata.parquet
feature_stats.parquet
filtered_features.parquet
top_feature_examples.parquet
feature_cards.parquet
coactivation_pairs.parquet
decoder_neighbors.parquet
geometry_vs_coactivation.parquet
bimodal_feature_candidates.parquet
```

Typical report artifacts:

```text
reports/<run_name>/manifest.json
reports/<run_name>/summary.md
reports/<run_name>/index.html
reports/<run_name>/plots/*.png
reports/<run_name>/tables/*.html
```

## Library usage

```python
from sae_feature_atlas import GemmaScopeBundle
from sae_feature_atlas.registry import make_config

cfg = make_config(model="gemma-3-1b-pt", layer=13, max_texts=100, top_k=64)
bundle = GemmaScopeBundle(cfg).load()

print(bundle.validate())

values, indices = bundle.topk_features("The Eiffel Tower is in Paris.", top_k=10)
print(indices)
```

## Git policy

Commit:

```text
pyproject.toml
uv.lock
README.md
docs/
src/
scripts/
examples/
notebooks/ # only curated notebooks
reports/ # only hand-written reports or .gitkeep files
```

Do not commit generated data:

```text
data/raw/*.jsonl
data/processed/**/*.parquet
data/processed/**/*.npy
```

## License

Copyright © 2026 Serafim Tkachenko. All rights reserved.

No license is granted for reuse, redistribution, or derivative works without explicit written permission.
