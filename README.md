# SAE Feature Atlas

`sae_feature_atlas` is a layered toolkit for working with Gemma Scope SAEs and building feature-level interpretability atlases.

The project has three clearly separated layers:

1. **Core toolkit** — resolve Gemma Scope SAE ids, load model/SAE pairs, collect residual activations, encode SAE activations, and save reusable artifacts.
2. **Feature atlas** — build feature statistics, examples, coactivation, decoder geometry, bimodality candidates, automated inspection, feature cards, and a readable report.
3. **Research extensions** — add residual-space coverage profiles, graph alignment between decoder geometry and empirical coactivation, and steering-candidate hypotheses for later causal validation.

The intended workflow is:

```text
choose model + layer
-> automatically resolve the matching Gemma Scope SAE
-> choose corpus and activation-storage mode
-> run one configurable pipeline preset
-> get feature cards, analysis tables, plots, automated inspection, and report
-> optionally run research-grade geometry analysis and steering-candidate selection
```

## Installation

```bash
git clone https://github.com/serafim-tkachenko/sae-feature-atlas.git
cd sae-feature-atlas
uv sync
uv run hf auth login
```

## Quick checks

List supported model aliases:

```bash
uv run sae-atlas list-models
```

List pipeline presets:

```bash
uv run sae-atlas list-presets
```

Resolve the Gemma Scope SAE selected by model/layer/site:

```bash
uv run sae-atlas resolve-sae \
  --model gemma-3-1b-pt \
  --layer 13
```

Smoke-test model/SAE compatibility:

```bash
uv run sae-atlas smoke-test \
  --model gemma-3-1b-pt \
  --layer 13
```

## Pipeline presets

The recommended interface is `sae-atlas run --preset ...`.

### `core`

Collects the reusable low-level artifacts.

```text
collect
```

Use this when you only need token metadata, sampled residual vectors, and sparse SAE activations.

```bash
uv run sae-atlas run \
  --preset core \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus pile-10k \
  --max-texts 100 \
  --max-seq-len 256 \
  --activation-mode topk \
  --top-k 32
```

### `atlas`

Builds the reusable feature atlas.

```text
collect → features → coactivation → geometry → geometry-vs-coactivation
→ bimodality → inspection → space → cards → report
```

Use this when you want feature cards, plots, tables, and a human-readable report.

```bash
uv run sae-atlas run \
  --preset atlas \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus pile-10k \
  --max-texts 300 \
  --max-seq-len 256 \
  --activation-mode topk \
  --top-k 32
```

`--steps all` defaults to the `atlas` preset for backwards compatibility.

### `research`

Builds the atlas and adds geometry-aware research metrics.

```text
collect → features → coactivation → geometry → geometry-vs-coactivation
→ bimodality → inspection → space → coverage → alignment → candidates
→ cards → report
```

Use this when you want the full research-grade feature cards with residual coverage, graph alignment, and steering-candidate scores.

```bash
uv run sae-atlas run \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus pile-10k \
  --max-texts 300 \
  --max-seq-len 256 \
  --activation-mode topk \
  --top-k 32
```

### `paper`

Currently uses the same steps as `research`, but is reserved for larger, more expensive runs.

Recommended starting point:

```bash
uv run sae-atlas run \
  --preset paper \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-research \
  --max-texts 2000 \
  --max-seq-len 512 \
  --activation-mode topk \
  --top-k 64
```

## Manual step selection

Advanced users can select steps directly:

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus pile-10k \
  --max-texts 300 \
  --steps collect,features,coactivation,geometry,bimodality,inspection,space,cards,report
```

Available steps:

```text
collect                    collect model residuals and SAE activations
features                   build feature statistics and top examples
coactivation               compute same-token feature coactivation
geometry                   compute nearest SAE decoder directions
geometry-vs-coactivation   compare decoder geometry with empirical coactivation
bimodality                 find activation-distribution bimodality candidates
inspection                 run automated feature/pair artifact triage
space                      compute residual PCA and decoder PCA/UMAP/LDA
coverage                   project decoder directions onto residual PCA components
alignment                  compare decoder-neighbor graph with coactivation graph
candidates                 rank feature candidates for later steering validation
cards                      merge all available metrics into feature_cards.parquet
report                     generate Markdown/HTML reports and plots
```

## Main outputs

A run writes to:

```text
data/processed/<run_name>/
reports/<run_name>/
```

### Core outputs

```text
token_metadata.parquet
sae_activations_topk.parquet or sae_activations_positive.parquet
residual_vectors_sample.npy
residual_vectors_metadata.parquet
token_activation_summary.parquet
```

### Atlas outputs

```text
feature_stats.parquet
filtered_features.parquet
top_feature_examples.parquet
feature_cards.parquet
coactivation_pairs.parquet
decoder_neighbors.parquet
geometry_vs_coactivation.parquet
bimodal_feature_candidates.parquet
inspection_feature_summaries.parquet
inspection_pair_summaries.parquet
decoder_feature_pca.parquet
decoder_feature_umap.parquet
decoder_feature_lda.parquet
reports/<run_name>/summary.md
reports/<run_name>/index.html
reports/<run_name>/inspection_report.md
```

### Research outputs

These are generated by the `research` and `paper` presets:

```text
feature_coverage_profiles.parquet
feature_graph_alignment.parquet
graph_alignment_summary.parquet
feature_steering_scores.parquet
```

The canonical final table is still:

```text
feature_cards.parquet
```

Research steps enrich this same table rather than creating a separate competing feature-card file.

## What is a feature card?

A feature card is a multi-evidence profile of an SAE latent. Depending on the selected steps, it can include:

```text
activation evidence:
  frequency, intensity, top examples, bimodality

artifact evidence:
  quote/punctuation/space/boundary/source risk

coactivation evidence:
  empirical same-token feature neighbors

decoder geometry evidence:
  nearest decoder directions, PCA/UMAP/LDA coordinates

residual coverage evidence:
  alignment with high/low variance residual PCA components

graph-alignment evidence:
  agreement between decoder-neighbor and coactivation neighborhoods

steering-candidate evidence:
  atlas_steering_score, steering_risk_score, candidate reason
```

Automated labels and scores are triage signals, not final semantic explanations.

## Corpus choices

```text
pile-10k              diverse small Pile sample; good debug/default pilot corpus
tinystories           simple narrative text; useful smoke test but narrow
fineweb-edu-sample    streaming educational web sample; better quality, heavier
mixed-research        mixed Pile + educational/manual technical snippets
jsonl:/path/file.jsonl custom corpus with {"source", "text"} rows
```

## Activation modes

### `topk`

```bash
--activation-mode topk --top-k 64
```

Stores the strongest K SAE features per token. This is good for bounded storage and fast first-pass feature cards.

Caveat: feature frequency means frequency of appearing among saved top-K features, not true positive activation frequency.

### `positive`

```bash
--activation-mode positive
```

Stores all positive SAE features. This is larger and slower, but gives a more faithful activation-frequency estimate.

## Interpreting research metrics

### Residual coverage

`coverage` computes how each normalized SAE decoder direction projects onto residual activation PCA components.

Important columns:

```text
pc_norm_mass_top_20      normalized mass in top residual PCs
effective_pc_dim         how many residual components the direction uses
pc_entropy               entropy of the component alignment profile
coverage_bucket          coarse coverage label
```

These metrics answer: does this SAE direction live mostly in high-variance activation space, or is it distributed across many lower-variance components?

### Graph alignment

`alignment` compares two notions of feature relatedness:

```text
decoder geometry:      nearest neighbors by decoder cosine
empirical coactivation: nearest neighbors by same-token activation overlap
```

Important columns:

```text
gca_at_5
gca_at_10
gca_at_20
graph_alignment_bucket
```

`GCA@k = |N_geom^k(i) ∩ N_coact^k(i)| / k`.

High agreement suggests decoder-space neighborhoods and empirical usage neighborhoods are aligned. Disagreement cases are often the most interesting for research.

### Steering candidates

`candidates` does not perform steering. It ranks features for later steering validation.

Important columns:

```text
atlas_steering_score
steering_risk_score
steering_candidate_reason
```

The intended future experiment is to compare naive SAE feature steering against atlas-guided steering candidate selection.

## Git hygiene

Generated data and reports are ignored by default. Commit source code and stable documentation, not large generated artifacts.

Recommended generated artifacts to keep out of git:

```text
data/processed/**
reports/**/plots/
reports/**/tables/
reports/**/*.json
reports/**/index.html
```

Commit curated `summary.md` or selected report snippets only when they are intentionally part of a presentation or paper draft.

## License

Copyright © 2026 Serafim Tkachenko. All rights reserved.

No license is granted for reuse, redistribution, or derivative works without explicit written permission.
