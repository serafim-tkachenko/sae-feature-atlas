# SAE Feature Atlas

SAE Feature Atlas is a local research toolkit for exploring how Gemma Scope SAE
features cover and organize transformer activations.

The project focuses on **descriptive analysis**, not causal-control experiments. It collects model activations, encodes them through a Gemma Scope
SAE, and builds feature-level evidence from activation statistics, examples,
coactivation, decoder geometry, bimodality, residual-space coverage, and
inspection reports.

## What the pipeline answers

The current workflow is organized around a small analysis checklist:

1. Build a reproducible activation dataset.
2. Filter usable features and collect top activation examples.
3. Measure same-token feature coactivation.
4. Compare decoder-neighbor geometry with empirical coactivation.
5. Detect possible low/high activation regimes and save examples for manual review.
6. Inspect features and feature pairs for obvious artifacts.
7. Compare SAE decoder directions with sampled residual-stream PCA structure.
8. Build feature cards and markdown/html reports.

## Installation

```bash
uv sync
```

The package exposes one CLI:

```bash
uv run sae-atlas --help
```

## Recommended first run

Preview the exact resolved configuration and artifact dependencies:

```bash
uv run sae-atlas plan \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --site resid_post \
  --max-texts 100 \
  --top-k 32
```

Check that the selected model/SAE can load:

```bash
uv run sae-atlas smoke-test \
  --model gemma-3-1b-pt \
  --layer 13 \
  --site resid_post
```

Run the analysis:

```bash
uv run sae-atlas run \
  --preset research \
  --model gemma-3-1b-pt \
  --layer 13 \
  --site resid_post \
  --max-texts 100 \
  --top-k 32
```

## Presets

```text
core      collect
atlas     collect -> features -> coactivation -> geometry -> geometry-vs-coactivation -> bimodality -> inspection -> space -> cards -> report
research  atlas + coverage + alignment
```

## Steps

```text
collect                   collect token metadata, sparse SAE activations, and residual samples
features                  compute feature statistics, filters, top examples, and base feature cards
coactivation              compute same-token feature coactivation pairs
geometry                  compute nearest neighbors between SAE decoder directions
geometry-vs-coactivation  compare decoder-neighbor geometry with empirical coactivation
bimodality                score activation distributions and save low/high regime examples
inspection                build automated feature and pair inspection summaries
space                     compute residual PCA, decoder PCA, and decoder UMAP
coverage                  compare decoder directions with residual PCA components
alignment                 compare decoder-neighbor and coactivation-neighbor graphs
cards                     merge available evidence into canonical feature cards
report                    write markdown/html reports, plots, and table previews
```

## Inspection commands

```bash
uv run sae-atlas inspect-feature --run-name <run> --feature-id 123 --n 10
uv run sae-atlas inspect-pair --run-name <run> --feature-i 123 --feature-j 456 --n 10
uv run sae-atlas inspect-bimodal-feature --run-name <run> --feature-id 123 --n 6
```

Use these after running at least `collect` and `features` - pair inspection also
benefits from `coactivation` and `geometry-vs-coactivation`.

## Main artifacts

Generated artifacts live under:

```text
data/processed/<run_name>/
reports/<run_name>/
```

Important data artifacts:

```text
token_metadata.parquet
sae_activations_topk.parquet or sae_activations_positive.parquet
residual_vectors_sample.npy
feature_stats.parquet
filtered_features.parquet
top_feature_examples.parquet
coactivation_pairs.parquet
decoder_neighbors.parquet
geometry_vs_coactivation.parquet
bimodal_feature_candidates.parquet
bimodal_peak_examples.parquet
inspection_feature_summaries.parquet
inspection_pair_summaries.parquet
residual_pca_summary.parquet
decoder_pca_summary.parquet
decoder_feature_pca.parquet
decoder_feature_umap.parquet
feature_coverage_profiles.parquet
feature_graph_alignment.parquet
feature_cards.parquet
```

Important report artifacts:

```text
summary.md
index.html
inspection_report.md
plots/
tables/
manifest.json
```

## Interpretation caveats

- Feature cards are multi-evidence profiles, not final explanations.
- Automated labels are heuristic triage labels, not ground-truth semantics.
- In `topk` mode, feature frequency means frequency among stored top-k rows, not
  true positive activation frequency.
- Bimodality is a statistical signal - low/high examples require manual review.
- Decoder cosine and coactivation are descriptive relationships, not causal proof.
- Residual PCA coverage depends on sampled corpus, layer, and number of PCA components.
