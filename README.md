# SAE Feature Atlas

`sae_feature_atlas` is a local research toolkit for collecting, storing,
inspecting, and visualizing SAE activations from transformer models.

The current reference path is **Gemma Scope**. The project is designed to make it
easy to choose a model, layer, SAE, and corpus; collect residual-stream and SAE
activations; and generate a readable research report with feature cards, plots,
and tables.

The core workflow is:

```text
choose model + layer + SAE
-> choose corpus and activation-storage mode
-> collect residual activations and SAE activations
-> filter features
-> analyze coactivation, decoder geometry, bimodality, and residual PCA coverage
-> generate feature cards, grouped plots, tables, and a research report
```

## What this project is

SAE Feature Atlas has two connected goals:

1. **Reusable toolkit** — collect and inspect SAE activation datasets in a
   reproducible way.
2. **Research workflow** — support the current analysis-workflow analysis around
   coactivation, decoder geometry, bimodality, residual PCA coverage, and
   cross-view feature analysis.

## Installation

```bash
git clone https://github.com/serafim-tkachenko/sae-feature-atlas.git
cd sae-feature-atlas
uv sync
uv run hf auth login
```

NB! Some datasets and models may require Hugging Face access, accepted model terms,
or a configured HF token.

## Quick checks

List supported model aliases:

```bash
uv run sae-atlas list-models
```

List pipeline presets:

```bash
uv run sae-atlas list-presets
```

List supported corpora:

```bash
uv run sae-atlas list-corpora
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

Preview a full run before executing it:

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

`plan` is a dry-run command. It should show the selected model, layer, hook,
SAE, corpus, steps, outputs, and how the run maps to the analysis-workflow checklist.

## Recommended first run

Start with a small run before moving to larger corpora:

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

Then scale up:

```bash
uv run sae-atlas run \
  --preset paper \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-large \
  --max-texts 2000 \
  --max-seq-len 512 \
  --activation-mode topk \
  --top-k 64
```


## Preset model

SAE Feature Atlas intentionally keeps only three presets:

```text
core      collect reusable activation artifacts
atlas     build a descriptive feature atlas for inspection
research  add cross-view analysis: residual PCA coverage and graph alignment
```

There is no separate `paper` preset. Larger or publication-oriented runs should
use `--preset research` with larger corpora and stricter review settings.

## Pipeline presets

The recommended interface is:

```bash
uv run sae-atlas run --preset <preset>
```

### `core`

Collects reusable low-level activation artifacts.

```text
collect
```

Use this when you only need token metadata, sampled residual vectors, and sparse
SAE activations.

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

Use this when you want feature statistics, coactivation pairs, decoder geometry,
bimodality candidates, feature cards, plots, and a compiled report.

```bash
uv run sae-atlas run \
  --preset atlas \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-research \
  --max-texts 300 \
  --max-seq-len 256 \
  --activation-mode topk \
  --top-k 32
```

### `research`

Builds the atlas and adds geometry-aware research metrics.

```text
collect → features → coactivation → geometry → geometry-vs-coactivation
→ bimodality → inspection → space → coverage → alignment → candidates
→ cards → report
```

Use this when you want the full analysis-workflow analysis with residual coverage,
graph alignment, and cross-view analysis scores.

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

## Manual step selection

Advanced users can select steps directly:

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-broad \
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
space                      compute residual PCA and decoder PCA/UMAP/optional projection
coverage                   project decoder directions onto residual PCA components
alignment                  compare decoder-neighbor graph with coactivation graph
candidates                 rank feature candidates for later manual validation
cards                      merge available metrics into feature_cards.parquet
report                     generate Markdown/HTML reports and plots
```

## Analysis-workflow checklist

The current research workflow is designed to cover the analysis workflow:

| # | Analysis-workflow item | Main artifacts |
|---:|---|---|
| 0 | Collect SAE activations for one model/SAE/corpus | `token_metadata`, `sae_activations_*`, `residual_vectors_sample` |
| 1 | Filter features by frequency / quality | `feature_stats`, `filtered_features` |
| 2 | Find coactivating features by token/sample overlap | `coactivation_pairs` |
| 3 | Find bimodal activation-strength features and inspect low/high regimes | `bimodal_feature_candidates`, `bimodal_peak_examples` |
| 4 | Find close decoder directions and compare with coactivation | `decoder_neighbors`, `geometry_vs_coactivation` |
| 5 | Check which decoder directions lie in high/low-variance residual PCs | `feature_coverage_profiles` |
| 6 | Estimate how many residual PCs each decoder direction uses | `effective_pc_dim`, `pc_entropy`, `pc_center_of_mass` in `feature_coverage_profiles` |

The most important qualitative artifact for point 3 is:

```text
bimodal_peak_examples.parquet
```

It is intended to compare examples from low-activation and high-activation modes
for candidate bimodal features.

## Corpus choices

Use `list-corpora` for the current registry:

```bash
uv run sae-atlas list-corpora
```

Typical options:

```text
pile-10k              diverse small Pile sample - good debug/default pilot corpus
tinystories           simple narrative text - useful smoke test but narrow
fineweb-edu-sample    educational web sample; better quality, heavier
wikimedia-en          encyclopedic text
openwebtext           web text, broader distribution
math-small            small math/scientific reasoning corpus
code-small            small code/technical corpus
mixed-research        mixed pilot corpus
mixed-broad           broader mixed corpus for less cherry-picked analysis
mixed-large           larger mixed corpus for research-style runs
jsonl:/path/file.jsonl custom corpus with {"source", "text"} rows
```

For objective analysis, prefer mixed corpora over a single narrow source. For
debugging, use smaller corpora and lower `max_texts`.

## Activation modes

### `topk`

```bash
--activation-mode topk --top-k 64
```

Stores the strongest K SAE features per token.

Advantages:

- bounded storage;
- faster first-pass analysis;
- useful for feature cards and exploratory work.

Caveat:

- feature frequency means “frequency of appearing among saved top-K features”,
  not true positive activation frequency.

### `positive`

```bash
--activation-mode positive
```

Stores all positive SAE features.

Advantages:

- more faithful activation-frequency estimates;
- better for final frequency/coactivation claims.

Caveat:

- larger and slower;
- may be much more expensive for broad corpora.

## Main outputs

A run writes to:

```text
data/processed/<run_name>/
reports/<run_name>/
```

Core outputs:

```text
token_metadata.parquet
sae_activations_topk.parquet or sae_activations_positive.parquet
residual_vectors_sample.npy
residual_vectors_metadata.parquet
token_activation_summary.parquet
```

Atlas outputs:

```text
feature_stats.parquet
filtered_features.parquet
top_feature_examples.parquet
feature_cards.parquet
coactivation_pairs.parquet
decoder_neighbors.parquet
geometry_vs_coactivation.parquet
bimodal_feature_candidates.parquet
bimodal_peak_examples.parquet
inspection_feature_summaries.parquet
inspection_pair_summaries.parquet
decoder_feature_pca.parquet
decoder_feature_umap.parquet

```

Research outputs:

```text
feature_coverage_profiles.parquet
feature_graph_alignment.parquet
graph_alignment_summary.parquet
feature_intervention_scores.parquet
```

Report outputs:

```text
reports/<run_name>/summary.md
reports/<run_name>/index.html
reports/<run_name>/inspection_report.md
reports/<run_name>/plots/
reports/<run_name>/tables/
```

## Notebook usage

The notebook-friendly access layer is the recommended way to inspect saved runs:

```python
from sae_feature_atlas.storage import AtlasRun

run = AtlasRun.from_dir("data/processed/<run_name>")

cards = run.feature_cards()
feature_stats = run.feature_stats()
coactivation = run.coactivation_pairs()
bimodal_examples = run.bimodal_peak_examples()

examples = run.feature_examples(feature_id=12345, n=20)
```

See `docs/notebook_usage.md`.

## Residual activations vs PCA coordinates

Residual activations are the original model activations at a selected residual
stream hook. For a model with hidden size `d_model`, each token has a residual
vector in `R^d_model`.

Residual PCA coordinates are not new model activations. They are a diagnostic
coordinate system fitted on a sample of residual vectors. PCA rotates the
activation cloud into directions of decreasing variance.

SAE decoder directions also live in residual-stream space. Decoder PCA coverage
asks where those SAE directions lie relative to high-variance and low-variance
directions of the residual activation cloud.

In short:

```text
residual activation      = original model vector for a token
residual PCA coordinate  = projection of that vector onto PCA components
decoder PCA projection   = projection of an SAE decoder direction onto PCA components
```

## What is a feature card?

A feature card is a multi-evidence profile of an SAE latent. Depending on the
selected steps, it can include:

```text
activation evidence:
  frequency, intensity, top examples, bimodality

bimodality evidence:
  low/high activation-regime examples

artifact evidence:
  quote/punctuation/space/boundary/source risk

coactivation evidence:
  empirical same-token feature neighbors

decoder geometry evidence:
  nearest decoder directions, PCA/UMAP/optional projection coordinates

residual coverage evidence:
  alignment with high/low variance residual PCA components

graph-alignment evidence:
  agreement between decoder-neighbor and coactivation neighborhoods

cross-view analysis evidence:
  atlas_intervention_score, intervention_risk_score, candidate reason
```

Automated labels and scores are triage signals, not final semantic explanations.

## Current non-goals

Current non-goals:

- stable top-level Python API;
- broad provider/plugin architecture;
- support for every SAE family;
- claiming causal intervention from qualitative examples alone;
- web dashboard before the core analysis-workflow workflow is reliable.

The immediate goal is to make one Gemma Scope path very clear, reproducible, and
useful for research inspection. Additional SAE backends such as Llama Scope can
be added later once the core artifact format and report workflow are stable.

## Documentation map

```text
docs/project_overview.md     project goals and architecture
docs/quickstart.md           first run and debugging workflow
docs/cli.md                  CLI reference
docs/configuration.md        configuration and presets
docs/corpus_guide.md         corpus choices and tradeoffs
docs/concepts.md             residual activations, SAE activations, PCA concepts
docs/metrics_guide.md        metric definitions and limitations
docs/geometry_methods.md     decoder geometry and residual PCA methods
docs/feature_cards.md        feature-card schema and interpretation
docs/analysis_plan.md          explicit analysis-workflow alignment
docs/reporting.md            report and plot organization
docs/notebook_usage.md       Python access layer for saved runs
docs/library_vision.md       toolkit vision and non-goals
```

## Git hygiene

Generated data and reports are ignored by default. Commit source code and stable
documentation, not large generated artifacts.

Recommended generated artifacts to keep out of git:

```text
data/processed/**
reports/**/plots/
reports/**/tables/
reports/**/*.json
reports/**/index.html
```

Commit curated `summary.md` or selected report snippets only when they are
intentionally part of a presentation or paper draft.

## License

Copyright © 2026 Serafim Tkachenko. All rights reserved.

No license is granted for reuse, redistribution, or derivative works without
explicit written permission.
