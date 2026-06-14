# SAE Feature Atlas

`sae_feature_atlas` is a research toolkit for building interpretable feature atlases over Gemma Scope SAEs

The intended workflow is:

```text
choose model + layer
-> automatically resolve the matching Gemma Scope SAE
-> choose corpus and activation-storage mode
-> run one configurable pipeline
-> get feature cards, analysis tables, plots, automated inspection, and a readable report
-> use the generated feature-card dataset for downstream interpretability research
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

Small local run:

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus pile-10k \
  --max-texts 100 \
  --top-k 32 \
  --steps all
```

Serious pilot:

```bash
uv run sae-atlas run \
  --model gemma-3-1b-pt \
  --layer 13 \
  --corpus mixed-research \
  --max-texts 500 \
  --top-k 64 \
  --steps all
```

## Pipeline steps

```text
collect                    collect model residuals and SAE activations
features                   build feature statistics and top examples
coactivation               compute same-token feature co-activation
geometry                   compute nearest SAE decoder directions
geometry-vs-coactivation   compare decoder geometry with empirical co-activation
bimodality                 find activation-distribution bimodality candidates
space                      analyze residual-space and decoder feature-space PCA
inspection                 generate automated feature/pair triage reports
cards                      enrich feature_cards.parquet with all available metrics
report                     generate Markdown/HTML reports and plots
```

`--steps all` runs the complete pipeline in a sensible order.

## Corpus choices

```text
pile-10k              diverse small Pile sample - good debug/default pilot corpus
tinystories           simple narrative text - useful smoke test but narrow
fineweb-edu-sample    streaming educational web sample - better quality, heavier
mixed-research        mixed Pile + educational/manual technical snippets
jsonl:/path/file.jsonl custom corpus with {"source", "text"} rows
```

## Activation modes

### `topk`

```bash
--activation-mode topk --top-k 64
```

Stores the strongest K SAE features per token. Good for bounded storage and fast first-pass feature cards

Caveat: feature frequency means **frequency of appearing among saved top-K features**, not true positive activation frequency

### `positive`

```bash
--activation-mode positive
```

Stores all positive SAE features - larger and slower

## Main outputs

A run writes to:

```text
data/processed/<run_name>/
reports/<run_name>/
```

Important outputs:

```text
feature_cards.parquet
inspection_feature_summaries.parquet
inspection_pair_summaries.parquet
coactivation_pairs.parquet
decoder_neighbors.parquet
geometry_vs_coactivation.parquet
bimodal_feature_candidates.parquet
residual_pca_summary.parquet
decoder_pca_summary.parquet
decoder_feature_pca.parquet
reports/<run_name>/index.html
reports/<run_name>/inspection_report.md
```

## Documentation

- [Project overview](docs/project_overview.md)
- [Metrics guide](docs/metrics_guide.md)
- [Feature cards](docs/feature_cards.md)
- [Corpus guide](docs/corpus_guide.md)
- [Geometry methods](docs/geometry_methods.md)
- [Scientific roadmap](docs/scientific_roadmap.md)

## License

Copyright © 2026 Serafim Tkachenko. All rights reserved.

No license is granted for reuse, redistribution, or derivative works without explicit written permission.
