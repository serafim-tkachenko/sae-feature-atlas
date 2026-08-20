# SAE Feature Atlas

SAE Feature Atlas is a descriptive research toolkit for studying Gemma Scope sparse-autoencoder (SAE) features. Its central objects are empirical SAE activations, same-token interaction structure, decoder geometry, and reproducible context evidence. It does not currently make causal or validated semantic claims.

## Scientific populations

Every result should name the population it estimates:

- **All stored activations** are every sparse row persisted by collection. In `topk` mode, these are activations retained by the configured top-k procedure, not all positive SAE activations.
- **True-positive activations** are conceptually different. They are fully observed only in `positive` collection mode.
- **Analysis activations** are stored rows whose target tokens pass the explicit corpus, token-quality, position, and activation-row eligibility policy.
- **Analysis features** have sufficient support in the analysis population for configured downstream analyses.

`feature_stats.parquet` retains both stored-population and analysis-population counts. Frequencies use token denominators from token metadata, including tokens that retained no row for a particular feature.

## Current analyses

Core empirical analyses:

1. stored and analysis feature-activity statistics;
2. tokenizer-decoded activation examples with raw token evidence;
3. same-token coactivation on an explicit eligible-token universe;
4. decoder cosine neighbors and orientation-safe comparison with coactivation;
5. qualified GMM bimodality candidate detection;
6. feature and example inspection.

Diagnostic and exploratory analyses:

- residual-stream PCA;
- normalized-decoder PCA;
- decoder UMAP;
- decoder/residual-PC alignment;
- graph-neighborhood alignment.

PCA and UMAP are hypothesis-generation aids, not evidence of semantic axes or clusters. Decoder/residual-PC alignment is not SAE reconstruction quality, residual variance reconstructed, semantic importance, or dictionary coverage.

## Research direction

**Hypothesis / planned experiment -- not a current finding:** some SAE latents may not have a single activation-invariant contextual identity. Conditioning on activation magnitude may reveal distinct coactivation neighborhoods or semantic context distributions that are hidden by assigning one global description to the latent.

The next tests may include posterior-confident regime membership, frequency/support-matched controls, permutation or null baselines, regime-conditioned coactivation, representative contexts, annotation from empirical evidence, held-out evaluation, top-k sensitivity, and replication across SAE/model/layer configurations if the initial signal is real.

## Installation and first run

```bash
uv sync
uv run sae-atlas plan --preset research --model gemma-3-1b-pt --layer 13 --max-texts 100 --top-k 32
uv run sae-atlas smoke-test --model gemma-3-1b-pt --layer 13
uv run sae-atlas run --preset research --model gemma-3-1b-pt --layer 13 --max-texts 100 --top-k 32
```

Presets:

```text
core      collect
atlas     collect -> features -> coactivation -> geometry -> geometry-vs-coactivation
          -> bimodality -> inspection -> space -> cards -> report
research  atlas + coverage (alignment diagnostic) + graph alignment
```

The historical CLI step name `coverage` is retained, but it writes `decoder_residual_pc_alignment.parquet`.

## Important artifacts

```text
data/processed/<run_name>/
  source_texts.jsonl
  token_metadata.parquet
  sae_activations_topk.parquet or sae_activations_positive.parquet
  token_activation_summary.parquet
  residual_vectors_sample.npy
  feature_stats.parquet
  analysis_features.parquet
  top_feature_examples.parquet
  coactivation_pairs.parquet
  coactivation_metadata.json
  decoder_neighbors.parquet
  geometry_vs_coactivation.parquet
  bimodality_evaluated_features.parquet
  bimodal_feature_candidates.parquet
  bimodal_peak_examples.parquet
  inspection_feature_summaries.parquet
  inspection_pair_summaries.parquet
  residual_pca_summary.parquet
  decoder_pca_summary.parquet
  decoder_feature_pca.parquet
  decoder_feature_umap.parquet
  decoder_residual_pc_alignment.parquet
  feature_graph_alignment.parquet
  feature_cards.parquet
  lineage.json

reports/<run_name>/
  summary.md
  index.html
  inspection_report.md
  inspection_report.json
  manifest.json
```

Human-facing examples decode a contiguous token-ID span with the actual tokenizer. They also retain text and token positions, raw token IDs/strings, target token identity, activation, target quality, and display quality.

## Lineage and existing artifacts

`lineage.json` records an artifact schema version, deterministic collection and analysis fingerprints, Git SHA, and dirty state. Population-defining changes cannot silently reuse incompatible derived artifacts. The `features` step is the intentional boundary where a compatible raw collection may be reused to regenerate all analysis-policy-dependent outputs.

Usually reusable after verifying collection identity and internal coherence:

- source texts and token metadata;
- stored sparse activations;
- token activation summaries;
- residual samples, with sampling caveats.

Must be regenerated under this schema:

- feature statistics and analysis features;
- examples, coactivation, geometry/coactivation comparisons;
- bimodality outputs and inspection summaries;
- triage labels, graph alignment, feature cards, plots, and reports.

Legacy raw collections without lineage must be explicitly reviewed and migrated or recollected; their existence alone is not sufficient evidence of compatibility.

## Interpretation caveats

- Top-k observations are rank-censored; stored frequency is not true-positive activation frequency.
- Coactivation in top-k mode is joint retained feature membership on the same eligible token. PMI is support-sensitive; a minimum pair count is enforced.
- Missing empirical edges are not automatically observed zeros: they may be unsupported, ineligible, or omitted by retention/storage guards.
- Decoder geometry is descriptive and not proof of semantic equivalence or causal interaction.
- A two-component GMM preference does not prove two semantic concepts.
- `interpretability_triage_score` and triage labels prioritize review; they are not semantic descriptions or confidence estimates.
- Semantic annotations require empirical-context evidence, counterexamples, uncertainty, provenance, and held-out validation.
