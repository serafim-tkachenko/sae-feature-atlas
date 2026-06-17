# Analysis workflow

This page maps SAE Feature Atlas to the current public research workflow.

The workflow is designed to support a coherent analysis of SAE features across
activation statistics, empirical coactivation, decoder geometry, bimodality, and
residual PCA coverage.

## Analysis item 0: collect SAE activations

Goal:

```text
Take an SAE for one model, collect a diverse text corpus, and save SAE
activations for direct notebook access.
```

Main commands:

```bash
uv run sae-atlas run --preset core ...
uv run sae-atlas run --preset research ...
```

Main artifacts:

```text
token_metadata.parquet
sae_activations_topk.parquet
sae_activations_positive.parquet
residual_vectors_sample.npy
residual_vectors_metadata.parquet
```

Status:

```text
implemented
```

Remaining improvements:

- make run artifacts easy to publish as Kaggle/Hugging Face datasets;
- add more example notebooks.

## Analysis item 1: filter features

Goal:

```text
Filter features by frequency and quality, for example removing features that
are too rare for reliable downstream analysis.
```

Main artifacts:

```text
feature_stats.parquet
filtered_features.parquet
```

Status:

```text
implemented
```

Research caution:

Filtering thresholds affect all downstream analysis.

## Analysis item 2: find coactivating features

Goal:

```text
Find features with high overlap by token, text, or sample.
```

Main artifact:

```text
coactivation_pairs.parquet
```

Current implementation focuses on same-token coactivation.

Status:

```text
implemented
```

Possible extension:

- add same-text or same-window coactivation modes;
- compare same-token vs same-text results.

## Analysis item 3: find bimodal activation-strength features

Goal:

```text
Find features whose activation-strength distribution is bimodal and compare
texts/tokens from the low-activation and high-activation regimes.
```

Main artifacts:

```text
bimodal_feature_candidates.parquet
bimodal_peak_examples.parquet
```

Status:

```text
implemented after UX patch, but requires qualitative review
```

Important:

The score only identifies candidates. The real research step is comparing
low-activation and high-activation examples.

Safe claim:

```text
We can rank candidate bimodal features and inspect examples from low/high
activation regimes.
```

Unsafe claim:

```text
Bimodality proves controllable semantic strength.
```

## Analysis item 4: decoder-neighbor directions and coactivation

Goal:

```text
Find close SAE decoder directions and check how often the corresponding features
activate together.
```

Main artifacts:

```text
decoder_neighbors.parquet
geometry_vs_coactivation.parquet
```

Status:

```text
implemented
```

Key research cases:

```text
high decoder cosine + high coactivation
high decoder cosine + low coactivation
low decoder cosine + high coactivation
```

## Analysis item 5: decoder directions in high/low-variance residual components

Goal:

```text
Check which SAE decoder directions lie in high-variance or low-variance
components of the residual activation space.
```

Main artifacts:

```text
residual_vectors_sample.npy
decoder_feature_pca.parquet
feature_coverage_profiles.parquet
```

Status:

```text
implemented
```

Interpretation caution:

Residual PCA is fitted on the sampled corpus. Results are corpus-dependent.

## Analysis item 6: spread across residual components

Goal:

```text
Measure how many residual PCA components each SAE decoder direction uses.
```

Main artifact:

```text
feature_coverage_profiles.parquet
```

Relevant columns:

```text
effective_pc_dim
pc_entropy
pc_center_of_mass
pc_norm_mass_top_k
```

Status:

```text
implemented
```

Possible improvement:

- add `n_pcs_for_50pct_mass`, `n_pcs_for_80pct_mass`, and
  `n_pcs_for_90pct_mass`.

## Open research synthesis

After items 0–6 are run, the next question is whether they form a coherent
picture.

Useful synthesis questions:

1. Do decoder-neighbor features also coactivate?
2. Do disagreement cases reveal useful structure?
3. Are bimodal features concentrated in certain residual PCA coverage buckets?
4. Are high-coverage or low-coverage features more interpretable?
5. Do features for manual follow-up correspond to features with coherent evidence across
   activation, geometry, coactivation, and coverage?

## Recommended report structure

```text
1. Run overview
2. Analysis checklist
3. Activation dataset
4. Feature filtering
5. Coactivation
6. Bimodality and activation regimes
7. Decoder geometry vs coactivation
8. Residual PCA coverage
9. Candidate features for later manual validation
10. Limitations
```
