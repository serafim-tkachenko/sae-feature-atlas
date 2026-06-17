# Reporting and visualization

The report should help a researcher understand the run without reading all code
or opening every table manually.

## Main report files

```text
reports/<run_name>/summary.md
reports/<run_name>/index.html
reports/<run_name>/inspection_report.md
```

The report should be organized around research questions, not just file names.

## Recommended report structure

```text
1. Run overview
2. Analysis-workflow checklist
3. Activation dataset
4. Feature filtering
5. Coactivation analysis
6. Bimodality and activation regimes
7. Decoder geometry vs coactivation
8. Residual PCA coverage
9. Research candidates
10. Limitations
11. Artifact index
```

## Plot organization

Plots should be grouped by research question:

```text
plots/
  01_feature_filtering/
  02_coactivation/
  03_bimodality/
  04_decoder_geometry/
  05_residual_pca_coverage/
  06_graph_alignment/
  07_intervention_candidates/
```

This is easier to inspect than one flat plot directory.

## Tables to highlight

### Activation dataset

```text
token_metadata.parquet
sae_activations_topk.parquet or sae_activations_positive.parquet
residual_vectors_sample.npy
```

### Feature filtering

```text
feature_stats.parquet
filtered_features.parquet
```

### Coactivation

```text
coactivation_pairs.parquet
```

### Bimodality

```text
bimodal_feature_candidates.parquet
bimodal_peak_examples.parquet
```

### Decoder geometry

```text
decoder_neighbors.parquet
geometry_vs_coactivation.parquet
```

### Residual PCA coverage

```text
decoder_feature_pca.parquet
feature_coverage_profiles.parquet
```

### Feature cards

```text
feature_cards.parquet
```

## Report writing rules

Use precise language:

```text
This table ranks candidate bimodal features.
```

Avoid overclaiming:

```text
This proves that the feature controls semantic intensity.
```

Better:

```text
Low/high activation-regime examples should be manually inspected to determine
whether the two modes correspond to a meaningful semantic or contextual change.
```

## Limitations section

Every report should mention:

- corpus;
- activation mode;
- model/layer/SAE;
- max texts and max sequence length;
- filtering thresholds;
- whether results use top-k or positive activations;
- whether examples were manually inspected;
- whether intervention was actually performed.

## Research summary style

For each section, include:

```text
Question:
Evidence:
Main artifacts:
Limitations:
Next inspection step:
```

This makes the report useful for both the author and external researchers.
