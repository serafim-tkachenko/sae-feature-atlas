# Geometry methods

This document explains the geometry methods used in SAE Feature Atlas and how they relate to recent papers

## Three spaces

The project distinguishes three spaces:

1. **Residual activation space**: sampled residual-stream vectors from the model.
2. **SAE feature activation space**: sparse activation coefficients over SAE features.
3. **SAE decoder feature space**: the point cloud of normalized SAE decoder vectors.

Most feature-geometry plots in this project use the third space: normalized decoder vectors.

## PCA

PCA is the linear baseline.

It answers:

-> Is the feature or residual point cloud anisotropic? How much variance is captured by a few directions?

Outputs:

- `residual_pca_summary.parquet`
- `decoder_pca_summary.parquet`
- `decoder_feature_pca.parquet`

PCA is useful for large-scale shape/eigenvalue analysis, but it can be too coarse for local semantic neighborhoods

## UMAP

UMAP is a nonlinear visualization method used over normalized decoder vectors

It answers:

-> Do local neighborhoods of SAE decoder features suggest visible modular structure?

Outputs:

- `decoder_feature_umap.parquet`
- `decoder_umap_by_artifact_score.png`
- `decoder_umap_by_bimodality.png`
- `decoder_umap_by_manual_priority.png`

Caveat:

UMAP is an exploratory visualization - it can suggest local neighborhoods but does not prove semantic clustering

## optional projection

optional projection means **Linear Discriminant Analysis**

It answers:

-> Do heuristic labels such as `likely_artifact`, `bimodal_candidate`, or `high_frequency` separate in decoder space?

Outputs:

- ``
- `decoder_optional_supervised_projection_by_primary_label.png`

optional projection is supervised and should only be used after labels exist - current labels are heuristic triage labels, not ground truth
