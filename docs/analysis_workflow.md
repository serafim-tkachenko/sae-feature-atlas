# Analysis workflow

The workflow has three levels.

## `core`

Collects token metadata, SAE activations, token activation summaries, and a
sample of residual-stream vectors.

## `atlas`

Adds reusable feature-atlas artifacts:

1. feature statistics and filters;
2. top activation examples;
3. coactivation pairs;
4. decoder-neighbor geometry;
5. geometry/coactivation comparison;
6. bimodality scores and low/high examples;
7. automated inspection summaries;
8. residual PCA, decoder PCA, decoder UMAP;
9. feature cards and reports.

## `research`

Adds research extensions:

1. feature coverage profiles in residual PCA space;
2. graph alignment between decoder-neighbor and coactivation-neighbor graphs.

## Important distinction

`bimodal_feature_candidates.parquet` is a statistical ranking of activation
distributions. It does not mean that a feature has a final semantic explanation.
The useful research step is to compare the saved low/high context examples.
