# Geometry methods

## Decoder neighbors

Nearest neighbors use cosine similarity between normalized SAE decoder directions. The empirical comparison uses exactly the analysis-feature universe on both source and target sides. Decoder edges retain direction and rank.

## Geometry/coactivation comparison

Coactivation identity is unordered, so `(i,j)` and `(j,i)` share a canonical pair key. An unmatched directed geometry edge is marked missing rather than assigned zero coactivation. This matters when pairs are unsupported, ineligible, or absent because of storage retention.

## Exploratory projections

Decoder PCA and decoder UMAP are unsupervised diagnostics for anisotropy, visualization, and hypothesis generation. A visible cluster is not evidence of a semantic category.

Decoder/residual-PC alignment measures squared projection of decoder directions into a sampled residual PCA basis. It does not measure SAE reconstruction, reconstructed residual variance, feature importance, or full activation-space coverage.
