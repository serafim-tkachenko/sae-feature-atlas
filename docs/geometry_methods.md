# Geometry methods

The geometry part of the project uses unsupervised diagnostics only.

## Decoder PCA

PCA over normalized SAE decoder vectors. Useful for checking global anisotropy
and broad structure in decoder space.

## Decoder UMAP

UMAP over normalized SAE decoder vectors. Useful for visualization and manual
triage. UMAP plots can be colored by artifact score, bimodality score, or manual
priority, but the embedding itself is not evidence of semantic clusters.

## Decoder-neighbor geometry

Nearest-neighbor search by decoder cosine similarity. This is compared with
same-token coactivation to identify where geometric closeness agrees or disagrees
with empirical usage overlap.
