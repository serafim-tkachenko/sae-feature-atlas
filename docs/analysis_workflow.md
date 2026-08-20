# Analysis workflow

## Core empirical path

1. `collect` persists source texts, token metadata, stored sparse activations, token summaries, residual samples, and lineage.
2. `features` constructs explicit stored/analysis populations, writes dual-population statistics, selects analysis features, and decodes top contexts.
3. `coactivation` estimates supported same-token relationships on the explicit eligible-token universe.
4. `geometry` restricts both query and target sides to analysis features for empirical comparison.
5. `geometry-vs-coactivation` matches canonical unordered empirical pairs while retaining directed decoder rank.
6. `bimodality` records all evaluations separately from threshold-qualified candidates and builds representative posterior-confident examples.
7. `inspection` uses analysis activations and decoded evidence for triage.

## Diagnostic path

`space` computes residual PCA, normalized-decoder PCA, and decoder UMAP. `coverage` is a historical CLI name for decoder/residual-PC alignment. `alignment` compares retained geometry and coactivation neighborhoods. These are diagnostics or hypothesis-generation aids, not semantic evidence.

## Cards and reports

`cards` merges population-labeled empirical and diagnostic artifacts. `report` describes estimands and caveats. Triage labels are not semantic annotations.

## Lineage boundary

All non-collection steps require compatible lineage. `features` may accept a collection fingerprint match with a changed analysis fingerprint because it is the deliberate regeneration boundary; it then records the new analysis fingerprint. Later steps require the full match.
