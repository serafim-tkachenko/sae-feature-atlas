# Metrics guide

## Feature statistics

- `stored_activation_count`: persisted rows for the feature.
- `analysis_activation_count`: stored rows that pass analysis eligibility.
- `stored_token_frequency`: stored count divided by every collected token.
- `analysis_token_frequency`: analysis count divided by every eligible token.
- `analysis_to_stored_support_ratio`: analysis support divided by stored support.
- activation quantiles: computed on analysis activations.

Under top-k collection, stored frequency means retained top-k membership per collected token. It is not true-positive activation frequency.

## Coactivation

`coactivation_count`, Jaccard, PMI, `P(j|i)`, and `P(i|j)` share the same eligible-token universe and unique token-feature representation. PMI remains sensitive to low support, so pairs below the configured support threshold are not estimated. Absence from the saved table may mean unsupported, ineligible, or not retained by the storage guard; consult `coactivation_metadata.json`.

## Decoder geometry

`decoder_cosine` is descriptive cosine similarity between normalized decoder directions. Canonical pair keys allow an empirical pair `(i,j)` to match directed geometry edges in either orientation. Missing coactivation metrics remain null.

## Bimodality

`delta_bic` compares one- and two-component Gaussian fits on log activation. Candidate qualification additionally requires component-weight and separation thresholds. Evaluated rows record point count, both BICs, weights, ordered means, variances/stds, convergence, iterations, seed, `n_init`, fit status, and rank-censoring semantics.

## Decoder/residual-PC alignment diagnostic

`pc_mass_observed` is absolute squared decoder-direction projection into the sampled fitted residual-PC subspace. `pc_norm_mass_top_k` describes where that observed mass lies within a proper prefix of the fitted basis. `effective_pc_dim`, entropy, and center of mass describe its spread.

A normalized mass over all fitted PCs is identically one and is not emitted. These quantities do not measure reconstruction quality, reconstructed residual variance, semantic importance, or full dictionary coverage.

## Graph alignment

`gca_at_k` is exploratory overlap between retained decoder-neighbor and coactivation-neighbor sets. It inherits both graphs' support and retention policies.
