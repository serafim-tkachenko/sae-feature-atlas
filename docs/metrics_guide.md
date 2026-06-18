# Metrics guide

## Feature statistics

`token_frequency` is the share of token positions where a feature appears in the
saved activation table. In `topk` mode this means frequency among saved top-k
rows, not true positive activation frequency.

`p99_activation` is a high-percentile activation-strength summary.

## Coactivation

`coactivation_count` counts same-token occurrences of a feature pair.

`jaccard` normalizes overlap by union count.

`pmi` is pointwise mutual information. It can be noisy for rare pairs.

## Decoder geometry

`decoder_cosine` is cosine similarity between normalized SAE decoder directions.
It is a geometric signal, not semantic proof.

## Bimodality

`bimodality_score` is the BIC improvement of a two-component activation-strength
model over a one-component model. Higher values suggest the feature may have
weak/high activation regimes worth inspecting manually.

## Coverage

`pc_norm_mass_top_k` describes how much of a decoder direction's sampled PCA mass
lies in the first `k` residual PCA components.

`effective_pc_dim` is a participation-ratio style measure of spread across PCA
components.

`coverage_bucket` is a coarse triage bucket derived from coverage metrics.

## Graph alignment

`gca_at_k` measures overlap between top-k decoder-neighbor features and top-k
coactivation-neighbor features.
