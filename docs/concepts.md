# Concepts

## Residual activations

Residual activations are vectors from a selected transformer residual-stream hook,
for example `blocks.13.hook_resid_post`.

## SAE activations

The selected SAE encodes residual activations into sparse feature
activations. The pipeline stores either top-k activations per token or all
positive activations, depending on `--activation-mode`.

## Feature cards

A feature card is a table row that merges available evidence for one feature:
frequency, top examples, inspection scores, bimodality, decoder neighbors,
coactivation, coverage, and graph alignment.

## Coactivation

Coactivation means two SAE features are active on the same token position. It is
a usage-overlap measure, not causal evidence.

## Decoder geometry

Decoder geometry compares SAE decoder directions by cosine similarity. Nearby
decoder directions may be related, but cosine similarity is not proof of shared
semantics.

## Bimodality

Bimodality analysis asks whether a feature's activation strengths look better
explained by two regimes than one. The artifact is named
`bimodal_feature_candidates.parquet` because this is only a statistical signal;
manual review should use `bimodal_peak_examples.parquet`.

## Residual PCA coverage

Coverage compares normalized SAE decoder directions with PCA components fitted on
sampled residual vectors. This helps describe where features sit relative to the
observed residual-space variance.
