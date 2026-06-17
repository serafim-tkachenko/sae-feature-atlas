# Concepts

This page explains the main concepts used by SAE Feature Atlas.

## Residual activations

A residual activation is the model's hidden state at a selected residual stream
hook for a specific token.

If the model has hidden size `d_model`, each token has a vector:

```text
h_token ∈ R^d_model
```

These are original model activations.

## SAE activations

A sparse autoencoder maps a residual activation into a sparse feature vector:

```text
h_token → SAE encoder → feature activations
```

Each SAE feature has an activation value for the token. Most features should be
inactive or near-zero for a given token.

## Top-k SAE activations

In `topk` mode, only the strongest K SAE features per token are saved.

This keeps storage manageable, but frequency estimates become top-k frequency,
not true activation frequency.

## Positive SAE activations

In `positive` mode, all positive SAE activations are saved.

This is more faithful but more expensive.

## SAE decoder directions

Each SAE feature has a decoder direction. Decoder directions live in the same
residual-stream space as the original residual activations.

Decoder geometry asks whether two SAE decoder directions are close in this
space, often using cosine similarity.

## Coactivation

Coactivation asks whether two SAE features are active on the same token or text.

Current same-token coactivation should not be interpreted as causal interaction.
It is an empirical usage signal.

## Decoder geometry vs coactivation

Decoder geometry and empirical coactivation are different signals.

Two features may have close decoder directions but rarely coactivate. Two
features may coactivate often but have distant decoder directions. Both cases are
interesting.

## Bimodality

A feature is a bimodality candidate if its activation strengths appear to have
two regimes.

The key research question is not only whether the distribution is bimodal, but
whether low-activation and high-activation examples differ in interpretable ways.

Relevant artifacts:

```text
bimodal_feature_candidates.parquet
bimodal_peak_examples.parquet
```

## Residual PCA

Residual PCA is fitted on sampled residual activation vectors.

It creates a diagnostic coordinate system ordered by variance. The first
components are high-variance directions in the sampled residual activation cloud.

## Residual PCA coordinates

Residual PCA coordinates are projections of residual activations onto PCA
components. They are not new model activations.

```text
residual activation      = original model vector
residual PCA coordinate  = projection of that vector onto PCA components
```

## Decoder PCA projection

SAE decoder directions can also be projected onto residual PCA components.

This answers:

```text
Does this SAE direction align with high-variance residual directions,
or is it spread across lower-variance components?
```

## Features for manual follow-up

Features for manual follow-up are feature hypotheses for later intervention experiments.
They are not evidence of causal intervention by themselves.

A feature should not be treated as a reliable intervention vector until tested with
a controlled intervention protocol.
