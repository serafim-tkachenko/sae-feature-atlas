# Metrics guide

This page defines the main metrics and how to interpret them safely.

## Feature statistics

### Activation frequency

Measures how often a feature appears in saved activations.

In `topk` mode, this means:

```text
frequency of appearing among saved top-K features
```

not true positive activation frequency.

In `positive` mode, it is closer to true positive activation frequency.

### Mean / max / quantile activation

Summarizes activation strength for a feature.

Useful for ranking and sanity checks, but not sufficient for semantic
interpretation.

### Source coverage

Counts how many sources/domains contain a feature.

Useful for detecting whether a feature is broad or corpus-specific.

## Filtering metrics

Feature filtering usually removes:

- features with too few token occurrences;
- features with too few text occurrences;
- features that are too common;
- activation rows below a minimum activation threshold;
- source/token artifacts.

Filtering affects every downstream metric.

## Coactivation metrics

### Coactivation count

Raw number of times two features are active on the same token.

Limitation: favors common features.

### Jaccard overlap

```text
|A ∩ B| / |A ∪ B|
```

Measures overlap relative to the union of activation sets.

### Conditional probabilities

```text
P(feature_j active | feature_i active)
P(feature_i active | feature_j active)
```

Useful when the relationship is asymmetric.

### PMI

Pointwise mutual information highlights pairs that cooccur more than expected
under independence.

Limitation: can overemphasize rare pairs. Use minimum-count thresholds.

## Decoder geometry metrics

### Decoder cosine similarity

Cosine similarity between normalized SAE decoder directions.

It measures geometric closeness in residual-stream space.

It does not prove semantic similarity, coactivation, or causal interaction.

### Decoder neighbors

Top nearest decoder directions for each feature.

Useful for finding geometry neighborhoods.

## Geometry-vs-coactivation metrics

This compares:

```text
decoder neighbors
empirical coactivation neighbors
```

Important cases:

- high geometry / high coactivation;
- high geometry / low coactivation;
- low geometry / high coactivation;
- low geometry / low coactivation.

Disagreement cases are often the most interesting research examples.

## Bimodality metrics

### Bimodality score

A score based on whether a two-component distribution explains activation
strengths better than a one-component distribution.

Interpretation:

```text
higher score → stronger candidate for two activation regimes
```

Limitations:

- depends on activation mode;
- depends on number of examples;
- may reflect corpus/domain mixture;
- does not prove semantic strength difference by itself.

### Low/high activation-regime examples

The table:

```text
bimodal_peak_examples.parquet
```

is used to compare low-mode and high-mode examples for candidate features.

This is necessary for qualitative interpretation.

## Residual PCA coverage metrics

### Observed PCA mass

How much of a decoder direction's norm lies in the sampled PCA subspace.

Low observed mass means the PCA sample/components may not cover much of that
direction.

### Normalized top-k PC mass

Fraction of the observed PCA-subspace mass that lies in the first K PCs.

Useful for asking whether the direction is concentrated in high-variance residual
components.

### Effective PC dimension

Approximate number of residual PCs used by a decoder direction.

Lower values mean the direction is more concentrated. Higher values mean it is
more spread out.

### PC entropy

Entropy of the decoder direction's PCA projection profile.

Higher entropy means more spread across components.

### PC center of mass

Average component index weighted by projection mass.

Lower values indicate more alignment with early high-variance PCs.

## Graph-alignment metrics

### GCA@k

```text
GCA@k = |N_geom^k(i) ∩ N_coact^k(i)| / k
```

Measures agreement between decoder-neighbor graph and coactivation-neighbor
graph.

High GCA suggests geometry and empirical usage neighborhoods agree.

Low GCA suggests disagreement, which may be important.

## Intervention-candidate scores

Intervention scores rank features for later intervention experiments.

They do not perform interventions.

They should be treated as prioritization signals, not causal evidence.

## General interpretation rule

Every metric should be interpreted with:

- corpus;
- activation mode;
- thresholds;
- sample size;
- model/layer/SAE;
- limitations.

Avoid turning ranking metrics into strong semantic claims without qualitative
inspection and controlled validation.
