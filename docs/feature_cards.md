# Feature cards

A feature card is a multi-evidence profile of one SAE feature.

The goal is to help a researcher inspect a feature quickly without pretending
that automated metrics are final semantic explanations.

## Main artifact

```text
feature_cards.parquet
```

The cards table merges feature-level evidence from several stages.

## Typical sections

### Activation evidence

Examples:

```text
token_count
text_count
activation_mean
activation_p95
activation_max
top_examples_json
```

Use this to understand how often and how strongly the feature appears.

### Bimodality evidence

Examples:

```text
bimodality_score
bimodal_low_examples_json
bimodal_high_examples_json
```

Use this to compare low-activation and high-activation regimes.

The key question is:

```text
Do low and high activation examples appear to represent different strength
levels or different contexts?
```

### Artifact-risk evidence

Examples:

```text
quote_risk
punctuation_risk
space_risk
boundary_risk
source_risk
```

These are triage signals for features that may represent formatting, tokenization
artifacts, or corpus artifacts.

### Coactivation evidence

Examples:

```text
top_coactivation_neighbors_json
coactivation_count
jaccard
pmi
```

Use this to see empirical neighbors.

### Decoder geometry evidence

Examples:

```text
top_decoder_neighbors_json
decoder_cosine
decoder_pca_x
decoder_pca_y
decoder_umap_x
decoder_umap_y
```

Use this to inspect geometric neighbors in SAE decoder space.

### Residual PCA coverage evidence

Examples:

```text
pc_norm_mass_top_20
effective_pc_dim
pc_entropy
pc_center_of_mass
coverage_bucket
```

Use this to understand whether a feature's decoder direction aligns with
high-variance residual directions or is spread across many residual PCs.

### Graph-alignment evidence

Examples:

```text
gca_at_5
gca_at_10
gca_at_20
graph_alignment_bucket
```

Use this to see whether decoder geometry neighbors and empirical coactivation
neighbors agree.

### Intervention-candidate evidence

Examples:

```text
atlas_intervention_score
intervention_risk_score
intervention_candidate_reason
```

These are candidate-selection signals only. They are not causal intervention results.

## How to read a feature card safely

Do:

- compare top examples;
- compare low/high activation examples for bimodal candidates;
- check whether the feature is corpus-specific;
- check whether geometry and coactivation agree;
- check artifact-risk signals;
- inspect limitations.

Do not:

- treat automated labels as final explanations;
- claim causal influence without intervention experiments;
- generalize from one corpus to all text;
- ignore activation mode and thresholds.

## Recommended manual workflow

1. Sort by a research-relevant score.
2. Pick a candidate feature.
3. Inspect top activation examples.
4. If bimodal, compare low/high examples.
5. Inspect coactivation neighbors.
6. Inspect decoder neighbors.
7. Check residual PCA coverage.
8. Write a short human note with confidence level.

## Suggested confidence labels

```text
low         interesting but unclear
medium      repeated pattern visible, but needs more evidence
high        clear repeated pattern across examples and metrics
artifact    likely tokenization/format/source artifact
```
