# Metrics guide

## Feature statistics

### `n_token_activations`

Number of saved activation rows for a feature.

In `topk` mode, this means the number of times the feature appeared among saved top-K features

In `positive` mode, this means the number of positive activations

### `token_frequency`

```text
n_token_activations / number_of_considered_tokens
```

In top-k mode, this is not true positive activation frequency

### `n_texts` and `text_frequency`

How many texts contain the feature and what fraction of the corpus that represents

### `p95_activation`, `p99_activation`, `max_activation`

Activation-strength tail metrics - high values should be interpreted through examples and artifact diagnostics

## Co-activation metrics

### `coactivation_count`

Number of tokens where two features appear together

### `jaccard`

```text
count(i and j) / count(i or j)
```

High Jaccard means high overlap between saved token sets

### `PMI`

```text
log(P(i and j) / (P(i)P(j)))
```

PMI can highlight rare strong associations, but it can overemphasize rare artifacts

## Decoder geometry

### `decoder_cosine`

Cosine similarity between SAE decoder directions - it measures geometric similarity, not empirical co-occurrence

## Geometry/co-activation quadrants

| Quadrant | Interpretation |
|---|---|
| high cosine / high coactivation | geometrically similar and empirically co-active |
| high cosine / low coactivation | similar directions, context-separated |
| low cosine / high coactivation | co-active but geometrically distinct |
| low cosine / low coactivation | baseline / unrelated |

## Bimodality

### `bimodality_score`

```text
BIC(1-component GMM) - BIC(2-component GMM)
```

Computed on `log1p(activation)` - a high score indicates a statistical candidate for two activation regimes, not a confirmed semantic finding

## Automated inspection

### `artifact_score`

Heuristic score based on quote-like tokens, punctuation-like tokens, whitespace-like tokens, boundary-position concentration, and token concentration

### `inspection_labels`

Examples:

- `quote_like`
- `punctuation_like`
- `boundary_heavy`
- `token_concentrated`
- `source_concentrated`
- `likely_artifact`
- `semantic_candidate`

### `manual_priority`

Suggested order for human inspection: `high`, `medium`, or `low`.
