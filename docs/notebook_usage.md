# Notebook usage

Use the notebook-friendly access layer to inspect saved runs without remembering
individual file paths.

## Load a run

```python
from sae_feature_atlas.storage import AtlasRun

run = AtlasRun.from_dir("data/processed/<run_name>")
```

## Load common tables

```python
cards = run.feature_cards()
feature_stats = run.feature_stats()
coactivation = run.coactivation_pairs()
decoder_neighbors = run.decoder_neighbors()
geometry_vs_coactivation = run.geometry_vs_coactivation()
bimodal_candidates = run.bimodal_feature_candidates()
bimodal_examples = run.bimodal_peak_examples()
coverage = run.feature_coverage_profiles()
```

## Inspect one feature

```python
feature_id = 12345

card = cards[cards["feature_id"] == feature_id]
examples = run.feature_examples(feature_id=feature_id, n=20)
bimodal = bimodal_examples[bimodal_examples["feature_id"] == feature_id]
```

## Compare low/high activation regimes

```python
feature_id = 12345

rows = bimodal_examples[bimodal_examples["feature_id"] == feature_id]

low = rows[rows["peak_label"] == "low"]
high = rows[rows["peak_label"] == "high"]
```

Questions to ask:

- Do low and high examples differ semantically?
- Are high examples simply longer, more common, or from a different corpus?
- Are examples dominated by punctuation or formatting?
- Does the difference survive across sources?

## Inspect coactivation neighbors

```python
feature_id = 12345

pairs = coactivation[
    (coactivation["feature_i"] == feature_id) |
    (coactivation["feature_j"] == feature_id)
]
```

Sort by:

```text
coactivation_count
jaccard
pmi
p_j_given_i
p_i_given_j
```

Use count thresholds before trusting PMI.

## Inspect decoder neighbors

```python
feature_id = 12345

neighbors = decoder_neighbors[decoder_neighbors["feature_id"] == feature_id]
```

Compare decoder neighbors with coactivation neighbors to find agreement and
disagreement cases.

## Inspect residual PCA coverage

```python
feature_id = 12345

coverage_row = coverage[coverage["feature_id"] == feature_id]
```

Useful columns:

```text
effective_pc_dim
pc_entropy
pc_center_of_mass
pc_norm_mass_top_20
coverage_bucket
```

## Suggested notebook sections

1. Load run.
2. Show run metadata.
3. Show top features by frequency.
4. Show top coactivation pairs.
5. Show top bimodality candidates.
6. Compare low/high examples for selected features.
7. Compare geometry-vs-coactivation quadrants.
8. Inspect residual PCA coverage.
9. Pick candidate features for manual review.
10. Write short notes and limitations.

## Notebook interpretation rule

Treat notebook analysis as exploratory unless:

- the corpus is broad enough;
- thresholds are documented;
- examples are manually inspected;
- results are stable across reasonable parameter changes;
- claims are explicitly limited.
