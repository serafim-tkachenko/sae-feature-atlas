# Feature cards

The central output of the library is:

```text
data/processed/<run_name>/feature_cards.parquet
```

Each row corresponds to one SAE feature

## Core columns

| Column | Meaning |
|---|---|
| `feature_id` | SAE feature index |
| `n_token_activations` | Number of saved activation rows for this feature |
| `n_texts` | Number of texts where the feature appears |
| `token_frequency` | Frequency under the current activation storage mode |
| `text_frequency` | Fraction of texts where the feature appears |
| `mean_activation` | Mean saved activation |
| `max_activation` | Max saved activation |
| `p95_activation` | 95th percentile |
| `p99_activation` | 99th percentile |
| `top_examples_json` | Top activating examples with contexts |

## Analysis columns

Later pipeline steps can add or support:

| Column | Meaning |
|---|---|
| `coactivation_neighbors_json` | Strong same-token co-activation neighbors |
| `decoder_neighbors_json` | Nearest decoder directions |
| `bimodality_json` | Activation distribution mode information |
| `pca_alignment_json` | Alignment with residual PCA components |
| `natural_language_explanation` | Future NLA/SAEExplainer label |
| `human_notes` | Human annotation |
| `tags` | Optional tags |
