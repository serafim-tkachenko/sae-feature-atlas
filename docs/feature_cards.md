# Feature cards

The central artifact is:

```text
data/processed/<run_name>/feature_cards.parquet
```

Each row describes one SAE feature

A feature card should answer:

1. How often does the feature appear?
2. How strong are its activations?
3. What are its top examples?
4. Is it likely an artifact?
5. What features co-activate with it?
6. What decoder directions are close?
7. Is its activation distribution potentially bimodal?
8. Should a human inspect it?

## Column groups

### Identity

- `feature_id`
- `model_name`
- `layer`
- `hook_name`
- `sae_release`
- `sae_id`
- `run_name`
- `corpus`
- `activation_mode`
- `top_k`

### Activation statistics

- `n_token_activations`
- `n_texts`
- `token_frequency`
- `text_frequency`
- `mean_activation`
- `p50_activation`
- `p95_activation`
- `p99_activation`
- `max_activation`

### Examples

- `top_examples_json`

### Automated inspection

- `artifact_score`
- `manual_priority`
- `inspection_labels`
- `top_tokens_json`
- `top_positions_json`
- `quote_share`
- `punctuation_share`
- `boundary_share`
- `source_concentration`

### Co-activation

- `top_coactivation_neighbors_json`
- `max_coactivation_jaccard`
- `max_coactivation_pmi`

### Decoder geometry

- `top_decoder_neighbors_json`
- `max_decoder_cosine`
- `decoder_pc1`
- `decoder_pc2`

### Bimodality

- `bimodality_score`
- `log_mean_low`
- `log_mean_high`
- `activation_p50`
- `activation_p95`