# Feature cards

Feature cards are the canonical per-feature output of the project.

They are not final explanations. They are compact evidence profiles that make it
easier to decide what to inspect next.

## Typical columns

```text
feature_id
primary_label
manual_priority
inspection_labels
artifact_score
semantic_score
n_token_activations
n_texts
token_frequency
p99_activation
bimodality_score
max_decoder_cosine
max_coactivation_jaccard
max_coactivation_pmi
gca_at_5 / gca_at_10 / gca_at_20
graph_alignment_bucket
pc_norm_mass_top_20
effective_pc_dim
pc_entropy
coverage_bucket
decoder_pc1 / decoder_pc2
decoder_umap_x / decoder_umap_y
bimodal_low_examples_json
bimodal_high_examples_json
top_examples_json
top_decoder_neighbors_json
top_coactivation_neighbors_json
```

## Automated labels

Labels such as `likely_artifact`, `high_frequency`, `rare_high_intensity`,
`bimodal_candidate`, `coactivation_hub`, and `decoder_geometry_dense` are triage
labels. They should guide manual review, not replace it.

## Recommended use

Start with high-priority cards, then inspect:

```bash
uv run sae-atlas inspect-feature --run-name <run> --feature-id <id>
uv run sae-atlas inspect-bimodal-feature --run-name <run> --feature-id <id>
```
