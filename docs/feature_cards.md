# Feature cards

Feature cards are compact per-analysis-feature evidence profiles, not final explanations.

Typical fields include:

```text
feature_id
activation_population / feature_population
stored_activation_count / analysis_activation_count
stored_token_frequency / analysis_token_frequency
analysis_to_stored_support_ratio
primary_label / manual_priority / inspection_labels
artifact_score / interpretability_triage_score
bimodality_score
max_decoder_cosine
max_coactivation_jaccard / max_coactivation_pmi
gca_at_5 / gca_at_10 / gca_at_20
decoder_residual_pc_alignment_bucket
pc_mass_observed / effective_pc_dim / pc_entropy
decoder_pc1 / decoder_pc2
decoder_umap_x / decoder_umap_y
bimodal_low_examples_json / bimodal_high_examples_json
top_examples_json
top_decoder_neighbors_json / top_coactivation_neighbors_json
```

`interpretability_triage_score` rewards non-artifact-like diversity only to prioritize inspection. It is not semantic confidence. Labels such as `likely_artifact`, `high_frequency`, `rare_feature`, `high_intensity`, `bimodal_candidate`, `coactivation_hub`, `decoder_geometry_dense`, and `manual_review` are triage labels.

Use cards to choose evidence to inspect. A future semantic annotation should cite empirical evidence IDs, include counterexamples and failure modes, support abstention, and be evaluated on held-out contexts.
