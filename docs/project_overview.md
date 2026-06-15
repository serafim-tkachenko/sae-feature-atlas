# Project overview

SAE Feature Atlas turns a model/layer/corpus choice into reusable interpretability artifacts.

```text
texts
-> model residual activations
-> SAE activations
-> sparse activation table
-> feature statistics
-> feature cards
-> co-activation analysis
-> decoder geometry analysis
-> activation-distribution analysis
-> automated inspection report
-> human follow-up
```

## What each analysis answers

### Feature statistics

Question: **Which SAE features activate on this corpus, how often, and how strongly?**

Outputs:

- `feature_stats.parquet`
- `filtered_features.parquet`
- `feature_cards.parquet`

### Top examples

Question: **What contexts most strongly activate a feature?**

Output:

- `top_feature_examples.parquet`
- feature-card `top_examples_json`

### Co-activation

Question: **Which features tend to appear on the same tokens?**

Output:

- `coactivation_pairs.parquet`

### Decoder geometry

Question: **Which SAE decoder directions are close?**

Output:

- `decoder_neighbors.parquet`

### Geometry vs co-activation

Question: **Are geometrically close features also empirically co-active?**

Output:

- `geometry_vs_coactivation.parquet`

Important disagreement cases:

- high cosine / low co-activation: similar directions, different contexts;
- low cosine / high co-activation: co-active but geometrically distinct, possibly compositional;
- high cosine / high co-activation: potentially redundant or tightly related features;

### Bimodality

Question: **Does a feature have weak and strong activation regimes?**

Output:

- `bimodal_feature_candidates.parquet`

### Automated inspection

Question: **Which candidates are likely artifacts, and which deserve manual inspection?**

Outputs:

- `inspection_feature_summaries.parquet`
- `inspection_pair_summaries.parquet`
- `inspection_report.md`

Automated inspection is heuristic - it does not replace human interpretation
