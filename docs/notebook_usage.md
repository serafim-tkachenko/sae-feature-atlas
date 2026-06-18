# Notebook usage

Use `AtlasRun` to load generated artifacts in notebooks:

```python
from sae_feature_atlas.storage.run import AtlasRun

run = AtlasRun.from_dir("data/processed/<run_name>")
cards = run.feature_cards()
coactivation = run.coactivation_pairs()
bimodal = run.bimodal_candidates()
coverage = run.coverage_profiles()
```

Useful first checks:

1. Preview `run.artifact_status()`.
2. Sort feature cards by `manual_priority`, `artifact_score`, or `bimodality_score`.
3. Inspect top examples for a feature with `run.feature_examples(feature_id)`.
4. Compare decoder-neighbor and coactivation-neighbor behavior.
5. Check coverage buckets and graph-alignment buckets.
