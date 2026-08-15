# Notebook usage

```python
from sae_feature_atlas.storage.run import AtlasRun

run = AtlasRun.from_dir("data/processed/<run_name>")
stored = run.sae_activations()
analysis_features = run.analysis_features()
cards = run.feature_cards()
coactivation = run.coactivation_pairs()
evaluated = run.bimodality_evaluated()
candidates = run.bimodal_candidates()
pc_alignment = run.decoder_residual_pc_alignment()
```

Start with `run.artifact_status()`. Interpret stored activations according to the collection mode, use analysis features/contexts for human-facing inspection, and treat PCA/UMAP/alignment outputs as diagnostics. The legacy `run.coverage_profiles()` method is an explicit compatibility alias for `run.decoder_residual_pc_alignment()`.
