# Research separation in 0.4.0

The reusable API remains under `sae_feature_atlas`: configuration, runtime, collection, storage, analysis, inspection, pipeline and reporting. Existing CLI preset names remain supported.

Experimental `sae_feature_atlas.scientific` modules moved to `model_behavior_research.scientific` in [Model Behavior Research](https://github.com/serafim-tkachenko/model-behavior-research). Study scripts, protocols, notebooks, tests and report snapshots moved with them. Install the research repository for these workflows; the old experimental namespace is no longer part of toolkit 0.4.0.

The original report revision remains in Git history at `75426d1c6efefe5c5e6d85e689a2b94b38b6b7b7`. The research repository preserves a source snapshot and migration manifest. Historical reports are unchanged.

Generated local data and model caches are not removed. Reuse requires checking original configuration, corpus/model revisions and lineage.
