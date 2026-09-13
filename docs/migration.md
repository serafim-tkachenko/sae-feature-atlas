# Compatibility notes for 0.4.0

The reusable API remains under `sae_feature_atlas`: configuration, runtime, collection, storage, analysis, inspection, pipeline and reporting. Existing CLI preset names remain supported.

The experimental `sae_feature_atlas.scientific` namespace is no longer included. Users of those modules should consult the [separate package migration guide](https://github.com/serafim-tkachenko/model-behavior-research/blob/main/docs/migration.md).

Existing saved artifacts still require compatible configuration, corpus/model revisions and lineage. Regenerate incompatible derived outputs before reuse.
