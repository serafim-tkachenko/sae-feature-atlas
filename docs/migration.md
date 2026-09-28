# Compatibility notes for 0.4.0

The reusable API remains under `sae_feature_atlas`: configuration, runtime, collection, storage, analysis, inspection, pipeline and reporting. Existing CLI preset names remain supported.

The experimental `sae_feature_atlas.scientific` namespace is no longer included. Users of those modules should consult the [separate package migration guide](https://github.com/serafim-tkachenko/model-behavior-research/blob/main/docs/migration.md).

Existing saved artifacts still require compatible configuration, corpus/model revisions and lineage. Regenerate incompatible derived outputs before reuse.

## Browser report compatibility

The report landing page `index.html` is now the offline feature explorer. The
previous static HTML layout is generated as `diagnostics.html`; Markdown,
plots, and table previews remain part of the full pipeline report.

`sae-atlas report --run-dir <directory>` can export older or partial runs for
inspection. It does not change the collection artifact schema or existing
sampling policy, and it does not certify or migrate legacy artifacts. Legacy
frequencies retain an unspecified population label when explicit denominators
are absent. Histogram population is always finite stored activation rows,
separate from current analysis support. Report JSON uses its own schema version 1.
