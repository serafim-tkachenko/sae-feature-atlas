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

## Saved evidence and empty analysis results

An explicitly empty feature selection is now authoritative. The exporter no longer
falls back to unselected feature statistics or revives embedded contexts/neighbors
when the corresponding saved table is present but empty. Missing or unreadable
optional tables can still use embedded evidence, with availability reported separately.

Feature generation and saved-evidence reanalysis support runs where no features
pass selection. They produce empty selection/example artifacts and a report explaining
how to check thresholds and sample size. A population with no eligible tokens still
requires a different eligibility policy before statistics can be computed.

Refreshing feature cards replaces previous automatic labels and review priorities.
Missing/nonfinite activation statistics cannot establish high intensity, and empty
diagnostics remove previously derived scores and examples. Regenerate cards to apply
these rules; exporting a saved report does not silently recalculate its labels.

Population construction rejects duplicate token positions and activation rows whose
positions, strings, sources or token IDs disagree with metadata. Correct the source
artifacts or recollect them instead of relying on silent deduplication or dropped rows.

`AtlasRun.sae_activations()` follows the storage mode recorded in `lineage.json`.
Without that identity, one saved format can be inferred; if both exist, pass
`mode="topk"` or `mode="positive"`. A missing recorded file is an error and is never
silently replaced with a different activation population.

Browser bookmarks retain compatibility with previous saved selections. Individual
feature saves now synchronize across tabs without overwriting unrelated selections.
Deep links and previous/next feature navigation reveal the selected row's list page.

Decoder neighbor search excludes self-matches even at cosine -1 and uses all
available slots when the query is outside the candidate set. Geometry/coactivation
merges now orient `p_j_given_i` and `p_i_given_j` to the directed geometry edge;
Jaccard and pair support remain symmetric. Regenerate affected geometry outputs
to correct previously saved directional probabilities or neighbor lists.
