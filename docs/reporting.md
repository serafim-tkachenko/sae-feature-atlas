# Reporting

The `report` step writes `summary.md`, `index.html`, plots, table previews, and a manifest under `reports/<run_name>/`.

Reports distinguish core empirical analyses from diagnostic/exploratory views. They state top-k storage semantics, population denominators, coactivation support sensitivity, GMM limitations, and the non-semantic status of triage labels and PCA/UMAP.

`manifest.json` includes the artifact schema, collection and analysis fingerprints, Git provenance, resolved configuration, population definitions, and artifact paths. `lineage.json` lives with processed data. Previously generated derived artifacts must not be mixed with new outputs merely because their files exist.
