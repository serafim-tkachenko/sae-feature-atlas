# Changelog

## Unreleased

- Add a portable React/TypeScript feature explorer with search, numeric and chart filters, highlighted contexts, stored-activation histograms, linked neighbors, bookmarks, and evidence export.
- Add `report --run-dir <directory> [--output report.html]` to render saved, partial, and legacy runs without model or plotting imports. Preserve legacy population labels and expose artifact availability and provenance.
- Make the explorer the report landing page; retain static diagnostics as `diagnostics.html` alongside the Markdown summary, plots, and table previews.
- Add `sae-atlas inspect-run <run_dir>` for read-only artifact inventory and optional feature cards and examples, without importing the model runtime.

## 0.4.0 — 2026-09-13

- Remove study-specific modules, execution notebooks and report snapshots from the toolkit distribution.
- Retain the reusable API, CLI presets and population/lineage contract.
- Add explicit build metadata, CI and toolkit-focused documentation.
- Breaking change: the experimental scientific namespace is no longer included; see the compatibility notes.
