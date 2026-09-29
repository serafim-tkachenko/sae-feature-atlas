# Reporting

The `report` step writes an interactive `index.html`, `summary.md`, static
`diagnostics.html`, plots, and table previews under `reports/<run_name>/`. The
pipeline also writes a manifest. The browser explorer is built with React and
TypeScript; all scripts, styles, and report data are embedded in the HTML.

## Open a saved run

```bash
uv run sae-atlas report --run-dir data/processed/<run_name>
# Or choose a portable HTML destination:
uv run sae-atlas report --run-dir data/processed/<run_name> --output reports/exploration.html
```

Open the resulting HTML directly in a modern browser. No server, network,
Node installation, tokenizer, or model weights are required to view it. The
directory-based command reads saved metadata and writes only the explorer;
it does not regenerate analyses, Markdown summaries, or static plots. Existing
summary and diagnostics files in the output directory are linked from Run details.

## A single-run research workflow

1. Check the model, SAE, corpus, storage mode, and provenance under **Run details**.
2. Enter an exact numeric feature ID, or search triage labels and exported
   strongest contexts. Filter by support and frequency, or drag a region of the
   frequency/p99 scatterplot to narrow the list. **Random feature** samples
   uniformly from the currently matching list.
3. Select a feature. Read highlighted target tokens in context, with activation
   values, source/text/position references, and explicit frequency denominators.
4. Compare its saved low/high regime examples and stored activation histogram.
   Follow coactivation or decoder neighbors by clicking their feature IDs.
   Neighbors outside the exported feature population are identified explicitly.
5. **Save feature** adds it to a local selection. **Saved** filters that selection;
   **Export selection** downloads JSON containing the selected feature evidence,
   run identity, fingerprints, artifact statuses, and export policy.

Feature links use `#feature=<id>` and survive browser reload/back navigation.
Opening a feature link collapses **Search, filters & landscape** to bring the
evidence forward; reopen it to search or adjust filters. **Previous/Next** in
the evidence panel follows the current filtered sort order. Section buttons jump
to contexts, the distribution, or related features and move keyboard focus there.
On small screens, selecting a feature scrolls to its evidence.

Feature rows show an example target token when available. **Snippet** shortens
saved left/right context and collapses line breaks; **Expanded** shows all of
that saved context. Activation bars compare each example with the strongest
value in its current group. Their scale is local to the group, not a percentile
or a confidence score. The underlying evidence is unchanged by display mode.

When sharing a local link, send the HTML too. Bookmarks stay in the current
browser when local storage is available; export JSON for a portable record.
Reports and selection exports contain corpus excerpts.

### Guidance while exploring

**Quick guide** introduces the find → inspect → compare → save workflow. The
**?** buttons beside metrics, charts, context groups, neighbors, and provenance
explain what each view measures, how to use it, and its limitations. Help opens
with a click, tap, or keyboard activation; Escape closes it and returns focus.
Legacy frequency has its own explanation so an unknown denominator is not
confused with current eligible-token frequency.

Inline hints explain the target-token highlight, support versus text count,
chart colors, and search scope. The highlight identifies the saved target
position; its color does not encode activation magnitude. Triage labels include
a suggested inspection step, and Run details describes each artifact's purpose.

The result count and removable filter chips show all active constraints,
including a chart selection or the saved-only view. Numeric filters reject
negative/fractional support and frequencies outside 0–100 or a reversed range.
An invalid support filter is ignored until corrected; an invalid frequency
range suspends both frequency bounds. Other valid filters still apply, and the
error stays visible. Reset filters clears all constraints.

See [Report UX design notes](report-ux.md) for the reference interfaces and
reasoning behind these choices.

## Evidence and population boundaries

- The explorer includes every saved feature card, with at most **8 contexts per
  example group** and **10 neighbors per relationship type**. It falls back to
  analysis features, then feature statistics, when cards are unavailable, and
  identifies that fallback in the report.
- Strongest contexts are the highest-activation saved examples. They are not a
  representative or held-out sample. Low/high contexts are only available when
  corresponding regime examples were saved. The viewer does not synthesize
  new examples or semantic descriptions.
- Histograms use **finite stored activation rows before analysis filtering**,
  in 24 linear bins. Export scans activations in batches; raw sparse rows are
  not embedded in the HTML. Histogram counts can therefore differ from the
  feature's analysis support. Top-k omissions are not observed zeros.
- Search covers IDs, triage labels, and exported strongest contexts, not the
  complete corpus. A chart selection and numeric filters affect the list;
  a previously selected feature remains inspectable even if filtered out.
- Coactivation and decoder neighbors remain separate evidence channels. A
  missing pair is not an observed zero; an empty context group does not mean a
  feature never activates. Automated labels and scores are inspection cues.

The presentation export uses `schema_version: 1`, independently of collection
artifact schema versions. Its top-level fields are `run`, `features`, `artifacts`,
`warnings`, `export_policy`, `coactivation_metadata`, and optional report links.
Each feature contains population-labeled summary values, context evidence IDs,
bounded neighbor lists, diagnostics, and an optional stored-activation histogram.
Unknown or non-finite numeric values remain JSON null, distinct from zero.

## Partial and legacy runs

Run details distinguishes **ready**, **missing**, **empty**, and **unreadable**
artifacts. Legacy runs or runs without usable lineage are visibly unverified.
Legacy frequency aliases are not relabeled as current analysis frequencies;
the old `semantic_score` is not presented as semantic confidence. Conflicting
card/run identities are rejected instead of silently combined.

Recorded provenance does not independently establish freshness of every file.
Use the pipeline's lineage checks when regenerating analysis artifacts. Viewing
a legacy run does not migrate it or make it compatible with current analyses.

Reports distinguish core empirical analyses from diagnostic/exploratory views. They state top-k storage semantics, population denominators, coactivation support sensitivity, GMM limitations, and the non-semantic status of triage labels and PCA/UMAP.

`manifest.json` includes the artifact schema, collection and analysis fingerprints, Git provenance, resolved configuration, population definitions, and artifact paths. `lineage.json` lives with processed data. Previously generated derived artifacts must not be mixed with new outputs merely because their files exist.
