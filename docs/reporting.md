# Reporting

The `report` step writes a markdown summary and an HTML report under:

```text
reports/<run_name>/
```

Main files:

```text
summary.md
index.html
inspection_report.md
inspection_report.json
plots/
tables/
manifest.json
```

The report is organized around the analysis checklist: activation collection,
feature filtering, coactivation, bimodality, geometry/coactivation comparison,
residual PCA coverage, graph alignment, feature cards, and caveats.

Reports are descriptive. They should not be read as causal-control evidence.
