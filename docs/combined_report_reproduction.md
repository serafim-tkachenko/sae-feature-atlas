# Historical combined report reproduction

These commands rebuild `reports/research_report/report.md` and its PDF from saved outputs. That report predates the retrospective constant-baseline comparison; use the [current study](https://github.com/serafim-tkachenko/model-behavior-research/blob/main/reports/sae_context_study/report.md) for the latest conclusion.

Use a separate checkout: the historical builders overwrite report artifacts. Rebuilding them does not provide new inference or an independent dataset.

## Reproduce without model inference

From the repository root in the existing environment:

```bash
.venv/bin/python scripts/verify_intervention_pilot.py --bundle outputs/context_pilot_v3 --results outputs/context_download_v3/full_v3 --out reports/research_report/raw_verification.json
.venv/bin/python scripts/analyze_signed_interactions.py
.venv/bin/python scripts/build_research_report.py
python scripts/render_scientific_pdf.py reports/research_report/report.md
```

The renderer needs ReportLab and a suitable serif font. Render the final PDF to images and inspect every page before distributing a changed PDF. `build_research_report.py` is the editorial source and recomputes displayed tables from CSVs. The Markdown is generated, so persistent editorial changes belong in that builder.

The signed analyzer reads the verified 96 JSON prompt outputs, not the 4B model. It reconstructs parity components from the four cells, checks the exact energy identity, assesses two-length consistency, fits signed-response means and adds a PC-only comparator to the original decoder-effect regressions. All bootstrap intervals are descriptive and conditional on fitted calibration. Its analysis plan is `docs/signed_interaction_analysis.md`.

Large raw archives are intentionally outside Git. Their names and integrity records are in the [artifact inventory](https://github.com/serafim-tkachenko/model-behavior-research/blob/main/artifacts/README.md) and manifests. The executed full pilot is `outputs/context_download_v3/full_v3`; smoke results are not pooled with it. These commands reconstruct saved inference and do not launch a GPU job.
