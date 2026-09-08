# Phase 1 report and Phase 2 handoff

The canonical colleague-facing report is `reports/research_report/report.md`, with its matching `report.pdf`. It combines the original foundation with the completed intervention pilot and two post-result analyses. Prior reports remain historical snapshots. Do not regenerate an older snapshot and present it as the new unified report.

## Reproduce without model inference

From the repository root in the existing environment:

```bash
.venv/bin/python scripts/verify_intervention_pilot.py --bundle outputs/context_pilot_v3 --results outputs/context_download_v3/full_v3 --out reports/research_report/raw_verification.json
.venv/bin/python scripts/analyze_signed_interactions.py
.venv/bin/python scripts/build_research_report.py
python scripts/render_scientific_pdf.py reports/research_report/report.md
```

The renderer needs ReportLab and a suitable serif font. The local bundled Windows runtime supplies these; Linux reproduction can install ReportLab. Render the final PDF to images and inspect every page before distributing a changed PDF. `build_research_report.py` is the editorial source and recomputes displayed tables from CSVs. The Markdown is generated, so persistent editorial changes belong in that builder.

The signed analyzer reads the verified 96 JSON prompt outputs, not the 4B model. It reconstructs parity components from the four cells, checks the exact energy identity, assesses two-length consistency, fits signed-response means and adds a PC-only comparator to the original decoder-effect regressions. All bootstrap intervals are descriptive and conditional on fitted calibration. Its analysis plan is `docs/signed_interaction_analysis.md`.

Large raw archives are intentionally outside Git. Their names and integrity records are in the existing intervention handoff and manifests. The executed full pilot is `outputs/context_download_v3/full_v3`; smoke results are not pooled with it. The Colab runtime was terminated after the full archive was verified. This extension launches no GPU job.

## Findings to preserve

- Foundation: layer 17 has 14/24 partner and 3/24 orthogonal discoveries in separate corrected families; layer 22 has 18/24 and 6/24. They are selected-feature conditional associations.
- Pilot: learned-versus-random raw interaction is most promising for 1645. Beyond-gain and original pooled context prediction do not establish the stronger hypothesis.
- Signed diagnostics: mean odd-odd energy fractions are approximately 90%, 97%, 97%; two-length scaling is fairly stable. This is compatible with generic local curvature, not a novel mechanism.
- Learned-versus-leading-PC selective superiority remains uncertain. No pooled signed-mean transfer interval is entirely positive.
- Incremental prediction: no pooled positive PC-only-minus-PC-plus-context interval. A positive 28027 Wikipedia-to-web comparison does not replicate across calibration settings; both richer models remain worse than the simpler nuisance model in that comparison.

## Phase 2 gates

1. Independently annotate saved contexts and validate behavioral endpoints, including alternatives based on numerical/string position. No semantic labels are established yet.
2. On development prompts, calibrate useful effects and regularization; compare learned context against nuisance, gain and generic-PC information on a common response target.
3. Only after a candidate passes that gate, run limited smaller-dose and pathway ablation/rescue experiments. Save signed intermediate projections, not only norms.
4. Freeze the confirmation protocol before collecting fresh deduplicated prompts; account for calibration uncertainty and multiplicity. Replicate with an independent SAE and subsequently a larger model if scientifically warranted.

All current data remain development material for the stronger claim. Do not relabel existing check groups as fresh confirmation. This is a continuation plan, not a claim that Phase 2 has been executed or preregistered.

The work remains on `research/activation-regimes`. Commit dates are organizational timestamps at the user's request; scientific execution timestamps remain in provenance. No branch push or PR is performed by this report update.
