# Experiment handoff

The scientific paper is `scientific_report.pdf`, with editable Markdown and
publication-quality figures alongside it. The experiment was actually run on a
local RTX 3080 Ti, not only prepared for Colab.

The main report has been rewritten for the research group, following the
motivation-method-results-interpretation structure of the author's
`qm9-egnn-tda` project. It is a six-page narrative with two focused figures.
Detailed parameter settings, software provenance and implementation discussion
are retained separately in `technical_appendix.md` and the manifest.

## Result and limits

1,000 documents; 234,018 collected tokens; 181,133 eligible tokens; 11,640,733
eligible positive activation rows. The model is Gemma 3 1B PT, with a Gemma Scope 2
layer-13 residual SAE of width 16,384 and target L0=60. No top-k collection cap.

22/24 selected features passed BY q<=0.05 under the primary conditional null.
The one-observation-per-document sensitivity retained sufficient support for
20/24 features, and 5/24 passed BY correction. Strict weak-BIC matching found no
controls. Six exploratory weak-separation control pairs were evaluable, and only
3/6 candidates exceeded their controls in raw JS. This is evidence for conditional
relational change, not proof of semantic polysemy or mixture-specific effects.

## Exact commands and defaults

```bash
uv sync --frozen
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m sae_feature_atlas.scientific.run --stage all --run-name gemma1b_regimes_replication
```

Default: 1,000 Pile-10k documents x <=256 tokens; collection seed 42; half the
documents for discovery; analysis seed 20260907; 512 discovery-screened features;
up to 24 candidates; GMM five starts, delta BIC>=10, weights>=0.1, separation>=2;
posterior>=0.9; each regime >=60 observations in >=20 documents; pooled partner
support>=10; 1,999 permutations per null; 300 document bootstraps. The full policy
and larger Colab configuration are in `docs/scientific_experiment.md`.

Use `notebooks/gemma_activation_regimes.ipynb` in Colab. Until the branch is pushed,
upload `outputs/sae_feature_atlas_source.zip` at the notebook's setup prompt. Select
an L4/A100 GPU runtime, provide an approved Hugging Face token if prompted, and
grant Drive access only if persistence is enabled. The larger explicit setting
uses 5,000 documents x512 tokens and requires high host RAM. No PR/push was made.

## Artifact bundle

`outputs/gemma1b_regimes_positive_analysis_bundle.zip` contains consolidated raw
positive activations, raw texts, token metadata, feature statistics, discovery
fits, posterior assignments, conditional edges, all null replicates, strict and
exploratory controls, one-per-document robustness, contexts, the scientific
manifest, source code, dependency locks and the PDF/Markdown paper.

Core data artifacts under `data/processed/gemma1b_regimes_positive/`:

- `regime_assignments.parquet`
- `regime_feature_summary.parquet`
- `regime_coactivation.parquet`
- `regime_neighborhood_comparison.parquet`
- `regime_null_results.parquet`
- `regime_matched_controls.parquet`
- `regime_context_examples.parquet`
- `scientific_experiment_manifest.json`

The `regime_weak_*` and `regime_one_per_document*` files preserve exploratory arms
separately. The six strict-size exploratory comparisons are in
`regime_control_comparison.parquet`. Annotation sheets are blank and awaiting
human review; no semantic annotation or causal finding is claimed.

## Validation

- 42 tests passed in the original environment.
- 42 tests passed in a clean, hash-locked CUDA-12.6 Colab-compatible environment.
- That clean environment also collected 25 real documents and completed a
  no-candidate analysis/report path, exercising negative-result handling.
- Notebook schema and every code cell compiled successfully.
- Ruff lint passed repository-wide; all new/modified Python and notebook files
  are formatted. The 32 baseline format failures in unrelated files remain.
- The paper was rendered and visually inspected, including tables and figures.
- The Colab frontend and Drive consent interaction were not executed here. The
  actual pinned environment, GPU collection and package stages were tested locally.

## Changed source files

Modified: `.gitignore`, `README.md`.

Created:

- `.gitattributes`
- `src/sae_feature_atlas/scientific/__init__.py`
- `src/sae_feature_atlas/scientific/collect.py`
- `src/sae_feature_atlas/scientific/regimes.py`
- `src/sae_feature_atlas/scientific/controls.py`
- `src/sae_feature_atlas/scientific/robustness.py`
- `src/sae_feature_atlas/scientific/run.py`
- `src/sae_feature_atlas/scientific/report.py`
- `src/sae_feature_atlas/scientific/narrative.py`
- `src/sae_feature_atlas/scientific/export.py`
- `tests/test_scientific_regimes.py`
- `scripts/build_scientific_notebook.py`
- `scripts/colab_stage.py`
- `scripts/render_scientific_pdf.py`
- `notebooks/gemma_activation_regimes.ipynb`
- `requirements/colab.in`
- `requirements/colab.lock`
- `docs/scientific_experiment.md`

Curated report artifacts: the main report in Markdown/PDF, the technical
supplement in Markdown, this handoff, fourteen PNG/PDF figures, and four numeric
CSV result tables. Large raw data remain in the local
analysis bundle rather than Git.

## Commit and PR structure

Branch: `research/activation-regimes`.

Four commits separate collection/provenance, statistical methods/tests,
Colab/reproducible export, and scientific findings/presentation. Commit dates are
distributed between the previous commit and tonight as requested. Experimental
checkpoint timestamps and manifest generation times retain actual provenance.
The subsequent report revision is a separate presentation commit on the same branch.

For the later PR, reviewers should examine the statistical estimands and null
assumptions first, then data lineage and runtime validation, then results and
presentation. The strict-control failure, exploratory amendment, pointwise
uncertainty limitations and dependence sensitivity must remain visible in the
PR description.
