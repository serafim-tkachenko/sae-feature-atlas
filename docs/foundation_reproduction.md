# Foundation experiment reproduction

This guide documents data layout and reproduction commands for the historical Gemma 3 4B foundation experiment. The frozen designs are [foundation protocol](foundation_protocol.md) and [geometry protocol](foundation_geometry.md); runtime details are in [execution notes](foundation_execution.md).

The [foundation report](../reports/gemma4b_foundation_v1/scientific_report.md) records the measured results. Work in a separate checkout when regenerating historical reports.

## Data layout

Each layer has a directory at `data/processed/gemma4b_foundation_v1_l17` or
`data/processed/gemma4b_foundation_v1_l22`:

- `sae_activations_positive.parquet`: all positive memberships and strengths,
  including collected tokens later excluded by the analysis population.
- `token_metadata.parquet`, `token_activation_summary.parquet`,
  `regime_token_population.parquet`: token identity, corpus split, duplicate group,
  raw support and the exact eligible analysis population.
- `regime_feature_summary.parquet`: frozen discovery fits and candidate selection.
- `source_confirmation/`: source-wise primary tests, sample keys, partner counts,
  null draws, and conjunctions corrected across all selected candidates.
- `geometry/targets.parquet`: frozen primary, random-quartile and reference samples.
- `geometry/residual_chunks/`: original bfloat16 residual bit patterns in uint16
  arrays, with token-key tables and per-chunk SHA-256 checksums.
- `geometry/residual_pca.npz`: full discovery covariance eigenvalues, eigenvectors
  and mean; basis vectors are columns, in descending variance order.
- `geometry/decoder_metrics.parquet`: all 65,536 decoder directions in raw, PCA
  and three regularized-whitening metrics.
- `geometry/screened_pair_geometry.parquet`: every pair in the 2,048-feature
  screen, with exact eligible-token and document intersections and Jaccard values.
- `geometry/orthogonal_*`: the prespecified secondary native-space endpoint.
- `geometry/encoder_control_*`: the separately labelled exploratory diagnostic.
- `geometry/pc_reconstruction.parquet`: evaluation variance and reconstruction
  error along each discovery PC.
- `geometry/context_examples_blinded.csv` and `context_examples_key.csv`:
  future annotation material. Give reviewers only the blinded sheet when blinded
  review is intended. No human validation is implied by the existing artifacts.

The corpus is retained at `data/raw/foundation_v1`, with source revisions, exact
document identifiers, text hashes and duplicate audit. Large data remain outside
Git. Download collection and replay archives from the [artifact inventory](https://github.com/serafim-tkachenko/model-behavior-research/blob/main/artifacts/README.md).

Extract each `foundation_layer*_collection.zip` archive into its corresponding
`data/processed/gemma4b_foundation_v1_l17` or `_l22` directory. Then extract
`foundation_native_residuals.zip`, `foundation_analysis.zip`, and finally
`foundation_final_delta.zip` into the repository root in that order. The final
delta contains completed control outputs and the last descriptive checks.
`foundation_runtime_sources.zip` preserves the exact execution sources and job
logs separately from subsequently formatted repository code. Use the recorded archive checksums to verify each download.

## Execution and reproduction

Use the pinned experiment environment in `requirements/colab.lock`. Original
collection used an A100 with native HF bfloat16 model inference and float32 SAE
encoding. No transformed TransformerLens residual coordinates were substituted.

For a new compatible workspace with the fixed corpus and collected layers:

```bash
python -m sae_feature_atlas.scientific.foundation --layer 17 --stage analyze
python -m sae_feature_atlas.scientific.foundation --layer 22 --stage analyze
python -m sae_feature_atlas.scientific.foundation --layer 17 --stage controls
python -m sae_feature_atlas.scientific.foundation --layer 22 --stage controls
python -m sae_feature_atlas.scientific.geometry_plan --layer 17
python -m sae_feature_atlas.scientific.geometry_plan --layer 22
python -m sae_feature_atlas.scientific.geometry_primary --layer 17
python -m sae_feature_atlas.scientific.geometry_primary --layer 22
python -m sae_feature_atlas.scientific.geometry_replay
python -m sae_feature_atlas.scientific.geometry_analysis --layer 17
python -m sae_feature_atlas.scientific.geometry_analysis --layer 22
python -m sae_feature_atlas.scientific.geometry_coverage --layer 17
python -m sae_feature_atlas.scientific.geometry_coverage --layer 22
python -m sae_feature_atlas.scientific.geometry_examples --layer 17
python -m sae_feature_atlas.scientific.geometry_examples --layer 22
python -m sae_feature_atlas.scientific.geometry_audit --layer 17
python -m sae_feature_atlas.scientific.geometry_audit --layer 22
python -m sae_feature_atlas.scientific.geometry_consistency --layer 17
python -m sae_feature_atlas.scientific.geometry_consistency --layer 22
```

The exploratory encoder diagnostic is invoked separately with
`python -m sae_feature_atlas.scientific.geometry_encoder_control --layer 17`
(and layer 22). The primary endpoint can run as soon as discovery fits exist;
it does not depend on the pooled secondary permutation jobs finishing.

`foundation_resume` retains completed pooled checkpoints and frozen fits, uses
grouped uniform permutations for remaining comparisons, then evaluates the
prespecified weak-separation control arm. It was used to complete slow pooled
jobs without changing the primary source tests. Checkpoint and replay provenance
must match; the tools intentionally refuse silently changed frozen inputs.

To verify the downloaded artifacts and regenerate the report:

```bash
python scripts/verify_foundation_artifacts.py
MPLBACKEND=Agg python scripts/foundation_report.py
python scripts/render_scientific_pdf.py reports/gemma4b_foundation_v1/scientific_report.md
```

The PDF renderer additionally needs ReportLab and a serif font. Rendering/QA
dependencies are separate from the model experiment environment. Scientific
figures are exported as PNG and SVG; no generated illustrative image is used as
experimental data. The schematic and isotropic reference simulation are explicitly labelled.
