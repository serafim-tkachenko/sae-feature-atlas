# Activation-conditioned neighborhood experiment

## Scientific audit

The original atlas can describe positive or retained-top-k feature activity,
same-token coactivation, decoder geometry, and the preference for a two-Gaussian
description of log1p positive activation. Its GMM screen is not a test of semantic
polysemy. It uses log1p, despite several older descriptions saying simply “log”.
The older example helper ranks posterior probabilities but does not enforce a
confidence threshold; the new scientific experiment applies an explicit threshold
and preserves ambiguous observations.

The existing coactivation implementation correctly defines its eligible-token
denominator and canonical unordered pairs. Its saved graph is support-filtered
and capped, however, and must not be used as a complete partner-membership matrix.
The new experiment works from raw sparse rows. Stored feature frequency, analysis
frequency, mixture diagnostics, decoder cosine, PCA/UMAP and graph overlaps all
have distinct populations and are descriptive unless paired with an appropriate
inferential design. Geometry is not a causal interaction or semantic equivalence.

The historical run contains only 100 documents, a top-32 collection, and no
current lineage. Its raw collection may also depend on TransformerLens coordinate
processing. It was not migrated or used as evidence. A fresh native Hugging Face
collection uses the published SAE hook and checks every parameter tensor against
a revision-pinned native checkpoint. The existing atlas runtime is unchanged.

## Default real experiment

- google/gemma-3-1b-pt; Gemma Scope 2 resid_post layer 13, width 16,384,
  medium target L0=60; actual native hook model.layers.13.output.
- 1,000 unique Pile-10k documents, seeded selection, maximum 256 tokens each.
- All strictly positive SAE activations, without a rank cap; model bfloat16,
  SAE float32; one document at a time; collection seed 42.
- Half the documents for discovery, half for evaluation; seed 20260907.
- Uniform screen of up to 512 discovery-supported latents (>=200 activations
  in >=30 documents); select up to 24 qualified candidates by discovery delta BIC.
- GMM: five starts, log1p activations, delta BIC>=10, minimum component weight
  0.1, standardized separation>=2; evaluation posterior threshold 0.9.
- Each evaluated regime: >=60 observations in >=20 documents; pooled partner
  support>=10; top-20 descriptive neighbors.
- 1,999 primary document/position/support permutations and 1,999 secondary
  token-identity/position/support permutations; 300 document bootstraps.
- Position bins 64, total-positive-support bins 16; BH and conservative BY
  correction across all selected candidates, including support failures as p=1.
- Strict control: converged delta BIC<10, nearest matching without replacement
  in log activation support and log document support, per-coordinate caliper 0.35.
  Control quantile tails are frozen on discovery documents; evaluation sample
  sizes must match candidate low/high counts exactly.

The first run had zero strict-control matches. A separately exported exploratory
arm screens up to 2,048 additional discovery latents and uses converged mixtures
with separation<2 as controls. This amendment was introduced after the first
analysis; it is not a successful strict-BIC control experiment. The additional
one-observation-per-document sensitivity was also motivated by skeptical review
and is not independent replication. Both are included prospectively in the
notebook for the next run.

## Running

```bash
uv sync --frozen
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python -m sae_feature_atlas.scientific.run --stage all --run-name gemma1b_regimes_replication
```

Individual stages: collect, analyze, controls, robustness, report. The local
default lock uses CUDA 13.0. Colab has a separate fully pinned, hash-checked CUDA
12.6 environment in requirements/colab.lock, validated in an isolated local
environment. Open notebooks/gemma_activation_regimes.ipynb in Colab and Run all.
Until this branch is pushed, setup asks for the delivered source ZIP. Once pushed,
it clones the research branch; the checked-out SHA is recorded. No remote branch
or PR was created during the local work.

The Colab larger configuration explicitly requests 5,000 documents x 512 tokens,
2,048 screened latents, 48 candidates, 9,999 permutations and 1,000 bootstraps.
It requires >=48 GB host RAM and >=12 GB GPU RAM. Both defaults retain all positive
activations. The notebook never silently reduces the sample, sequence length,
precision, support thresholds or repetitions. Wall time depends on GPU and corpus
L0. Drive persistence is optional; enabling it writes checkpoints directly there.

Hugging Face account access to gated Gemma weights is the user's responsibility;
the notebook uses an HF_TOKEN secret or requests a token only when access is
gated. GPU selection and optional Drive consent are the other manual steps.

## Outputs and interpretation

All requested regime tables, complete null replicates, fitted parameters,
assignment posteriors, contexts, raw text/token IDs, exact resource revisions,
source hashes and runtime provenance are saved under the run data directory.
The main report, PDF, figures, CSV tables and blind annotation sheet are under
reports/<run_name>. Strict, exploratory weak-control, and one-per-document
results have separate artifact names. Figures retain PDF vector versions.
The main report follows the narrative style of the author's QM9 geometry/topology
project. The detailed report and provenance discussion are generated separately
as technical_appendix.md, so the group-facing report remains focused on the
question, evidence and interpretation.

The partner estimand is conditional positive membership. JS normalizes partner
occurrence counts to sum one; conditional L1 and signed mass change retain
overall support differences. Event Jaccard/PMI use the complete eligible
evaluation-token denominator. PMI is omitted for insufficient regime-pair
support. Within the pooled partner universe a raw count of zero is observed zero;
outside that universe a missing saved edge is filtered, not imputed zero.

Permutation p-values require conditional exchangeability. Document strata do not
make adjacent tokens exchangeable; the independent-document sensitivity reduces
this issue but cannot remove lexical/corpus confounding. Positive JS bootstrap
bias is exported. Approximate pointwise 95% intervals use observed JS +/-1.96
document-bootstrap standard errors, clipped to [0,1]; raw percentile ranges are
also retained as diagnostics. These intervals are not simultaneous or
selection-adjusted and do not include GMM/corpus-selection uncertainty.

The final report supplies the actual evidence. It must not be read as proof of
discrete contextual identities or semantic polysemy. Publication-quality claims
need independent corpus/model/SAE replication, blinded semantic annotation and
interventions.

## Artifact compatibility and recovery

The original schema-2 lineage is preserved and the fresh collection writes its
own lineage. Regime-specific settings, code and input hashes have a separate
scientific manifest. A regime run must require compatible collection lineage;
it does not accept legacy raw files merely because they exist.

Collection saves hash-verified chunks every 25 documents and marks a chunk
complete only after all three tables are written. Per-feature assignments,
edges, summaries and nulls are checkpointed during analysis. Rerunning analysis
recomputes from intact raw data; per-feature checkpoints are recovery artifacts,
not automatically resumed caches. The supplemental discovery-fit cache validates
its feature set; use a new run name when changing experiment parameters.

No existing atlas tests were removed. Baseline: 34 tests passed, Ruff lint passed;
32 pre-existing files failed Ruff format checks. New files are formatted without
mass-formatting unrelated scientific code. Runtime package versions and generated
reports should be reviewed separately from statistical implementation in a PR.
