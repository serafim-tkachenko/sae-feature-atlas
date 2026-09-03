# Native context-intervention pilot

This implements the bounded development experiment proposed in the
[novelty assessment](novelty_assessment.md). It does not implement a fresh
confirmation study or claim a semantic mechanism. All original foundation
documents, including the old evaluation split, are development material here.

## Design and preparation

The configuration is [context_intervention_pilot.json](../experiments/context_intervention_pilot.json).
The initial panel contains layer-17 leads 1645 and 28027, their previously studied
weak-separation comparison feature 7087, and background comparison feature 2966.
Feature 2966 is not a newly matched control for either lead. It was added before
viewing intervention outcomes because 7087 lacked enough matched observations.
Unsupported candidates remain recorded in the preparation manifest.

The preparer checks the original residual-chunk hashes and pinned SAE checkpoint.
It uses only cached native positions in the eligible population. One seeded hash
of each duplicate group assigns all its occurrences to development-fit or
development-check, consistently across sources and features. These are not fresh
confirmation splits. The original foundation PCA basis is reused as a reference.

For each feature, fit-set positive activation quartiles define low and high
amplitude tails. The direction is an overlap-weighted average of within-stratum
high-minus-low residual means. Strata are source, token identity, position bins
of 64 and positive-support bins of 16. At least 20 overlapping observations are
required. The direction is projected away from the encoder and unit decoder;
no intervention outputs enter this fit. Numeric and word-position explanations
remain possible and are explicit downstream diagnostics.

Eight documents per supported feature/source/split are selected, with one token
per duplicate group in that cell. The portable bundle contains their exact token
prefixes, next-token targets where available, development labels and compact
vectors. It contains no model credentials. Preparation refuses to overwrite an
existing bundle.

```bash
OMP_NUM_THREADS=4 python -m sae_feature_atlas.scientific.intervention_prepare \
  --out outputs/context_pilot_v3
python scripts/build_intervention_bundle.py --prepared outputs/context_pilot_v3
```

Local preparation needs the pinned SAE weights and tokenizer cached in Hugging
Face, as well as the foundation data described in [the handoff](foundation_handoff.md).
The first two preparation directories made during implementation were development
iterations; the run bundle is `outputs/context_pilot_v3`.

## Native model execution

The runner loads the revision-pinned Gemma 3 4B text backbone in float32, with
TF32 disabled and eager attention. The foundation residuals were collected in
bfloat16; the precision difference is deliberate and must remain disclosed.
Float32 permits tightly audited, representable context edits. This is not a
claim that the original bfloat16 distribution and new activations are identical.

All four factorial cells recompute the full, identical token prefix with cache
reuse disabled. A hook replaces only the final token at native block 17.
Other positions and all upstream computation remain fixed between arms. The
context patch is calculated once from the baseline and reused in the combined
arm. It preserves encoder score, decoder coordinate and residual norm in exact
arithmetic; the actual float32 cells must satisfy a relative tolerance of 1e-5.
Threshold crossings, nonfinite states, changed inputs or corrupt checkpoints
cause an explicit failure.

Decoder doses are +/-0.1 times the fit-set median feature contribution norm.
Context chord lengths are 0.005 and 0.01 times the baseline residual norm, each
in both directions. Controls include three seeded random directions, a PCA-mass
control that is approximate after constraint projection, the leading PC, and a
within-stratum label-shuffle direction. All use the same chord construction and
length. The minimum design yields 56 factorial comparisons per supported prompt;
shared baseline, decoder and context arms reduce this to 87 forward passes.

The two readouts are mean token-logit contrasts: digits versus punctuation, and
three common word-start tokens versus three suffix tokens. They are explicit
formatting probes, not validated semantic behaviors. The runner verifies each
configured string maps to one token during preparation. For the supported tied
embedding model, mean output weights compute the contrasts without storing the
full vocabulary effect cloud. These are pre-softcap logit contrasts; next-token
loss monitoring separately applies the model's softcap if configured.

Recorded diagnostics include every scalar readout in all four cells, interactions
before final normalization, a normalization-only surrogate, native layerwise
interaction norms, next-token loss, and numerical constraint errors. The layer
trajectory is descriptive localization, not pathway ablation/rescue. Next-token
loss is a local quality measure, not a comprehensive generation evaluation.

```bash
python -m sae_feature_atlas.scientific.intervention_run \
  --bundle outputs/context_pilot_v3 --out outputs/context_smoke --limit 3
python -m sae_feature_atlas.scientific.intervention_run \
  --bundle outputs/context_pilot_v3 --out outputs/context_full
```

An A100 has sufficient memory for float32. `--limit` is a smoke option, sampling
features in round-robin order. Its results must not be pooled with the full run.
Each prompt is saved atomically with a checksum. Resume uses the same command
and output directory; changing the code, plan, runtime settings or limit requires
a different directory. The Colab job script uses the existing locked environment
and consumes/removes a temporary download token without logging it.

The completed run is backed up in `outputs/context_intervention_results.zip`;
extracting it gives `pilot/` and `full_v3/`. The downloaded copy is at
`outputs/context_download_v3`. The archive includes the exact execution bundle
and isolated experiment environment. It was verified locally before terminating
the A100 runtime.

## Analysis and interpretation

```bash
python scripts/verify_intervention_pilot.py \
  --bundle outputs/context_pilot_v3 --results outputs/context_download_v3/full_v3 \
  --out reports/context_intervention_pilot_v1/verification.json
python -m sae_feature_atlas.scientific.intervention_analysis \
  --bundle outputs/context_pilot_v3 --results outputs/context_full \
  --out reports/context_intervention_pilot_v1
python scripts/render_scientific_pdf.py reports/context_intervention_pilot_v1/report.md
```

PDF rendering requires ReportLab and a serif font, separate from the locked model
environment. Delivery used the bundled document runtime and Poppler for visual
inspection of all eight pages. One case has no next-token loss target because its
position is the final token in the fixed collection window; it remains in the
intervention analyses.

The analysis compares each learned-direction result with the mean of its three
paired random-direction controls. Descriptive bootstrap intervals give each
duplicate group equal weight and resample documents. They are not corrected
discovery tests. A scalar-gain residual measures deviation from rescaling the
baseline two-contrast effect vector, excluding squared baseline norms <=1e-8.
This residual still cannot establish a semantic role change.

Prediction checks use ridge regression with alpha=1 and fit-only feature scaling
and categorical encoding. Baselines include token identity, source, encoder score,
norm, token position, support and trailing digit/Latin-letter counts. Comparisons
add the learned context coordinate, eight foundation PCs, or restrict predictions
to a scalar multiple of the training mean effect. Tests include pooled-source
development-check and train-one-source/check-the-other. Intervals condition on
the fitted calibration model. The sample is small, and an added coordinate can
overfit; a positive training fit is not a success criterion.

## Next decision

Use this pilot to choose an interpretable endpoint, estimate a minimally useful
effect and plan sample size. Fresh confirmation needs a new frozen protocol,
deduplication against all 12,000 foundation documents and held-out templates.
The strongest novelty claim additionally requires selective effects beyond
generic gain/normalization, a validated pathway intervention and SAE replication.
If context adds no predictive value or the learned direction does not exceed
matched controls, report that limitation before increasing the experiment size.

## Completed development decision

The 96 feature-prompt cases (95 duplicate groups) yielded 5,376 factorial
comparisons. All numerical audits passed; the largest reported relative error
was 1.40e-6. Sixty repository tests passed after implementation.

Feature 1645 has a positive paired raw-interaction difference against the mean
random control on development-check documents. All pooled prediction-improvement
intervals and all beyond-gain comparison intervals include zero. These are small,
descriptive, uncorrected comparisons. The result supports retaining a modulation
lead, but does not establish the proposed transferable change in feature function.
Do not scale this exact design directly into confirmation on that basis.
