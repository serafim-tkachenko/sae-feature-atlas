# From observed SAE context to causal changes in feature function

Research decision note — 8 September 2026.

This is a targeted novelty assessment and a proposed follow-up, not a new result
or a preregistered protocol. It does not change the frozen foundation analyses.

## Decision

The strongest next question is:

> Can a context direction learned from ordinary activations predict and control
> which behavior a fixed SAE decoder intervention changes on new prompts, while
> the focal SAE activation stays unchanged?

The scientific contribution would be a bridge from observational geometry to a
transferable causal explanation of feature function. Merely detecting contextual
variation, a low-dimensional cloud of output effects, or an interaction between
two edits would substantially overlap existing work. Even changing effect
strength may be explained by generic gain or normalization. A more informative
result would explain a reproducible change in the relative effects on specified
behaviors, and identify a downstream computation responsible for that change.

This is the best candidate supported by our present leads, not established
novelty. The closest methods reviewed below do not establish this complete joint
claim. That is a scoped comparison, not a guarantee of global priority or eventual
publication. Several close papers are recent preprints. The report's related-work
section needs updating before it is presented as an account of this proposed
contribution.

## What the nearest work already covers

| Primary source | Relevant overlap | Consequence for our claim |
| --- | --- | --- |
| [Duan, *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* (2026)](https://arxiv.org/html/2606.08365v1) | Predicts aggregate feature steering stability and collateral effects from decoder, activation, coactivation and logit statistics; evaluates feature ranking on fresh contexts and a dictionary-width comparison. | Predictability from SAE statistics is already covered. Our target would be variation **within a feature across contexts**, followed by a causal test of the context coordinates. Frequency-only baselines would be inadequate. |
| [Hoang et al., *Sparse Autoencoders Encode Both Concepts and Functions: The Downstream Geometry of Feature Effects* (2026)](https://arxiv.org/html/2607.24645v1) | FEGA analyzes context-dependent clouds of feature-ablation logit effects, including their direction, magnitude and dimensionality. Its main effect geometry uses a reconstruction-relative baseline. | Another effect PCA, or showing that a feature has several output directions, is insufficient. We need input context coordinates learned from ordinary activations that prospectively explain and control those effects. |
| [Gao et al., *The Cylindrical Representation Hypothesis for Language Model Steering* (2026)](https://arxiv.org/html/2605.01844v1) | Studies normal-plane phase and steering sensitivity, including interventions; derives geometry using target-output optimization of sample-specific steering vectors. | Orthogonal context affecting steering is already covered. Our candidate direction must be shared across prompts and learned without optimizing against each test prompt's desired output. Its limited-observable non-predictability theorem is not refuted by using full hidden states. |
| [Vaidyanathan et al., *The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching* (2026)](https://arxiv.org/html/2606.27510v1) | Formalizes interactions between a component and its bypass, relates them to downstream curvature, and studies context-specific circuit roles. | Neither the four-cell interaction statistic nor its Hessian approximation is new. A useful contribution must discover and validate a particular transferable mechanism, rather than rediscover interaction. |
| [Li, *Activation-Space Order-Swap Geometry: A Site-Asymmetry Audit* (2026)](https://arxiv.org/html/2608.25315v1) | Separates site-dependent first-order changes and self-curvature from mixed interaction; includes generic-interaction controls. | Use simultaneous edits at the same hook and token. A nonzero interaction is not sufficient evidence of a distinctive learned mechanism. |
| [Lee et al., *Programming Refusal with Conditional Activation Steering* (2024/2025)](https://arxiv.org/html/2409.05907v3) | Uses activation-based conditions to decide when to apply a behavior vector. | Context-aware steering policies are prior work. Our question concerns how the unchanged model responds to the **same** decoder edit under different internal contexts. |
| [You et al., *Spherical Steering: Geometry-Aware Activation Rotation for Language Models* (2026)](https://arxiv.org/html/2602.08169v2) | Uses norm-preserving rotations and confidence-dependent steering. | Rotation and norm preservation are controls here, not proposed algorithmic novelty. |
| [Leask et al., *Sparse Autoencoders Do Not Find Canonical Units of Analysis* (2025)](https://arxiv.org/html/2502.04878v1) | Examines feature instability across SAE training choices and dictionary sizes. | Replication should compare functional effects and contextual subspaces, without treating feature identifiers as canonical concepts. |

Decoder geometry and cooccurrence are also established subjects; see
[Li et al., *The Geometry of Concepts: Sparse Autoencoder Feature Structure*](https://arxiv.org/abs/2410.19750).
The foundation atlas remains useful as data and candidate discovery, without
requiring those descriptive measurements to constitute the novelty.

## What our existing results provide

The [foundation report](../reports/gemma4b_foundation_v1/scientific_report.md)
contains Gemma 3 4B measurements from 12,000 FineWeb-Edu and Wikipedia documents,
with 4,047,982 eligible token positions per layer. Layer 17 has 14 of 24 selected
features passing the partner endpoint and 3 of 24 passing the separately corrected
orthogonal-displacement endpoint; layer 22 has 18 and 6, respectively. These are
observational findings in selected candidates, not population prevalence or
causal evidence.

Layer-17 features **1645 and 28027** are practical leads: both pass the native
orthogonal endpoint and survive the exploratory removal of the decoder and
covariance-times-encoder span. The latter analysis was added after seeing the
results. It neither removes every linear explanation nor supplies a prospective
test. Feature 863 is a borderline sensitivity case. Sampled feature-1645 excerpts
include digits in year-like strings, making formatting and position within a
number necessary controls rather than a semantic interpretation to assume.

The supported weak-separation controls also exhibit contextual change. We cannot
claim that bimodal activation distributions uniquely identify this phenomenon.
Use mixture selection as one candidate source, alongside supported controls and
a reproducibly sampled comparison panel.

Saved texts, token keys, SAE activations, decoder/encoder parameters and sampled
native residuals can support direction discovery, nuisance analysis and endpoint
design. Native residuals are cached for a fixed subset of positions, not every
collected token. A single saved token vector is insufficient to execute a
transformer tail: the prefix and other required states must be recomputed or
cached consistently for intervention runs.

**All existing data are development data for this new hypothesis**, including the
old evaluation split, because they informed the hypothesis and candidate choice.
They cannot retrospectively become its untouched confirmation set.

## A precise test

### Preserve the measured feature, then vary its context

Let \(h\in\mathbb R^D\) be the native residual at one fixed token and hook,
\(u_j=d_j/\|d_j\|\) the unit decoder direction, and

\[
a_j(h)=\operatorname{JumpReLU}_{\tau_j}(w_j^\top h+b_j)
\]

the focal SAE activation. The collected SAE uses an affine encoder without input
normalization. Therefore \(w_j^\top\delta=0\) preserves its preactivation and
activation, subject to numerical tolerance. Decoder orthogonality alone does not.
Preserving one SAE coordinate does not preserve every other feature or establish
that an intervention lies on the natural activation manifold.

Estimate a shared context direction \(c_j\) from the existing discovery contexts,
initially using a nuisance-adjusted high-minus-low residual contrast. Fit nuisance
adjustments using development data only. Project away \(w_j\) and \(u_j\), and
freeze the resulting direction before confirmation. A failure of this projection
to retain appreciable signal is an informative failure, not a reason to change
the constraints after viewing test effects.

For a stronger norm control, let \(Q\) project onto
\(\operatorname{span}(w_j,u_j)^\perp\), set \(r=Qh\), and define

\[
v=\frac{Qc_j-r(r^\top Qc_j)/\|r\|^2}
        {\|Qc_j-r(r^\top Qc_j)/\|r\|^2\|},\qquad
h_\theta=(I-Q)h+\cos\theta\,r+\sin\theta\,\|r\|v.
\]

For nondegenerate cases this preserves the encoder score, decoder coordinate and
total norm exactly in real arithmetic. Define \(\delta=h_\theta-h\) **once from
the baseline state**. The context direction is shared across prompts; this
constraint-preserving realization depends on each prompt's baseline state.
Freeze a degeneracy rule and match perturbation lengths across direction controls.

### Measure a factorial effect

Let \(F_s\) be the remaining model computation for prefix \(s\), with all other
states held fixed across the four conditions. Use the same additive decoder dose
\(\alpha u_j\) throughout:

\[
I_j(s)=Y(F_s(h+\delta+\alpha u_j))-Y(F_s(h+\delta))
       -Y(F_s(h+\alpha u_j))+Y(F_s(h)).
\]

Patch native activations at the same hook and position in all arms. Do not replace
the native state with a full SAE reconstruction. Do not recompute the context
patch after adding the decoder edit: that changes the intervention rule.

Use a predefined logit contrast as a primary scalar readout. A contrast of two
token logits equals their log-probability ratio; vocabulary-set probability ratios
are different, nonlinear quantities. Audit any final softcap separately. Generated
behavior can be a secondary validation, rather than the only noisy endpoint.

This interaction vanishes for a linear downstream map and linear readout, and
locally takes the familiar form
\(I_j\approx\alpha\,\delta^\top\nabla^2(Y\circ F_s)(h)u_j\).
That identity motivates controls; it is not our theoretical contribution.

### Distinguish a change in function from a change in gain

To support the stronger claim, preregister at least two meaningful output contrasts
after reviewing development examples. Denote their decoder-intervention effects
by a vector \(e_j(s)\). Fit a context-dependent **scalar-gain** model as a strong
null, and compare it with a model that can predict changes in relative output
effects. Do not label an arbitrary pair of vocabulary directions as two behaviors.

A particularly clear result would predict which of two specified effects becomes
dominant, then change that balance by moving context while preserving the focal
SAE score. A sign reversal is compelling if present, but must not be demanded or
selected after the fact. Exclude near-zero effect vectors from angular summaries
using a development-set threshold, and also report unnormalized errors.

For one scalar outcome, gain and functional reorientation cannot be separated in
general. If only scalar modulation survives, narrow the claim accordingly.

### Why exact norm preservation is still insufficient

Consider a normalization-only surrogate
\(Y(h)=a^\top h/R(h)\), where
\(R(h)=\sqrt{\|h\|^2/D+\epsilon}\). The constrained context patch above gives
equal denominators between context arms both before and after the decoder edit.
Nevertheless its four-cell interaction is

\[
I_{\rm norm}=(a^\top\delta)\left(\frac{1}{R_\alpha}-\frac{1}{R_0}\right),
\quad R_0=R(h),\quad R_\alpha=R(h+\alpha u_j),
\]

which need not vanish. Thus even an exactly norm-preserving context edit can
produce apparent modulation through ordinary normalization. This algebra is a
measurement check, not a new representation theory. Compare normalization-only
surrogates and readouts before final normalization, and locate the interaction
inside the actual model. Subtracting one toy prediction does not remove every
normalization effect from a transformer.

## Smallest useful follow-up

1. **Develop and freeze the candidates.** Start with the two layer-17 leads and a
   small supported comparison panel selected without fresh intervention outcomes.
   Review blinded context examples, settle behavior contrasts, and test numerical
   formatting explanations. Estimate one context direction per feature first.
   Any expansion to two to four directions requires development-only selection.

2. **Run a bounded intervention pilot on the existing 4B model.** Use a separate
   development prompt set to choose safe, measurable positive and negative doses,
   fit any response calibration, estimate uncertainty and set a minimum useful
   effect. Context-direction discovery must remain independent of intervention
   outputs if that is the claimed advantage. Output-based sign calibration is
   allowed, but would rule out calling the final predictor entirely zero-shot.

3. **Freeze a new confirmation protocol before fresh outputs.** Specify features,
   exclusions, doses, contrasts, prediction models, direction controls, sample
   size, multiplicity correction and success criteria. Collect new documents
   deduplicated against the original 12,000; hold out prompt templates as well.
   Test cross-source transfer with the same learned direction. Use paired,
   document-clustered uncertainty. Evidence in both sources must have the
   preregistered direction; taking the maximum of two two-sided p-values alone
   would not ensure directional agreement.

4. **Test prospective prediction and causal change together.** Context coordinates
   must improve held-out prediction beyond activation, token identity, position,
   sparsity, residual norm, source and relevant lexical/format controls. Include
   feature-level geometry/coactivation predictors, a scalar-gain model and a richer
   context model. Call a small subspace sufficient only if it retains useful
   predictive performance relative to richer alternatives. Its chosen dimension
   alone is not evidence that the underlying mechanism is low-dimensional.

5. **Compare with matched perturbations.** Random directions must satisfy the same
   encoder/decoder constraints and have identical perturbation lengths and
   calibration budgets. Add covariance-profile-matched and leading-PC controls,
   plus shuffled discovery labels. Monitor activation-neighborhood distance and
   ordinary model performance. Use several small doses, both signs, numerical
   precision checks, linear and normalization-only surrogates. A full local
   Jacobian is a useful expensive reference for response prediction, not a fair
   cheap-input baseline that the method must somehow outperform infinitesimally.

6. **Explain the positive case before scaling it.** Locate where the change in
   effect develops, including before final normalization. Identify a candidate
   attention or MLP computation; test a targeted clamp/ablation and a rescue,
   alongside matched controls. Check that the manipulation does not simply erase
   the main feature effect or broadly damage the model. Circuit-search data are
   development data too: lock the proposed mechanism and validate its intervention
   on another untouched set. Then replicate with another SAE at the same hook;
   a second model is required for a broad cross-model claim.

The first two steps determine whether a larger run is justified. Increasing model
size before checking these controls would not fix the novelty problem. The two
existing layers provide depth sensitivity, not independent model replication.

## Decision rules and eventual figures

| Outcome | Scientific interpretation |
| --- | --- |
| Shared observational direction predicts new-context effects, causally changes their relative behavior beyond controls, and a specified downstream pathway explains the change | A credible candidate contribution: an empirically grounded bridge from SAE context geometry to a transferable mechanism. Scope the claim to tested features, tasks and dictionaries. |
| Prediction works, but the context patch has no selective causal effect | A predictive result only; compare carefully with existing steering forecasts. |
| Only effect magnitude changes, or a normalization/gain control explains the result | Do not claim a change in feature function. Report the narrower modulation or measurement finding. |
| Matched random directions work equally well | Evidence of generic sensitivity, not a special causal role for the discovered context direction. |
| Success requires optimizing a fresh vector against every test prompt | Does not establish the proposed transfer claim; overlaps sample-specific steering work. |
| Only an output-effect PCA or more observational associations succeed | Useful extension of the foundation, but not this novelty claim. |

A successful follow-up report should center its graphics on the evidence chain:
the fixed feature and context constraints; held-out observed-versus-predicted
effects; the four intervention cells with paired uncertainty; changes in the
balance of predefined behaviors; learned versus matched-control dose responses;
normalization diagnostics; pathway ablation/rescue; and dictionary replication.
Extra decorative plots cannot replace a missing link in that chain.

The immediate handoff is to turn this note into a small pilot specification at
ordinary reasoning effort. The existing Markdown/PDF remains a foundation report.
Its related work should be updated together in the next report revision; no new
causal results should appear there until the proposed tests have actually run.
