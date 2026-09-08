"""Assemble the self-contained group report from verified studies and new diagnostics."""

from pathlib import Path
import hashlib
import json
import re
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research_report"
FOUNDATION = ROOT / "reports/gemma4b_foundation_v1"
PILOT = ROOT / "reports/context_intervention_pilot_v1"
SIGNED = PILOT / "signed_analysis"


INTRO = """# From activation geometry to context-dependent effects

## A study of SAE pseudo-concepts in Gemma 3 4B

Phase 1 research report | FineWeb-Edu and English Wikipedia | September 2026

## Abstract

A sparse autoencoder gives each dictionary entry a fixed decoder direction, but the model states in which that entry activates can differ substantially. We study whether activation-conditioned context can become an explanation of intervention effects, rather than merely a description of feature usage. The observational foundation covers 12,000 documents, two native residual layers and more than four million eligible token positions per layer. At the primary layer, 14 of 24 selected entries pass a held-out two-source partner-composition test and three pass a separately corrected test of displacement orthogonal to the decoder. A subsequent development pilot evaluates 96 feature-prompt cases with encoder-, decoder-coordinate- and norm-preserving context edits. Raw interaction exceeds random-direction controls for one candidate, but prediction improvement and superiority beyond scalar gain remain uncertain. A new analysis of saved signed responses finds predominantly odd-in-both-signs interactions, compatible with local mixed curvature; it does not establish a transferable learned response or superiority over the leading principal-component control. Together, the results distinguish reproducible contextual organization from a stronger causal explanation that has not yet been demonstrated. The report provides quantitative geometry, controlled intervention evidence and a concrete route toward testing whether feature-specific context predicts selective behavioral effects beyond generic activation geometry.

## 1. Research questions and the central idea

The central question is **whether ordinary activation context contains a transferable explanation of what a fixed SAE intervention does**. An SAE decoder describes the additive reconstruction contribution of a dictionary entry. It does not, by itself, specify how the rest of the transformer will transform that contribution at a particular token. We investigate the relationship between those two descriptions.

We call an SAE entry a **pseudo-concept**: a useful learned unit whose atomicity, semantic purity and uniqueness are not assumed. The same decoder direction can be used in different contexts without rotating, and different dictionary entries can share directions without playing identical roles. Activation magnitude can indicate context without constituting a distinct meaning.

| Research question | Measurement | Current answer |
| --- | --- | --- |
| RQ1. How do decoder directions occupy activation space? | Participation, PCA alignment, reconstruction and cooccurrence | Directions are distributed; directional mass and reconstructed variance tell different stories. |
| RQ2. Does feature strength mark context beyond the decoder axis? | Frozen intensity regimes, conditional partner tests, native orthogonal shifts | Supported for a subset of selected entries, with limited specificity and remaining confounds. |
| RQ3. Does this context predict or selectively change a decoder intervention? | Constrained four-cell edits, prediction checks and gain controls | Raw modulation is present; the stronger predictive and selective claim is not established. |
| RQ4. Is the modulation a coherent local response distinct from generic geometry? | Signed parity, length consistency, transfer and PCA controls | Locally structured responses coexist with uncertain transfer and no clear learned-over-PC advantage. |

![Figure 1. The scientific distinction. Amplitude alone moves a contribution along its fixed decoder direction. The contextual hypothesis concerns displacement beyond that direction. These synthetic points illustrate the question and are not experimental observations.](foundation/01_hypothesis.png)

Figure 1. The scientific distinction. Amplitude alone moves a contribution along its fixed decoder direction. The contextual hypothesis concerns displacement beyond that direction. These synthetic points illustrate the question and are not experimental observations.

The proposed stronger research contribution is a linked result: a context coordinate learned from unperturbed activations predicts a feature's selective effect on new prompts; changing that coordinate while holding the focal feature fixed changes the predicted effect; and a downstream pathway explains the change. Each link can fail. The present evidence supports contextual organization and supplies a working causal test, but does not complete this chain.

**Reading guide.** Sections 3-10 establish the observational geometry and its statistical limits. Sections 11-12 describe the completed intervention pilot. Section 13 adds exploratory analyses performed entirely on its saved outputs. Sections 14-16 integrate the evidence and define the next research stage. The two older study reports are incorporated here so this document can be read independently.

## 2. Related work and the intended contribution

Superposition motivates studying more representational features than dimensions [1]. Early SAE work and subsequent scaling studies establish useful sparse dictionaries while emphasizing reconstruction and evaluation [2-4]. Gemma Scope and Gemma Scope 2 provide pretrained dictionaries [5,6]. Decoder neighborhoods, cooccurrence and PCA structure are already studied in The Geometry of Concepts [7]; they are foundations for this project rather than priority claims. Noncanonical dictionary structure motivates our use of pseudo-concepts [8].

The most relevant steering work raises the bar further. The cylindrical representation hypothesis investigates context normal to a steering axis [9]. FEGA analyzes context-dependent clouds of logit effects produced by SAE interventions [15]. Pre-intervention side-effect prediction uses feature statistics to forecast steering modularity across features and model settings [16]. Work on multiple mediators relates activation-patching interactions to downstream curvature [17]. Therefore neither context dependence, output-effect geometry, prediction in general nor a nonzero four-cell interaction is a sufficient novelty claim.

Our proposed distinction is **within-feature, transferable explanation from native context to selective effect**, with encoder and decoder coordinates held fixed during the context intervention, explicit generic-geometry controls, and later pathway validation. This is a research target, not an established priority claim. The closest recent studies are preprints, and a final publication claim would require another focused literature review when the evidence is complete.

| Evidence stage | Data and separation | Interpretation |
| --- | --- | --- |
| Observational foundation | Discovery/evaluation documents; frozen mixtures; two-source tests | Conditional associations under stated exchangeability assumptions. |
| Intervention development | Previously collected documents, split into fit/check groups | Controlled effects of artificial edits on diagnostic logits; no fresh confirmation. |
| Signed secondary analysis | Same saved pilot outputs, analyzed after its initial results | Exploratory diagnosis of sign, local scaling and control specificity. |

The check subset of the pilot is not the foundation's untouched evaluation set repurposed as confirmation. All existing corpus observations are development material for the stronger hypothesis. Additional plots and analyses do not increase the number of independent documents.

"""


EXTENSION = r"""
## 11. From observational context to a controlled intervention

### 11.1 Candidate preparation and observation units

The observational study motivates asking whether a context direction has functional consequences. We selected primary-layer entries 1645 and 28027 as candidate cases, included weak-control entry 7087, and added entry 2966 as a background comparison before examining intervention outputs. Entry 2966 is not a matched causal control for the two candidates. A minimum of 20 within-stratum observations was required to estimate a context contrast; entry 7087 had only 17 and was not run. Its failure remains in the accounting.

| Pseudo-concept | Within-stratum observations | Fraction of contrast retained by constraint projection | Pilot status |
| --- | ---: | ---: | --- |
| 1645 | 1,418 | 0.387 | 32 feature-prompt cases |
| 28027 | 133 | 0.967 | 32 feature-prompt cases |
| 7087 | 17 | Not estimated | Insufficient overlap |
| 2966 | 53 | 0.973 | 32 feature-prompt cases; background comparison |

For each supported entry and source, eight fit and eight check documents were selected. In total, there are 96 feature-prompt cases from 95 distinct duplicate groups. A seeded group split prevents the same duplicate group from entering both fit and check across features. The comparison is intentionally small: each feature has only 16 check groups, and a source-specific check has eight. Many interventions on the same prefix improve within-prefix comparison, not sample size.

The learned direction is a fit-only high-minus-low mean contrast between positive-activation quartiles. Contrasts are calculated within source, token identity, position-bin and positive-support-bin strata, with overlap weights proportional to n_low times n_high divided by their sum. This focuses the estimate on strata containing both regimes. Its projection removes both the focal encoder direction and the unit decoder direction. No intervention outcomes are used to select this direction.

### 11.2 Preserving the focal feature while editing context

Let h be the native residual vector, u the unit decoder direction, w the encoder vector and Q the orthogonal projector onto the complement of their span. Holding only u's coordinate fixed does not hold the feature score fixed, because the encoder need not equal the decoder. We therefore edit in the intersection of the two constraints. A rotation within that subspace also preserves the total residual norm:

![Equation](equations/rotation.png)

Here r is Qh, v is the unit tangent component of the learned context direction after removing r, and beta is the desired chord length. The rotation sign reverses its tangent component. At finite length, the two edits are not exact negatives: both include the same small inward radial component. Degenerate directions fail explicitly. The construction preserves the encoder score, decoder coordinate and input norm in real arithmetic, and all three are checked again on representable float32 states.

The feature score's JumpReLU threshold is unchanged. One selected case is inactive under the float32 encoder despite being positive in the original bfloat16 collection; it is retained transparently. The study does not replace native hidden states with SAE reconstructions.

### 11.3 Four-cell design and output readouts

For each baseline prefix we evaluate the unedited state, decoder-only edit, context-only edit and their combined edit. The context displacement is calculated once at the baseline and reused exactly in both context-containing cells. All four arms recompute the identical full prefix without a key/value cache. The factorial interaction subtracts both individual effects:

![Equation](equations/interaction.png)

Y is a two-dimensional vector of output logit contrasts. A logit is the model's score before conversion into a probability. The first contrast compares the mean scores of digit tokens 0-9 with comma, period and semicolon. The second compares common word-start tokens (' the', ' and', ' of') with suffix tokens ('ing', 'tion', 'ly'). The token sets were resolved to single vocabulary IDs before inference. Scores are measured before any optional logit soft-capping.

These are formatting diagnostics motivated by observed contexts, not validated measures of numerical reasoning, syntax or semantic intent. Averaging token logits is a contrast of scores, not a difference of total probability masses. Both contrasts use their original units; an unweighted Euclidean norm gives them equal numerical weight, not equal scientific importance.

The two decoder doses are plus/minus 0.1 times the fit median positive activation multiplied by the decoder norm. The resulting absolute native doses are about 85.59 for entry 1645, 8.92 for 28027 and 12.97 for 2966. Context chord lengths are 0.5% and 1% of the baseline residual norm, in both signs. Thus raw magnitudes cannot be ranked across features as intrinsic importance: their native doses differ.

Seven context directions are evaluated: learned, three seeded random directions, leading activation PC, a spectrum-matched direction, and a shuffled-regime contrast. All undergo the same focal-coordinate and norm constraints. Spectrum matching is approximate because reprojecting its randomized PC coefficients changes their spectrum. These controls test different alternatives; beating shuffled labels is weaker evidence of feature specificity than beating a strong generic activation direction.

The design yields 56 factorial comparisons and 87 full-prefix forward passes per feature-prompt case: 5,376 comparisons and 8,352 forward passes overall. Reused cells are deliberate and retained in the dependence structure. Inference uses float32 with TF32 disabled, while the observational archive used bfloat16. Maximum audited relative constraint/additivity error is 1.40 times 10^-6, below the specified tolerance of 10^-5. This checks implementation fidelity, not whether the perturbed state is natural.

### 11.4 What counts as more than scalar gain?

Define e0 as the decoder's two-contrast effect at baseline and e1 as its effect after the context edit. A context edit can change their magnitude while retaining the same effect direction. We remove the best scalar multiple of e0 and measure the remaining component:

![Equation](equations/gain.png)

This is a least-squares projection in the two chosen contrast coordinates. A small residual means the change can be described mainly as scaling, including a negative scale; it does not show there is no other behavioral change outside the two readouts. Cases with squared baseline effect at most 10^-8 are omitted only from this unstable projection, while their raw interactions remain. A large residual still requires comparison with generic controls.

### 11.5 Normalization and other generic nonlinearities

Preserving the norm of h does not remove normalization interactions downstream. Even a simple normalized linear readout can have a nonzero factorial interaction. If Y(h) equals a dot h divided by R(h), and R is root mean square with a numerical stabilizer, the same constraints imply:

![Equation](equations/normalization.png)

The decoder-containing pair has a common denominator R_alpha, while the decoder-free pair has R_0. Their difference can multiply the context numerator. This surrogate is a diagnostic, not a complete transformer model and not a valid subtraction that removes all normalization effects. Similarly, a large intermediate residual interaction norm locates accumulation but does not identify an attention head, MLP or causal pathway.

### 11.6 Prediction and uncertainty

The original prediction target is e0: the decoder effect before a context intervention. Ridge regression uses fit-only preprocessing and a fixed regularization coefficient of one. It penalizes large coefficients while predicting the two output contrasts:

![Equation](equations/ridge.png)

The nuisance model includes encoder score, residual norm, position, log positive support, token identity, source and trailing digit/Latin-letter counts. Alternatives add the learned context score, or both that score and eight reference PCs; a scalar-gain alternative constrains output to the training-mean effect direction. The PCs are from the original foundation reference, not a new confirmation sample. Pooled calibration is evaluated on check groups; single-source calibration is evaluated on the other source's check groups.

The intercept is fitted without a penalty. Mean squared error (MSE) is the average squared discrepancy across the two contrasts. Positive nuisance-minus-context MSE indicates improvement. Uncertainty resamples duplicate groups, weights groups equally and keeps the fitted predictor fixed. The 95% percentile intervals use 2,000 seeded resamples, omit calibration-fit uncertainty and are not multiplicity-corrected. They are descriptive diagnostics, unlike the foundation's corrected conditional tests.

## 12. Intervention results: what the pilot supports

### 12.1 Raw interaction is stronger evidence than selective function

{{RAW_TABLE}}

Entry 1645 shows a positive learned-minus-mean-random raw interaction interval on check groups. Its beyond-gain interval includes zero. Neither of the other two entries has a beyond-gain interval entirely above zero. The raw result for 28027 lies very close to an interval boundary; it should not be treated as a robust binary distinction. These estimates describe selected cases and two formatting readouts, not the prevalence of causal context effects in the dictionary.

{{FIG:pilot/06_paired_control_check.png|Paired learned-minus-mean-random effects on development-check groups. Each point averages repeated signs and lengths within groups; intervals are descriptive. The beyond-gain endpoint removes the best scalar multiple of the baseline two-contrast effect.}}

The full direction comparison shows why a random baseline alone is insufficient. Leading-PC interventions can have effects comparable to the learned direction. Independent-looking error bars in the next figure are not a paired superiority test; the explicit paired comparisons appear in Section 13.

{{FIG:pilot/01_direction_controls.png|Interaction norms for all seven context directions, pooled over development cases and doses. Features have different native decoder doses. Separate intervals should not be read as paired tests; this is a descriptive view of the control hierarchy.}}

### 12.2 Propagation and prediction quality

Actual interactions are not exhausted by the illustrated normalization-only surrogate. That fact does not distinguish a feature-specific mechanism from other generic nonlinearities. Layerwise interaction norms accumulate downstream for learned and random edits; they cannot reveal a signed response pathway because only norms were retained at these layers.

{{FIG:pilot/02_normalization_diagnostic.png|Actual interaction norm against the normalization-only surrogate for learned context edits. The surrogate isolates one possible source of interaction; it is not a complete normalization correction.}}

{{FIG:pilot/03_layer_trajectory.png|Layerwise residual interaction norms for the learned direction and one seeded random example. These curves show propagation, with the intervention applied at block 17. They do not establish which component mediates an output effect.}}

Adding the learned context coordinate does not produce a pooled prediction-improvement interval entirely above zero for any of the three supported entries. The weaker conclusion is uncertainty, not proof that context is useless: fit and check each contain only 16 groups per feature and the chosen response variables may be poor measures of its function. Nevertheless, this pilot does not supply the predictive link required by the central hypothesis.

{{PREDICTION_TABLE}}

{{FIG:pilot/04_prediction_check.png|Prediction error on check groups after pooled fit-only calibration. Lower MSE is better. Context and PC models add different information to the nuisance baseline, while the gain model restricts the predicted output direction.}}

We also retain the change in negative log probability of the observed next token, measured in nats. Positive change means that the edit made the actual continuation less likely. This is a local quality check, not a complete generation evaluation. One prefix reaches the collection boundary and has no observed next token; its loss is missing while its valid interaction measurements remain.

{{FIG:pilot/05_prediction_quality.png|Interaction norm versus combined-edit change in next-token loss. This checks whether a visible diagnostic effect accompanies worse local prediction. It neither measures long-form generation quality nor validates the two contrasts as semantic endpoints.}}

## 13. New exploratory analysis using saved responses

The initial report summarized interaction magnitudes. Taking a norm discards sign, so large average magnitudes can coexist with cancellation across prompts. The saved four-cell outputs allow us to examine this limitation without another model run. This analysis was planned after viewing the initial norm and prediction results; it is explicitly post-result exploration. It neither changes the original endpoints nor converts the check split into fresh confirmation.

### 13.1 Separate sign-consistent interaction from even components

At a fixed context length, let I_st be the signed interaction for decoder sign s and context-rotation sign t, each plus/minus one. Its four parity components are:

![Equation](equations/parity.png)

The component C_11 is odd in both signs. A locally bilinear response has this symmetry: reversing either intervention reverses the interaction. The remaining components capture even contributions at these finite steps, including the shared inward radial term of the spherical context edit and higher-order response terms. Orthogonality of the four sign patterns gives an exact decomposition of the average squared interaction norm. The odd-odd energy fraction is not a semantic score.

{{PARITY_TABLE}}

For the learned direction, mean odd-odd energy fractions on check groups are about 90.0%, 96.7% and 96.9% for entries 1645, 2966 and 28027. Thus the unsigned effects are largely associated with a signed local interaction rather than being dominated by even-in-sign terms. The same property occurs for control directions. It is compatible with generic mixed curvature and does not establish a feature-specific mechanism.

{{FIG:signed/07_sign_parity.png|Fraction of squared interaction energy in the component odd in decoder sign and context sign. Each dot is one check group's average across the two lengths; black marks show means. All directions are retained, including random and generic geometry controls.}}

### 13.2 Local scaling at the two available context lengths

To compare the two lengths, we divide C_11 by the absolute fractional decoder dose a and context fraction b, defining K. This is a response coefficient in the experiment's fractional intervention coordinates. We also retain coefficients divided by actual native alpha and chord length in the numerical tables. The finite-difference relation and the bounded discrepancy are:

![Equation](equations/scaling.png)

For a differentiable readout and infinitesimal tangent edits, native-unit scaling approaches a mixed Hessian action along the decoder and context tangent. We have only one absolute decoder-dose magnitude and two finite context lengths, so these data cannot establish derivative convergence or rule out higher odd terms. K also remains feature-specific in its dose calibration: multiplying by fitted feature scale and baseline norm changes its units from native curvature.

Median relative discrepancies between the small- and large-length coefficients are 5.0% for 1645, 2.7% for 2966 and 1.8% for 28027. The largest discrepancy is 18.3% for 1645. These numbers suggest reasonably stable local scaling over the tested context-length range, while leaving decoder-dose convergence open. Ratios become unstable for vanishing responses; the raw signed coordinates are therefore shown alongside the summary.

{{FIG:signed/08_length_consistency.png|Signed response coefficients at context chords of 0.5% and 1% of residual norm. Circles show the digit contrast and triangles the word contrast. The diagonal represents equal coefficients after fractional-dose normalization. Each point is a check case, not an independent new experiment.}}

### 13.3 Do signed responses transfer from fit to check?

A stable mean signed response is a deliberately simple transfer target. For each feature and intervention direction, we average K over the two lengths within each group, estimate its mean on fit groups, and predict that vector on check groups. We compare its error with a zero-response baseline:

![Equation](equations/transfer.png)

Positive improvement means that the fit mean predicts the signed response better than zero. This is a different, exploratory target from the original regression of e0 on context: a successful constant mean would establish neither prompt-specific context prediction nor selective function. Calibration uncertainty is again omitted, and results for every direction and both cross-source calibrations are retained in the tables.

{{FIG:signed/09_signed_response.png|Signed learned-direction response coefficients, averaged across lengths, with both output contrasts displayed separately. Color distinguishes source; crosses denote fit groups and circles check groups. Dispersion and sign changes explain why interaction magnitude alone cannot demonstrate transfer.}}

{{TRANSFER_TABLE}}

The learned direction has no pooled transfer-improvement interval entirely above zero. Entry 28027 has a positive mean, but its interval spans zero; source-to-source learned transfer is also uncertain for all three entries. For 1645, removing one check group can change the pooled point estimate from negative to positive, exposing substantial influence at this sample size. Its large unsigned interaction is therefore not a reliable shared signed response in this pilot.

{{FIG:signed/11_signed_transfer.png|Fit-mean signed-response prediction compared with zero on check groups. Positive values favor transfer of a constant signed response. All seven directions are shown; intervals are uncorrected and conditional on the fitted mean. This is distinct from the original context-score prediction task.}}

The leading-PC direction for 28027 has a positive descriptive transfer interval, approximately 0.20 to 2.03 in squared coefficient units. This is one of many exploratory comparisons, not a corrected discovery. Its relevance is the alternative explanation it raises: a generic high-variance direction may organize responses at least as well as the feature-derived context direction. It does not establish PC superiority: the direction-specific response targets differ, and a common-target prediction comparison is needed.

### 13.4 Specificity depends on the control

The new paired analysis compares learned-minus-control norms on the same check groups, keeping signs, doses and lengths matched. For 1645 and 28027, beyond-gain differences favor learned over the approximate spectrum and shuffled-label controls in descriptive intervals. Yet all learned-minus-leading-PC beyond-gain intervals include zero, as do the learned-minus-mean-random beyond-gain intervals. Thus the analysis supports sensitivity to control choice, not a general selective advantage of the learned direction.

{{PC_TABLE}}

{{FIG:signed/10_specificity_controls.png|Paired learned-minus-control comparisons, separately for raw interaction and the component beyond scalar gain. Intervals resample the same check groups with equal group weights. Positive values favor learned. Leading-PC controls expose a more demanding alternative than shuffled labels alone.}}

These results narrow the next question. The immediate target is not to find another large interaction, but to determine whether a feature-derived context coordinate explains a reproducible selective effect after accounting for generic high-variance context. The existing signed outputs materially improve this diagnosis without adding inference cost, while also showing why the current strongest claim must remain open.

### 13.5 Does the learned score add information beyond PCs?

A final CPU-only diagnostic addresses the common-target prediction question directly. The original eight-PC predictor already includes the learned context score. We add the missing nuisance-plus-PC-only model, retain the same preprocessing and fixed ridge penalty, and measure PC-only MSE minus PC-plus-context MSE on the same check decoder effects. This comparison changes the predictor while keeping its target fixed. It was added after the earlier exploratory results and remains a development analysis.

{{INCREMENTAL_TABLE}}

{{FIG:signed/12_incremental_prediction.png|Incremental prediction improvement from adding the learned context score to the nuisance-plus-eight-PC model. Every supported feature and calibration source is shown. Positive means reduced MSE on the same decoder-effect target. Cross-source intervals use eight check groups and pooled intervals use sixteen; all are descriptive and uncorrected.}}

No feature has a positive pooled interval. Adding context worsens pooled prediction for 1645, with mean improvement -4.01 times 10^-5 and an interval entirely below zero under this fixed-fit bootstrap. The Wikipedia-to-web comparison for 28027 is positive, about 1.74 times 10^-5 with interval [4.05 times 10^-6, 3.67 times 10^-5], but does not replicate in the reverse direction or pooled analysis.

The apparent cross-source gain also illustrates why the baseline matters. In that 28027 comparison, PC-only MSE is approximately 0.000348 and PC-plus-context MSE 0.000330, whereas nuisance-only MSE is 0.000189 and nuisance-plus-context MSE 0.000182. The richer models remain worse than the simpler alternatives: an improvement over an underperforming PC model is not evidence of a successful selective predictor. Small calibration sets and fixed regularization are plausible limitations. The analysis reproduces every original prediction error exactly while isolating the previously missing increment.

This completes the immediately useful Phase 1 checks: the existing archive supports precise diagnosis of local modulation, control dependence and failed or unstable prediction. A stronger result now requires a better validated behavioral target and independent evidence, rather than further selection among these same comparisons.

## 14. Integrated interpretation

### 14.1 What the evidence establishes

The foundation gives a quantitative account of how an overcomplete SAE dictionary occupies residual space. Decoder participation spans hundreds of dimensions, alignment with high-variance PCs differs from reconstruction quality, and decoder proximity need not imply token-level coactivation. These are useful empirical results even before making a causal interpretation.

Controlled observational comparisons identify a subset of selected pseudo-concepts whose weak and strong occurrences differ in partner composition and native geometry beyond the focal decoder. The subsequent intervention pilot demonstrates that associated constrained context edits can modulate diagnostic decoder effects. The signed analysis shows mostly local odd-odd response structure at the tested lengths. These statements concern different estimands and should not be collapsed into one proof.

### 14.2 What remains unsupported

Neither mixture separation nor a large orthogonal displacement establishes distinct semantic identities. The background and matched controls show that contextual variation is not exclusive to strongly separated activation distributions. A large factorial interaction does not establish a novel mechanism, and preserving the feature score does not preserve every other property of the state. Current prediction and gain-control results do not establish the central stronger hypothesis.

The most coherent reading is that **activation-conditioned geometry is reproducible in selected cases, while its conversion into a specific, transferable explanation of intervention effects remains unresolved**. The leading-PC comparisons make generic activation geometry a concrete competing explanation. This is a scientifically useful boundary: it identifies the missing link instead of rewarding increasingly elaborate descriptions of the same data.

| Claim | Evidence available | Appropriate wording |
| --- | --- | --- |
| Dictionary directions broadly cover coordinates and PCs | Full dictionary participation and evaluation reconstruction | Descriptive geometry, with explicit metric and population. |
| Selected intensity regimes differ in context | Two-source conditional partner and orthogonal tests | Conditional association for selected entries. |
| Context edits modulate a decoder intervention | Native four-cell responses with numerical audits | Controlled interaction in diagnostic outputs. |
| Learned context uniquely predicts selective function | Prediction, gain and PC comparisons are inconclusive | Not established by the present pilot. |
| A particular downstream mechanism explains modulation | No pathway ablation/rescue | Open research question. |

## 15. Phase 2: a focused continuation

### 15.1 Improve the behavioral question before increasing scale

The next stage should validate what a candidate actually affects. The exported blinded context sheets can support independent annotation of weak and strong examples, with disagreement recorded rather than resolved by looking at outcomes. For entry 1645, numerical-string position and year formatting are concrete nuisance alternatives; entry 28027 includes word and subword fragments. These observations motivate candidate endpoints but do not determine their semantic labels.

Use a small development set to specify at least two interpretable, jointly meaningful output contrasts or continuation-level scores. Calibrate a minimum useful effect and a quality budget before choosing the confirmation sample size. Keep diagnostics on unrelated continuations so apparent selectivity is not simply general disruption. Larger models can be an external-validity step, but more parameters alone will not solve an ambiguous endpoint or a weak control.

### 15.2 Test feature-specific context against generic geometry

Freeze a predictor using feature-derived context and compare it with nuisance, scalar-gain and an appropriately regularized PC-context model. The next decisive endpoint is incremental prediction of selective effect on new documents, not the existence of an interaction. Match perturbation constraints and native step sizes, calibrate regularization without using confirmation outcomes, and compare prediction errors on paired prompts.

A complementary candidate is a shared response subspace learned on development data and evaluated on new prompts with a fixed projection. It is useful only if it predicts beyond the generic-PC and gain alternatives. A low-rank construction is not evidence that the model's full function is intrinsically low-dimensional, and a positive post-result fit must be labeled exploratory until tested independently.

### 15.3 Establish a mechanism with targeted inference

If a candidate passes the behavioral and predictive gate, run a small predeclared pathway experiment: block the proposed mediator, measure attenuation of the predicted interaction, and attempt a matched rescue. Include a neighboring or unrelated pathway control. Norm trajectories can guide where to inspect, but cannot replace these interventions. Save signed intermediate vectors or the necessary projections; the present archive's layer norms cannot recover them.

Use additional smaller absolute decoder doses to test local convergence and distinguish a persistent response from a finite-step artifact. The present two context lengths already inform a useful range, but cannot answer the decoder-dose question. These focused passes are far cheaper than repeating the entire positive-activation collection and should precede a larger confirmation run.

### 15.4 Freeze confirmation and replicate the analytical unit

Once the endpoint and control hierarchy are defensible, freeze the hypothesis, effect threshold, exclusions, multiplicity family and stopping rule. Estimate sample size from document-level variation with allowance for calibration uncertainty and source heterogeneity. Collect fresh deduplicated prompts; do not reuse this report's check cases as confirmation. Require replication across sources and an independent SAE dictionary, width or training seed before treating an individual entry as a stable unit of explanation. A larger model is a subsequent scope test rather than a substitute for these checks.

| Stage | Can use current saved results? | Decision before moving on |
| --- | --- | --- |
| Geometry, sign and generic-control diagnosis | Yes; completed here | Identify the specific unresolved contrast. |
| Blind review and endpoint specification | Existing excerpts can start this | Establish a meaningful behavioral target. |
| Decoder-dose and pathway pilot | Requires limited additional inference | Validate local behavior and a plausible mediator. |
| Fresh predictive and selective confirmation | Requires new prompts and inference | Beat gain and generic-geometry baselines by a useful amount. |
| SAE/model replication | Requires another dictionary or model | Assess robustness of the analytical unit and scope. |

## 16. Limitations and reproducibility

The observational study uses two English sources, a finite streaming sample, one model and two dependent layers. Duplicate detection is approximate. Conditional exchangeability is assumed within stated strata; lexical, position and sparsity controls do not eliminate topic, syntax or every encoder-related explanation. Candidate selection favors separated intensity fits and cannot estimate dictionary-wide prevalence. Strict matched controls were unavailable and supported weak-control pairs were few.

PCA describes covariance rather than semantics. Low-variance directions are sensitive to finite-sample estimation; whitening changes the metric and can amplify noise. Participation depends on coordinates. Good centered reconstruction does not show that every behaviorally important direction is reconstructed, and directional mass does not measure causal importance.

The intervention study has only three supported entries, two formatting contrasts and 16 check groups per entry. Its fit/check separation does not make the older corpus a fresh confirmation source. Artificial edits can leave the natural data manifold despite preserving three focal quantities. Float32 inference differs from the bfloat16 observational collection. An inactive case and a missing continuation target are retained with their respective limitations. Fixed-fit bootstrap intervals understate uncertainty if interpreted as including the entire training procedure; no such interpretation is intended.

The new signed analysis is post-result exploration across multiple directions and endpoints. A largely odd-odd response is consistent with generic local curvature. Two finite context lengths and a single absolute decoder-dose magnitude do not establish an infinitesimal derivative. The two-dimensional output metric limits what gain, energy and transfer mean. No completed blinded semantic validation, pathway ablation/rescue, fresh confirmation or independent-SAE replication is available.

Figures and tables are regenerated from retained artifacts. The new analyses use saved outputs without model execution. Protocols, exact versions and execution records are maintained separately. This Phase 1 report supports research discussion and a focused continuation; unresolved claims remain questions.

## References

{{REFERENCES}}

[15] Hoang et al. (2026). [Sparse Autoencoders Encode Both Concepts and Functions: The Downstream Geometry of Feature Effects](https://arxiv.org/abs/2607.24645).

[16] Duan (2026). [Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects](https://arxiv.org/abs/2606.08365).

[17] Vaidyanathan et al. (2026). [The Curse of Multiple Mediators: Hidden Interaction Effects in Activation Patching](https://arxiv.org/abs/2606.27510).
"""


def table(frame, columns, header, digits=4):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            if col == "interval":
                values.append(f"[{row.low:.{digits}g}, {row.high:.{digits}g}]")
            elif col == "feature_id":
                values.append(str(int(row[col])))
            elif isinstance(row[col], float):
                values.append(f"{row[col]:.{digits}g}")
            else:
                values.append(str(row[col]))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for source, name in [
        (FOUNDATION / "figures", "foundation"),
        (PILOT, "pilot"),
        (SIGNED, "signed"),
    ]:
        dest = OUT / name
        dest.mkdir(exist_ok=True)
        for path in source.glob("*.png"):
            shutil.copy2(path, dest / path.name)
    equations = {
        "rotation": [
            r"$Q=P_{\mathrm{span}(w,u)^\perp},\quad r=Qh,\quad v=\frac{Qc-r(r^\top Qc)/\|r\|^2}{\|Qc-r(r^\top Qc)/\|r\|^2\|}$",
            r"$\delta=(\cos\theta-1)r+\sin\theta\,\|r\|v,\quad \beta=2\|r\|\sin(|\theta|/2)$",
        ],
        "interaction": [
            r"$e_0=Y(h+\alpha u)-Y(h),\quad e_1=Y(h+\delta+\alpha u)-Y(h+\delta)$",
            r"$I=e_1-e_0=Y(h+\delta+\alpha u)-Y(h+\delta)-Y(h+\alpha u)+Y(h)$",
        ],
        "gain": [
            r"$g_* = \frac{e_0^\top e_1}{\|e_0\|_2^2},\qquad R_{\mathrm{gain}}=\|e_1-g_*e_0\|_2$"
        ],
        "normalization": [
            r"$Y(h)=\frac{a^\top h}{R(h)},\quad R(h)=\sqrt{\|h\|^2/d+\epsilon}$",
            r"$I_{\mathrm{norm}}=(a^\top\delta)\left(\frac{1}{R_\alpha}-\frac{1}{R_0}\right)$",
        ],
        "ridge": [
            r"$(\widehat B,\widehat c)=\arg\min_{B,c}\sum_{i\in\mathrm{fit}}\|e_{0i}-X_iB-c\|_2^2+\lambda\|B\|_F^2,\quad\lambda=1$"
        ],
        "parity": [
            r"$C_{pq}=\frac{1}{4}\sum_{s,t\in\{-1,+1\}}s^p t^q I_{st},\quad p,q\in\{0,1\}$",
            r"$F_{11}=\frac{\|C_{11}\|^2}{\sum_{p,q}\|C_{pq}\|^2},\quad\frac{1}{4}\sum_{s,t}\|I_{st}\|^2=\sum_{p,q}\|C_{pq}\|^2$",
        ],
        "scaling": [
            r"$K_b=\frac{C_{11}(b)}{|a|b},\quad \frac{C_{11}}{|\alpha|\beta}\ \longrightarrow\ H_Y(h)[u,v]$",
            r"$D=\frac{2\|K_{0.005}-K_{0.01}\|_2}{\|K_{0.005}\|_2+\|K_{0.01}\|_2}$",
        ],
        "transfer": [
            r"$\widehat\mu=\frac{1}{n_{\mathrm{fit}}}\sum_{i\in\mathrm{fit}}K_i,\quad\Delta_i=\frac{\|K_i\|^2-\|K_i-\widehat\mu\|^2}{2}$"
        ],
    }
    (OUT / "equations").mkdir(exist_ok=True)
    for name, expressions in equations.items():
        fig, ax = plt.subplots(figsize=(10, 0.62 * len(expressions)))
        ax.axis("off")
        for i, expression in enumerate(expressions):
            ax.text(
                0.5,
                1 - (i + 0.5) / len(expressions),
                expression,
                ha="center",
                va="center",
                fontsize=14,
            )
        fig.savefig(
            OUT / "equations" / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.14
        )
        plt.close(fig)
    source = (FOUNDATION / "scientific_report.md").read_text()
    foundation = source.split("## 3. Experimental material")[1].split("## 11. Interpretation")[0]
    foundation = "## 3. Experimental material" + foundation
    foundation = foundation.replace("](" + "figures/", "](foundation/")
    # Keep the full mathematical guide and quantitative results; omit five redundant views.
    for n in [6, 9, 15, 19, 20]:
        foundation = re.sub(rf"!\[Figure {n}\. [^\n]+\n\nFigure {n}\. [^\n]+\n*", "", foundation)
    # The dictionary-PCA paragraph would otherwise refer to an omitted figure.
    foundation = foundation.split("Finally, PCA of centered unit decoder vectors")[0]
    foundation = foundation.replace(
        "## 10. Complementary geometric checks",
        "## 10. Complementary geometric checks and transition to intervention",
    )
    foundation += "\nThe next stage asks whether these observational directions can explain changes in the effect of a fixed decoder edit. The geometry motivates the intervention; it does not predetermine its outcome.\n\n"
    body = EXTENSION
    raw = pd.read_csv(PILOT / "pooled_check_controls.csv")
    raw["endpoint"] = raw.endpoint.map(
        {"interaction_norm": "Raw interaction", "beyond_gain_norm": "Beyond gain"}
    )
    body = body.replace(
        "{{RAW_TABLE}}",
        table(
            raw,
            ["feature_id", "endpoint", "mean", "interval"],
            ["Entry", "Learned minus random", "Mean", "95% descriptive interval"],
        ),
    )
    pred = pd.read_csv(PILOT / "prediction_improvement.csv").query("train_source == 'pooled'")
    body = body.replace(
        "{{PREDICTION_TABLE}}",
        table(
            pred,
            ["feature_id", "mean", "interval"],
            ["Entry", "Pooled MSE improvement", "95% descriptive interval"],
        ),
    )
    sign = pd.read_csv(SIGNED / "signed_coefficients.csv").query(
        "direction == 'learned' and split == 'check'"
    )
    parity = sign.groupby("feature_id").odd_fraction.agg(["mean", "median"]).reset_index()
    length = pd.read_csv(SIGNED / "length_consistency.csv").query(
        "direction == 'learned' and split == 'check'"
    )
    parity["length_discrepancy"] = parity.feature_id.map(
        length.groupby("feature_id").relative_discrepancy.median()
    )
    body = body.replace(
        "{{PARITY_TABLE}}",
        table(
            parity,
            ["feature_id", "mean", "median", "length_discrepancy"],
            [
                "Entry",
                "Mean odd-odd energy fraction",
                "Median fraction",
                "Median relative length discrepancy",
            ],
        ),
    )
    transfer = pd.read_csv(SIGNED / "signed_transfer.csv").query(
        "direction == 'learned' and calibration == 'pooled'"
    )
    body = body.replace(
        "{{TRANSFER_TABLE}}",
        table(
            transfer,
            ["feature_id", "mean", "interval", "leave_one_out_min", "leave_one_out_max"],
            [
                "Entry",
                "MSE improvement",
                "95% descriptive interval",
                "Leave-one-out minimum",
                "Leave-one-out maximum",
            ],
        ),
    )
    pc = pd.read_csv(SIGNED / "specificity_controls.csv").query("control == 'leading_pc'")
    pc["endpoint"] = pc.endpoint.map(
        {"interaction_norm": "Raw interaction", "beyond_gain_norm": "Beyond gain"}
    )
    body = body.replace(
        "{{PC_TABLE}}",
        table(
            pc,
            ["feature_id", "endpoint", "mean", "interval"],
            ["Entry", "Learned minus PC", "Mean", "95% descriptive interval"],
        ),
    )
    incremental = pd.read_csv(SIGNED / "incremental_prediction.csv")
    incremental["calibration"] = incremental.calibration.map(
        {"pooled": "Pooled", "fineweb-edu-sample": "Web to Wiki", "wikimedia-en": "Wiki to Web"}
    )
    body = body.replace(
        "{{INCREMENTAL_TABLE}}",
        table(
            incremental,
            ["feature_id", "calibration", "mean", "interval"],
            ["Entry", "Calibration to check", "MSE improvement", "95% descriptive interval"],
        ),
    )
    body = body.replace("{{REFERENCES}}", source.split("## References\n")[1].strip())
    text = INTRO + foundation + body
    # Assign a single continuous figure sequence after merging both studies.
    counter = 0

    def figure(match):
        nonlocal counter
        counter += 1
        caption, path = match.group(1), match.group(2)
        caption = re.sub(r"^Figure \d+\. ", "", caption)
        return f"![Figure {counter}. {caption}]({path})\n\nFigure {counter}. {caption}\n\n"

    text = re.sub(r"!\[(Figure \d+\. [^\n]+)\]\(([^\n]+)\)\n\nFigure \d+\. [^\n]+\n*", figure, text)

    def new_figure(match):
        nonlocal counter
        counter += 1
        path, caption = match.group(1), match.group(2)
        return f"![Figure {counter}. {caption}]({path})\n\nFigure {counter}. {caption}"

    text = re.sub(r"\{\{FIG:([^|]+)\|([^}]+)\}\}", new_figure, text)
    # Give each equation a readable, numbered caption and retain its exact source image.
    eq_counter = 0

    def equation(match):
        nonlocal eq_counter
        eq_counter += 1
        return f"![Equation {eq_counter}]({match.group(1)})\n\nEquation {eq_counter}."

    text = re.sub(r"!\[Equation\]\(([^)]+)\)", equation, text)
    assert "{{" not in text
    paths = re.findall(r"!\[[^\n]*\]\(([^)]+)\)", text)
    assert all((OUT / p).exists() for p in paths)
    (OUT / "report.md").write_text(text.rstrip() + "\n")
    # Curate the self-contained package: only figures referenced in the final report.
    for folder in ["foundation", "pilot", "signed"]:
        for path in (OUT / folder).glob("*.png"):
            if path.relative_to(OUT).as_posix() not in paths:
                path.unlink()
    (OUT / "tables").mkdir(exist_ok=True)
    for source_dir, prefix in [(PILOT, "pilot"), (SIGNED, "signed")]:
        for p in source_dir.glob("*.csv"):
            shutil.copy2(p, OUT / "tables" / f"{prefix}_{p.name}")
    manifest = {
        "status": "observational foundation plus development intervention and exploratory reanalysis",
        "figures": counter,
        "equations": eq_counter,
        "words": len(text.split()),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_reports": [str(FOUNDATION.relative_to(ROOT)), str(PILOT.relative_to(ROOT))],
    }
    (OUT / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
