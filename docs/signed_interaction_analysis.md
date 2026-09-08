# Exploratory analysis of saved signed interactions

This analysis was specified after the development pilot's norm and prediction results were inspected, but before calculating the signed summaries below. It uses the completed saved four-cell readouts only: no new model execution, new documents or independent confirmation. The original protocols and endpoints remain unchanged.

The question is whether the observed modulation is a coherent local response associated specifically with the learned context direction, rather than a large unsigned response to generic perturbations.

1. Decompose each two-dose, two-context-sign interaction grid into its four sign-parity components. Report the fraction of squared two-readout interaction energy in the component odd in both signs. This component is the finite-difference analogue of a mixed derivative, not a new estimator or theory.
2. Compare that signed component divided by absolute decoder dose and context fraction at the two available context lengths. Also retain division by actual native alpha and chord length. Only context length varies in magnitude; the existing data cannot establish convergence as decoder dose tends to zero.
3. For each direction and feature, learn a constant signed response vector on fit documents and evaluate its squared error on check documents against a zero-response baseline. Include pooled and cross-source calibration. Report all features and all directions. This is an exploratory new prediction target, separate from the original prediction of baseline decoder effects. Positive improvement does not demonstrate context-score prediction or semantic function.
4. Pair learned-direction raw and beyond-gain interaction norms against leading PC, approximate spectrum control, shuffled labels and mean random directions on the existing check cases. These are generic-geometry specificity diagnostics.

Average repeated conditions within duplicate group before inference. Use 2,000 seeded duplicate-group bootstrap resamples, with equal group weights. Intervals are descriptive, uncorrected, conditional on the frozen calibration and selected features; calibration uncertainty is omitted. Report leave-one-group-out ranges of point estimates to expose influence. Retain unsupported feature 7087 in the report's support accounting. Do not select endpoints, signs, doses or features according to favorable outcomes.

The two readouts are unstandardized logit contrasts. Euclidean norms, energy fractions and prediction error depend on this fixed choice of output coordinates. Signed component plots will display both contrasts separately. Layerwise saved norms cannot identify signed propagation or a mediating pathway.

## Additional Phase 1 diagnostic

After the signed/control analysis, add a PC-only prediction baseline to the original decoder-effect task. The existing `pc8_context` predictor already includes both eight PCs and the learned score. Compare PC-only MSE minus PC-plus-context MSE on the same check groups, for all supported entries and pooled/cross-source calibration. Keep the original fixed ridge penalty, preprocessing and target. This is a further explicitly exploratory check of incremental information; it uses no new model inference. Positive results would still require fresh-data validation and richer generic baselines.
