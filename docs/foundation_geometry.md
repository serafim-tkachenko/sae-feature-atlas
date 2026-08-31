# Geometry extension to the foundation experiment

Added at the user's request during activation collection, before the new run's
discovery fits or evaluation results were inspected. The primary protocol and
its statistical family remain fixed. This is a prespecified secondary geometry
extension; it does not turn the primary association test into a causal test.

## Central research question

Does activation strength mark a reproducible change in the representational
context of a feature, beyond movement along its own decoder direction?

The distinction matters because changing a feature's amplitude alone moves its
SAE reconstruction along a fixed direction. A contextual interpretation requires
evidence about the surrounding representation, and ultimately semantic review.

## Measurements

1. Compare native normalized decoder directions with low/high coactivation
   neighborhoods. Report cosine similarities and top-20 neighbor overlap in the
   full decoder space, using the supported partner universe explicitly.
2. Use decoder PCA as a visualization, with explained variance shown. Projection
   distance and apparent clusters are not inferential evidence.
3. Replay the pinned model to capture native residuals at both layers in one
   pass. Fit residual PCA on a fixed source-balanced sample of discovery
   documents only (300 per source, eligible positions every 16 tokens). Capture
   the exact held-out token locations used by source confirmation for all selected
   features, so geometric comparisons share the primary observational population.
4. For each feature/source, compute the mean high-minus-low native residual
   displacement. Decompose it into its component along the unit decoder
   direction and its orthogonal complement. Report norms and cross-source cosine
   agreement of the orthogonal displacement. A near-zero displacement has no
   stable direction and must be flagged rather than assigned a confident angle.
   Calibrate the squared orthogonal mean-shift norm with 4,999 permutations
   under the same within-source token/position/support null as the primary
   endpoint. Use the maximum source p-value and BY across selected features as
   a separate secondary family per layer. Finite-sample nonzero displacement
   alone is not evidence of contextual change. Cross-source cosine agreement is
   descriptive, and is not an additional independent replication test.
5. Compare the native context displacement with the signed decoder-space sum of
   binary partner-probability changes. This is a descriptive consistency check:
   the two measurements share representations and are not independent evidence.
6. Fit a full residual PCA basis on the discovery reference sample. Show the
   variance spectrum and absolute decoder squared mass in the first 1, 5, 20
   and 64 PCs, as well as the remaining and lowest-variance components. Report
   participation ratio (1/sum(p^2)), entropy effective dimension, and the number
   of components carrying 90% of directional mass in raw and PCA coordinates.
   These masses are not reconstruction quality or feature importance.
7. Use regularized whitening with variance floors of 0.001 times the largest
   discovery eigenvalue, and 0.0001/0.01 sensitivity floors. Normalize transformed
   decoder directions before computing participation. Whitening changes the
   metric and amplifies low-variance directions; its output is basis-dependent.
8. Select 64 features uniformly from the discovery-supported screen, regardless
   of mixture qualification. Freeze positive-activation quartiles on discovery,
   then sample one active occurrence per duplicate group per source. Quantile
   comparisons are descriptive and retain support failures; they are not added
   to the primary multiple-testing family. Reference sampling and this arm use
   seed 20260908 with fixed offsets 109 and 103, respectively.
9. Compare actual SAE reconstruction error with native residual variance in a
   separate, fixed evaluation reference sample (300 documents per source,
   eligible positions every 16 tokens). Per-PC reconstruction R-squared is
   distinct from decoder alignment and may be negative in weakly represented
   directions.

Use all selected features for quantitative summaries. Choose illustrated cases
by the primary conjunction q-value, then feature ID, before viewing geometry.
Cases with failed source support remain visible in the support accounting.
Native residual replay and all geometric outputs have separate provenance.

## Planned graphics

- The hypothesis: amplitude along a decoder direction versus contextual change.
- Source allocation and discovery/evaluation design.
- Discovery activation distributions and frozen regime assignments.
- Source-specific effect replication, with corrected outcomes.
- Individual effect sizes, null references and uncertainty intervals.
- Signed changes in partner occurrence probabilities.
- Decoder geometry and neighborhood overlap.
- Residual variance spectrum and decoder/PC alignment.
- Held-out native context projections in discovery-fitted PCA coordinates.
- Full-space orthogonal displacement and cross-source direction agreement.
- Strict and weak-control comparisons, plus dependence sensitivities.

Figures should answer these distinct questions and carry clear captions. Blinded
context exports support a subsequent colleague annotation exercise; no completed
human semantic validation is implied by the geometric figures.
