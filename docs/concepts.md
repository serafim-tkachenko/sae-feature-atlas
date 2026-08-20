# Concepts

## Scientific activation populations

`all_stored_activations` contains every persisted sparse activation row. Under top-k collection, appearance means retained top-k membership and must not be called a true-positive activation.

`analysis_activations` is the deterministic subset whose target token passes the configured corpus, position, token-quality, and activation-row policy. Artifact activity remains in the stored population even when excluded from human-facing analysis.

`analysis_features` contains features meeting configured support requirements in the analysis population. Downstream call sites state this population explicitly.

## Context evidence

Raw evidence records text ID, target and context positions, contiguous token IDs, raw token strings, target token ID/string, activation value, target quality, and display quality. Display text is decoded from the contiguous ID span by the actual tokenizer rather than by joining token strings. Punctuation in the surrounding display context does not invalidate a clean target.

## Coactivation

Under top-k storage, coactivation is joint retained feature membership on the same eligible token. Counts, marginals, and PMI use one unique token-feature representation and the full explicit eligible-token denominator. Minimum support and retention metadata distinguish measured edges from missing or truncated ones.

## Decoder geometry

Decoder cosine neighbors are directed. Empirical coactivation pairs are unordered, so comparisons use canonical pair keys while preserving directed neighbor rank. Unmatched edges are missing, not zero coactivation.

## Bimodality

Every eligible feature receives an evaluation status. A separate candidate artifact contains only converged fits meeting configured delta-BIC, component weight, and separation thresholds. Under top-k storage, these distributions are rank-censored. GMM preference is not evidence of distinct semantic concepts.

## Triage and semantic annotation

Artifact scores, `interpretability_triage_score`, manual priority, and labels such as `likely_artifact`, `high_frequency`, `bimodal_candidate`, and `manual_review` are heuristic triage.

A semantic annotation is a separate structured record built primarily from empirical contexts. It supports descriptions, scope, evidence IDs, counterexamples, uncertainty/failure modes, confidence, abstention, regime-specific descriptions, and annotator provenance. Decoder and coactivation neighbors are supplementary evidence, not direct language translations of decoder vectors.

## Diagnostic geometry

Residual PCA, normalized-decoder PCA, decoder UMAP, graph alignment, and decoder/residual-PC alignment are descriptive diagnostics or exploratory tools. They do not establish semantic axes, reconstruction quality, or causal structure.
