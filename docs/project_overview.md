# Project overview

SAE Feature Atlas is a local analysis workflow for Gemma Scope SAEs. It is meant
to make a run inspectable: you can go from corpus texts and token activations to
feature cards, plots, reports, and manual examples.

## Current scope

The project currently studies:

- sparse SAE activation statistics;
- top token/context examples per feature;
- same-token feature coactivation;
- SAE decoder-neighbor geometry;
- geometry/coactivation agreement and disagreement;
- possible bimodal activation regimes;
- automated artifact/inspection triage;
- residual PCA coverage by decoder directions;
- graph alignment between decoder-neighbor and coactivation-neighbor graphs.

## Out of scope for now

The project does not currently claim:

- causal-control experiments;
- production-ready behavior editing;
- automatic semantic interpretation

Those can be future research directions, but keeping them out of the core repo
makes the current tool easier to understand and trust.
