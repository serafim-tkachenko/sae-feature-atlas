# Project overview

SAE Feature Atlas is a local research toolkit for collecting, storing,
inspecting, and visualizing sparse autoencoder activations from transformer
models.

The current reference backend is Gemma Scope. The code should stay structured so
that additional SAE families can be added later, but the immediate goal is one
high-quality end-to-end path rather than a broad provider framework.

## Goals

The project has two connected goals:

1. **Reusable toolkit**
   - choose a model, layer, SAE, and corpus;
   - collect residual-stream activations;
   - encode SAE activations;
   - save reusable artifacts;
   - load artifacts from notebooks;
   - generate readable feature cards and reports.

2. **Research workflow**
   - follow the analysis workflow around feature filtering, coactivation,
     bimodality, decoder geometry, and residual PCA coverage;
   - generate evidence tables and visualizations;
   - keep claims explicit and limited by the available evidence.

## Pipeline overview

```text
config
→ runtime loading
→ activation collection
→ feature filtering/statistics
→ coactivation analysis
→ decoder geometry
→ bimodality and activation regimes
→ residual PCA / decoder coverage
→ graph alignment and features for manual follow-up
→ feature cards
→ report
```

## Design principles

### Keep model-specific code at the boundary

Gemma Scope-specific logic should stay in loading/config/runtime code. Generic
analysis should consume artifacts such as:

- activation tables;
- token metadata;
- residual vector samples;
- decoder weights;
- feature IDs;
- run metadata.

Analysis modules should not depend on a Gemma-specific runtime object.

### Prefer explicit research artifacts

Every important analysis step should write a table or figure that can be loaded
later. This makes the project useful as a research tool rather than only a CLI
script.

### Do not overclaim

Automated labels, intervention scores, graph-alignment buckets, and bimodality
scores are triage signals. They help decide where to inspect. They are not final
semantic interpretations or causal evidence by themselves.

## Current non-goals

- Stable top-level Python API.
- Support for every SAE family.
- Web dashboard.
- Claims about causal intervention without a controlled intervention protocol.
- Production-level dataset hosting.
