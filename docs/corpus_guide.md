# Corpus guide

Corpus choice strongly affects SAE feature statistics, coactivation patterns,
bimodality candidates, and feature-card examples.

For objective analysis, avoid relying only on a narrow corpus. Use small corpora
for debugging and broader mixed corpora for research claims.

## Corpus types

### Small/debug corpora

Use these for smoke tests and fast iteration.

```text
tinystories
pile-10k
```

They are useful for validating the pipeline, but they are too narrow for strong
research claims.

### Focused corpora

Use these to study domain-specific feature behavior.

```text
wikimedia-en
openwebtext
math-small
code-small
fineweb-edu-sample
```

Focused corpora are useful for asking whether some features specialize in
narrative, encyclopedic, technical, mathematical, or code-like contexts.

### Mixed corpora

Use these for broader analysis.

```text
mixed-research
mixed-broad
mixed-large
```

Mixed corpora reduce the risk that the atlas is dominated by one writing style
or source. They are preferred for analysis-workflow analysis.

### Custom JSONL corpus

You can pass:

```text
jsonl:/path/to/file.jsonl
```

Expected row format:

```json
{"source": "my_source", "text": "Text to analyze"}
```

The `source` field is important. It allows later filtering, source-risk checks,
and domain-level comparisons.

## Recommended progression

Start small:

```bash
uv run sae-atlas run \
  --preset research \
  --corpus mixed-broad \
  --max-texts 50 \
  --max-seq-len 128
```

Then scale:

```bash
uv run sae-atlas run \
  --preset research \
  --corpus mixed-broad \
  --max-texts 300 \
  --max-seq-len 256
```

For larger larger research runs:

```bash
uv run sae-atlas run \
  --preset paper \
  --corpus mixed-large \
  --max-texts 2000 \
  --max-seq-len 512
```

## Corpus and metric interpretation

### Feature frequency

Feature frequency is corpus-dependent. A feature that is frequent on code may be
rare on narrative text.

### Coactivation

Coactivation pairs are also corpus-dependent. Features that coactivate on math
text may not coactivate on ordinary prose.

### Bimodality

Bimodality candidates should be inspected across sources. A “bimodal” feature
may simply separate two domains, such as code and prose, rather than reflect a
strength continuum inside one concept.

### Decoder geometry

Decoder geometry is model/SAE-internal, but geometry-vs-coactivation comparisons
depend on the empirical corpus.

### Residual PCA coverage

Residual PCA is fitted to sampled residual vectors. The PCA basis changes when
the corpus changes.

## Practical advice

For quick debugging:

```text
tinystories or pile-10k
```

For the main analysis-workflow analysis:

```text
mixed-broad
```

For more objective analysis:

```text
mixed-large
```

For controlled domain comparisons:

```text
run the same model/layer on multiple focused corpora and compare outputs
```

## Avoid overclaiming

Do not claim that a feature is globally frequent, globally bimodal, or generally
coactivating unless the result is checked across corpora or explicitly scoped to
the selected corpus.
