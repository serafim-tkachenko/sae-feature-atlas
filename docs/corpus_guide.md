# Corpus guide

Corpus choice strongly affects activation statistics, coactivation, bimodality,
and examples. Treat every run as corpus-conditioned.

Recommended development pattern:

```bash
uv run sae-atlas run --preset core --corpus pile-10k --max-texts 50
uv run sae-atlas run --preset atlas --corpus pile-10k --max-texts 500
uv run sae-atlas run --preset research --corpus mixed-broad --max-texts 1500
```

Use `list-corpora` to see available presets:

```bash
uv run sae-atlas list-corpora
```

For larger publication-oriented experiments, keep using the `research` preset and
increase corpus size explicitly
