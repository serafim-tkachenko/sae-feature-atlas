# Reanalyzing saved evidence

The browser exporter displays saved examples and statistics. Changing the current
token policy does not retroactively change an older run. Reports flag formatting
targets in exported examples; they never hide those targets while retaining the
old summary statistics.

To apply the current token policy to an older run without repeating model inference:

```bash
uv run python scripts/reanalyze_saved_evidence.py \
  --run-dir data/processed/OLD_RUN \
  --output-dir data/processed/NEW_RUN \
  --manifest reports/OLD_RUN/manifest.json \
  --tokenizer /path/to/matching/tokenizer.json
uv run sae-atlas report --run-dir data/processed/NEW_RUN --output reports/NEW_RUN/index.html
```

The output directory must be new and outside the original run. The script checks
token/activation joins, sparse key uniqueness, finite positive activations and
every saved token ID/string against the supplied local tokenizer before writing.
It uses the manifest's model/collection identity and feature-selection thresholds,
and today's default token filter and analysis settings. It does not download models.

Statistics, denominators, feature selection, strongest examples, coactivation,
inspection summaries, activation regimes and triage labels are recomputed together.
New examples come from the complete saved activation table, so valid alternatives
remain available when all the old strongest examples were quotes or punctuation.
Surrounding context remains intact. Histograms deliberately retain all stored rows,
including excluded targets, and identify that population in the report.

Saved decoder cosines are reused for retained source features when available;
they are not a fresh nearest-neighbor search. Old PCA and alignment diagnostics
are omitted. The new `reanalysis.json` records the filter, settings, input/output
hashes, tokenizer hash and limitations. It does not create a collection fingerprint:
legacy collection remains unverified. Replaying model inference requires the
original checkpoint revisions and runtime settings.

For interpretation, read several independent contexts. A sidebar example target
is an observation, not a semantic feature name. In a causal language model, right
context helps the reader but was unavailable when the target activation was computed.
