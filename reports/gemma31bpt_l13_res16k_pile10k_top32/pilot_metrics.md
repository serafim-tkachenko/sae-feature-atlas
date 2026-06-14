# Pilot metrics

Run: `gemma31bpt_l13_res16k_pile10k_top32`

## Setup

- Model: `google/gemma-3-1b-pt`
- SAE: `gemma-scope-2-1b-pt-res / layer_13_width_16k_l0_medium`
- Hook: `blocks.13.hook_resid_post`
- Corpus: `pile-10k`
- Texts: 300
- Tokens: 67,167
- Activation mode: top-k
- Top-k: 32

## Generated artifacts

- Saved SAE activation rows: 2,146,591
- Observed active features: 13,241
- Filtered features / feature cards: 7,321
- Coactivation pairs saved: 100,000
- Decoder-neighbor rows: 146,420
- Geometry-vs-coactivation rows: 146,420
- Bimodality candidates: 4,535

## Caveats

- Frequencies are top-k frequencies, not true positive activation frequencies
- Coactivation output is capped at the top 100k pairs
- Bimodality candidates are statistical candidates and require manual inspection
- This is a single-layer pilot, not yet a final multi-layer analysis