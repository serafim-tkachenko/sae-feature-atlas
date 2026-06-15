# SAE Feature Atlas report: `gemma31bpt_l13_res16k_pile10k_top32`

## Setup
- Model: `google/gemma-3-1b-pt`
- SAE: `gemma-scope-2-1b-pt-res` / `layer_13_width_16k_l0_medium`
- Layer/hook: `13` / `blocks.13.hook_resid_post`
- Corpus: `pile-10k`
- Activation mode: `topk`
- Top-k: `32`

## Artifacts
- Feature cards: `(7321, 60)`
- Feature stats: `(13241, 10)`
- Top examples: `(230058, 10)`
- Coactivation pairs: `(100000, 9)`
- Decoder neighbors: `(146420, 4)`
- Geometry vs coactivation: `(146420, 14)`
- Bimodal candidates: `(4535, 11)`
- Inspection feature summaries: `(7321, 18)`
- Inspection pair summaries: `(30, 12)`

## Interpretation caveats
- Frequencies depend on activation storage mode.
- In top-k mode, frequency means occurrence among saved top-k features.
- Automated inspection labels are heuristic triage, not final explanations.
- Bimodality scores are statistical candidates and require manual validation.
