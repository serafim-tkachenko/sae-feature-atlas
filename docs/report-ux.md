# Report UX design notes

The first workflow is **explore one saved run and inspect feature evidence**.
The design should help a researcher answer: What am I looking at? What does
this value measure? What should I inspect next? How do I keep the evidence?

## Reference interfaces

Reviewed September 29, 2026. These are interface observations and design
inferences, not results from researcher interviews or usability studies.

| Reference | Observed pattern | Application to this report |
| --- | --- | --- |
| [Neuronpedia feature dashboard](https://www.neuronpedia.org/gpt2-small/6-res_scefr-ajt/650) and [feature documentation](https://docs.neuronpedia.org/features) | Local help affordances, token-level context evidence, snippet/full controls, separate explanations and statistics, lists and feature links. | Put a definition and interpretation guidance beside each section. Explain what our highlight actually encodes. Keep provenance, saved evidence, and sharing discoverable. |
| [Anthropic feature browser](https://transformer-circuits.pub/2023/monosemantic-features/vis/a1.html) | A linked reading guide, explicit search scope, and separate search and sorting controls. The feature data did not finish loading during the inspection, so these observations concern its navigation controls. | Provide a quick guide and state that search covers exported strongest contexts. Show active constraints so users can explain why a feature disappeared. |
| [SAE Vis](https://github.com/callummcdougall/sae_vis) and [SAEDashboard](https://github.com/jbloomAus/SAEDashboard) | Their documentation separates feature-centered inspection from prompt-centered analysis and describes activations, logits, and correlation views. SAE Vis references the Anthropic interface above. | Keep this export centered on a selected feature and distinct evidence channels. Prompt experiments require a separate inference workflow and data contract. |

## Decisions implemented

- **Prioritize reading evidence.** Feature links open a compact view with search
  controls collapsed. A persistent desktop list includes example target tokens;
  Previous/Next follows the filtered sort order. Section navigation reduces
  scrolling, while snippet/expanded modes let users control context length.
  Activation bars show a labeled comparison within the current example group.
- **Explain at the point of use.** A short inline hint gives the essential
  meaning; an accessible help dialog adds interpretation and caveats. It works
  by touch and keyboard, without relying on hover. The native modal manages
  focus, Escape, and background interaction.
- **Turn labels into a next step.** For example, decoder similarity invites
  comparing directions and their contexts. Labels remain heuristic cues, not
  semantic explanations. Their precise thresholds belong to the saved analysis
  version; legacy files are not silently reinterpreted.
- **Make filtering inspectable.** Counts and removable chips expose the combined
  effects of search, labels, numeric bounds, chart selection, and saved features.
  Invalid numeric input gets actionable feedback instead of an unexplained
  empty view. The selected evidence stays available outside the filtered list.
- **Explain missing evidence.** Artifact purposes accompany filenames. Help
  distinguishes missing, empty, unreadable, and ready files, as well as unknown
  values versus zero. Legacy frequency retains an explicitly unknown population.
- **Explain saving before it is needed.** A visible sharing guide describes the
  shortlist, browser storage, JSON export, and the need to send the HTML along
  with a local feature link. The empty saved view tells users how to start.

## Data boundaries

The export stores target-position activation values, not complete per-token
activation arrays for each passage. A heatmap of every surrounding token would
imply data we do not have. The target highlight therefore has a fixed color and
an explicit legend. The histogram also keeps its own population description:
stored rows before analysis filtering can differ from summary support.

Likewise, the report does not have recorded logits, causal interventions, or
validated semantic descriptions for every feature. Adding those panels requires
new evidence, not placeholder charts or inferred labels.

## Next usability checks

Ask researchers to find a feature, explain its frequency and support, compare a
neighbor, clear one filter, and export a shortlist. Observe errors and points of
hesitation. Check whether they distinguish target highlighting from a heatmap,
triage from meaning, and missing evidence from an observed zero. Use those
observations to decide whether the next investment should be comparison of two
features, better search, or local notes linked to evidence.
