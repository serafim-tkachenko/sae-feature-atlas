import { createContext, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";

const topics = {
  filters: {
    title: "Finding and filtering features",
    description:
      "A numeric search finds an exact feature ID. Other searches match triage labels and text in the exported strongest contexts, ignoring case.",
    points: [
      "Start with a word or a label, then narrow by support or frequency. Enter percentages directly: 1 means 1%, not 100%.",
      "All filters combine, including a chart selection and Saved. Remove an individual filter using its chip, or use Reset filters to see the whole report again.",
      "Sort by support, frequency, or p99 to put larger recorded values first; Feature ID sorts from smallest to largest. Random feature samples from the matching list.",
    ],
    note: "Search does not run the model or search the full corpus. A previously selected feature stays open even when it falls outside your filters.",
  },
  guide: {
    title: "A quick guide to this report",
    description:
      "Explore one saved run, form a hypothesis about a feature, and keep the evidence you want to investigate further.",
    points: [
      "Find: search a feature ID, a triage label, or words in the saved strongest contexts. All active filters combine.",
      "Inspect: read several highlighted tokens and their surrounding text. Look for recurring patterns and examples that challenge your hypothesis.",
      "Compare: check low/high activation groups when available, then follow coactivating features or similar decoder directions.",
      "Keep: use Save feature to build a shortlist. Export selection downloads its evidence and run metadata as JSON; Copy link points to a feature in this HTML report.",
    ],
    note: "A feature is one entry in an SAE dictionary. Its number identifies it only within that SAE. This report shows saved evidence; testing a new prompt requires a separate model run.",
  },
  frequency: {
    title: "Eligible-token frequency",
    description:
      "The share of eligible token positions where this feature has a retained analysis activation. For example, 1% means about 1 in 100 eligible tokens.",
    points: [
      "Use frequency to distinguish rare patterns from broadly occurring ones. Check support and contexts before interpreting a rare feature.",
      "With top-k storage, a feature counts only when it is among the retained features at that token. Positive activations outside that selection are not counted.",
    ],
    note: "The denominator is shown below the metrics. Frequencies from different eligibility or storage policies are not directly comparable.",
  },
  legacyFrequency: {
    title: "Legacy token frequency",
    description:
      "This percentage was saved by an older analysis. Its denominator and token eligibility policy are not recorded in the feature evidence.",
    points: [
      "Use it as a rough within-run navigation cue. Do not assume it uses the current eligible-token definition.",
      "Consult the original configuration and artifacts before comparing it with a newer run.",
    ],
    note: "Viewing this report does not migrate or recompute legacy statistics.",
  },
  support: {
    title: "Support and texts",
    description:
      "Support counts saved activation rows for this feature in the summary population. Texts with support counts distinct text IDs represented by those rows.",
    points: [
      "One text can contribute many token positions, so support can be much larger than the text count.",
      "A minimum support filter helps you find features with more observations. It does not guarantee diverse sources or independent evidence.",
    ],
    note: "Analysis support uses eligible analysis rows. Recorded support in a legacy run may use a different population. Neither is the number of context examples embedded here.",
  },
  activation: {
    title: "p99 activation",
    description:
      "The 99th percentile of this feature’s recorded activation values in the summary population: roughly 99% of those values are at or below it.",
    points: [
      "Use it to locate features with a strong upper tail, then inspect their actual contexts.",
      "Activation is the SAE coefficient at a token position. It is not a probability, confidence score, or measure of causal importance.",
    ],
    note: "Scales depend on the SAE and collection policy. Missing values mean not recorded, not zero.",
  },
  labels: {
    title: "Triage labels and review priority",
    description:
      "These automated cues suggest what to inspect. They do not explain what a feature means.",
    points: [
      "High frequency and high intensity describe relative frequency or activation strength. Rare high intensity combines low frequency with strong activations.",
      "Decoder geometry dense points to strong direction similarity; coactivation hub points to strong token overlap. These labels alone do not establish how many neighbors there are.",
      "Likely artifact flags patterns such as formatting, punctuation, or token position. Bimodal candidate flags a distribution worth checking. Manual review is a suggestion to inspect the feature.",
    ],
    note: "Rules and thresholds depend on the saved analysis version. Review priority and triage scores are not semantic confidence.",
  },
  contexts: {
    title: "Reading activation contexts",
    description:
      "The green highlight marks the target token whose activation is recorded. A token can be a word, part of a word, punctuation, or whitespace.",
    points: [
      "The activation number belongs to that target position. Surrounding text provides context; its unhighlighted tokens do not imply zero activation.",
      "Snippet shortens the surrounding text and collapses line breaks. Expanded shows all saved left/right context. The activation bar compares the target value with the strongest saved value in the current group; it is not a percentile.",
      "Strongest shows saved high-activation examples, not a random or representative sample. Compare several texts and look for counterexamples.",
      "Low and high regime examples come from lower- and higher-mean components of a fitted activation model. A two-component fit does not establish two meanings.",
      "Open Evidence reference for the text ID, position, source artifact, and any additional saved context.",
    ],
    note: "Highlight color identifies the target; it does not encode activation strength. The count on each group is the number of examples included in this report.",
  },
  landscape: {
    title: "Reading the feature landscape",
    description:
      "Each point is a feature. Farther right means more frequent; higher means a larger p99 activation. Neither direction means better quality.",
    points: [
      "The vertical position uses log(1 + p99) to fit both small and large values. Tick labels show the original activation values.",
      "Click a point to inspect it. Drag a rectangle to restrict the feature list; numeric filters and search still apply. Clear chart selection removes that rectangle’s filter.",
      "The orange point is the selected feature. Faded points fall outside the chart selection. Overlapping features can share a position; use the list to inspect them individually.",
    ],
    note: "The chart follows search and other filters. Features missing either plotted value are omitted. The list and numeric filters provide keyboard alternatives to dragging.",
  },
  histogram: {
    title: "Reading the activation distribution",
    description:
      "Each bar counts stored activation values in a magnitude range. Taller bars mean more saved values in that range; farther right means stronger activations.",
    points: [
      "Use the shape to spot concentration, a long tail, or possible separated groups. Shape alone does not establish distinct meanings.",
      "This histogram uses finite stored rows before analysis filtering. Its row count can differ from the support shown above.",
    ],
    note: "Under top-k storage, omitted activations are not zeros. This is not a distribution over every token in the corpus.",
  },
  coactivation: {
    title: "Coactivation, Jaccard, and pair support",
    description:
      "Coactivating features have retained activations at the same token positions. Jaccard measures overlap: shared positions divided by positions where either feature is present.",
    points: [
      "Jaccard ranges from 0 to 1. A value near 1 means the retained token sets overlap strongly; it does not mean the features have the same meaning.",
      "Pair support counts shared token positions. Read it alongside Jaccard: a large overlap score based on very few observations may be fragile.",
      "Open a neighbor to compare its contexts. Missing pairs may have been excluded by eligibility, support thresholds, or export limits.",
    ],
    note: "Coactivation is association, not causation. The saved support and retention policy is available in Run details.",
  },
  decoder: {
    title: "Decoder neighbors and cosine similarity",
    description:
      "A decoder direction is the vector an SAE feature contributes when reconstructing a model activation. Cosine compares the directions of two such vectors.",
    points: [
      "Cosine ranges from −1 to 1: near 1 means aligned, near 0 means orthogonal, and near −1 means opposite.",
      "Use neighbors to look for related or redundant directions, then compare their activation contexts. Similar directions need not activate on the same text.",
    ],
    note: "Direction similarity does not establish equivalent meanings or causal effects. Outside report means that neighbor has no exported feature card here.",
  },
  saved: {
    title: "Saving and sharing evidence",
    description:
      "Save feature adds a feature to your shortlist in this browser. Saved restricts the list to that shortlist and combines with your other filters.",
    points: [
      "Save at least one feature to enable Export selection. The JSON download contains selected features, their saved evidence, and run provenance.",
      "Browser storage may be unavailable or cleared. Export your selection to keep a portable record.",
      "Copy link includes the selected feature ID. To share a local file link, send the HTML report as well; the recipient cannot access your local file automatically.",
    ],
    note: "Saving changes your local selection only. It does not alter the report’s scientific data or create an interpretation label.",
  },
  availability: {
    title: "Evidence availability and provenance",
    description:
      "Run details shows which saved files supplied this report. Availability is a description of files, not a quality score for the analysis.",
    points: [
      "Ready: a readable artifact is available. Empty: the artifact exists but contains no rows or entries. Missing: the file was not found. Unreadable: it could not be read or lacks required columns.",
      "Missing optional artifacts are normal for partial runs and alternative storage modes. Not recorded means no usable value was exported; it does not mean zero.",
      "Fingerprints identify collection and analysis configurations. The source revision identifies the code commit when recorded.",
    ],
    note: "Recorded provenance does not independently prove that all files were regenerated together. Legacy runs lack enough metadata to verify those relationships.",
  },
  diagnostics: {
    title: "Additional diagnostics",
    description:
      "These values help prioritize follow-up analysis. None is a validated explanation of a feature’s meaning.",
    points: [
      "Artifact and interpretability triage scores summarize inspection heuristics. Higher interpretability triage suggests closer inspection, not a confirmed semantic concept.",
      "Bimodality score is the saved BIC improvement of a two-component over a one-component activation fit. Support, component weights, and separation also matter.",
      "Observed PC mass measures decoder mass captured by the fitted principal-component subspace. Effective PC dimension describes how spread that mass is across components.",
      "Graph alignment at 10 compares local neighborhoods from geometry and coactivation. It depends on the graphs and their support policies.",
    ],
    note: "PCA and graph alignment are exploratory diagnostics; neither is a reconstruction-quality or semantic-confidence score.",
  },
} as const;

type Topic = keyof typeof topics;
type Help = { topic: Topic; extra?: string };
const HelpContext = createContext<(help: Help) => void>(() => {});

export function HelpProvider({ children }: { children: ReactNode }) {
  const [help, setHelp] = useState<Help | null>(null);
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (help && !dialog.current?.open) dialog.current?.showModal();
  }, [help]);
  const content = help ? topics[help.topic] : null;
  return (
    <HelpContext.Provider value={setHelp}>
      {children}
      <dialog
        ref={dialog}
        className="help-dialog"
        aria-labelledby="help-title"
        aria-describedby="help-description"
        onClose={() => setHelp(null)}
        onClick={(event) => {
          if (event.target === event.currentTarget) dialog.current?.close();
        }}
      >
        {content && (
          <div className="help-content">
            <div className="section-heading">
              <p className="eyebrow">REPORT GUIDE</p>
              <button
                autoFocus
                onClick={() => dialog.current?.close()}
                aria-label="Close help"
              >
                Close ×
              </button>
            </div>
            <h2 id="help-title">{content.title}</h2>
            <p id="help-description">{content.description}</p>
            {help?.extra && <p className="help-current">{help.extra}</p>}
            <ul>
              {content.points.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
            <p className="help-note">{content.note}</p>
          </div>
        )}
      </dialog>
    </HelpContext.Provider>
  );
}

export function HelpButton({
  topic,
  children,
  extra,
}: {
  topic: Topic;
  children?: ReactNode;
  extra?: string;
}) {
  const open = useContext(HelpContext);
  return (
    <button
      type="button"
      className={children ? "help-link" : "help-button"}
      aria-label={
        children ? undefined : `About ${topics[topic].title.toLowerCase()}`
      }
      aria-haspopup="dialog"
      onClick={() => open({ topic, extra })}
    >
      {children ?? "?"}
    </button>
  );
}

export const labelHint = (label: string) =>
  ({
    decoder_geometry_dense:
      "Strong decoder similarity suggests related directions to compare.",
    coactivation_hub:
      "Strong token overlap suggests coactivating features to compare.",
    likely_artifact:
      "Inspect whether formatting, punctuation, or position explains the pattern.",
    bimodal_candidate:
      "Inspect the activation distribution and compare the saved regimes.",
    high_frequency:
      "Frequently retained in this run; check how diverse the contexts are.",
    high_intensity:
      "Strong upper-tail activations; inspect which contexts produce them.",
    rare_high_intensity:
      "Infrequent but strong activations; check the amount and diversity of evidence.",
    rare_feature:
      "Infrequently retained; check support before drawing conclusions.",
    manual_review:
      "The saved heuristics flag this feature for closer inspection.",
    unlabeled: "No triage cue was assigned. Start with the saved contexts.",
  })[label] ??
  "An automated inspection cue from the saved analysis; check the contexts.";

export const diagnosticNames: Record<string, string> = {
  bimodality_score: "Two-component fit improvement (BIC)",
  pc_mass_observed: "Decoder mass in observed PC subspace",
  effective_pc_dim: "Effective PC dimension",
  gca_at_10: "Graph alignment at 10 neighbors",
};

export const artifactDescriptions: Record<string, string> = {
  "feature_cards.parquet": "Feature summaries and triage cues",
  "analysis_features.parquet": "Analysis feature summaries (fallback)",
  "feature_stats.parquet": "Collection feature statistics (fallback)",
  "top_feature_examples.parquet": "Strongest activation contexts",
  "bimodal_peak_examples.parquet": "Low and high regime contexts",
  "decoder_neighbors.parquet": "Similar decoder directions",
  "coactivation_pairs.parquet": "Same-token feature overlap",
  "token_metadata.parquet": "Token, text, and source references",
  "sae_activations_topk.parquet":
    "Retained top-k activations for distributions",
  "sae_activations_positive.parquet": "Positive activations for distributions",
  "lineage.json": "Run identity and provenance",
  "coactivation_metadata.json": "Pair support and retention policy",
};
