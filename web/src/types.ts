export interface Example {
  evidence_id: string;
  artifact: string;
  activation: number | null;
  text_id: number | null;
  token_pos: number | null;
  source: string | null;
  left_context: string | null;
  center_token: string | null;
  right_context: string | null;
  display_context: string | null;
  activation_population: string | null;
}
export interface Neighbor {
  id: number;
  value: number | null;
  support: number | null;
  pmi: number | null;
}
export interface Feature {
  id: number;
  label: string;
  priority: string | null;
  frequency: number | null;
  population: string;
  support: number | null;
  text_count: number | null;
  denominator: number | null;
  stored_frequency: number | null;
  stored_support: number | null;
  p99: number | null;
  median: number | null;
  artifact_score: number | null;
  triage_score: number | null;
  examples: Record<"top" | "low" | "high", Example[]>;
  neighbors: Record<"decoder" | "coactivation", Neighbor[]>;
  histogram: {
    edges: number[];
    counts: number[];
    count: number;
    population: string;
  } | null;
  diagnostics: Record<string, number | null>;
}
export interface Report {
  schema_version: number;
  run: {
    name: string;
    provenance: string;
    artifact_schema: number | null;
    fingerprints: Record<string, string>;
    git: Record<string, unknown>;
    model_name: string | null;
    sae_release: string | null;
    sae_id: string | null;
    layer: number | null;
    hook_name: string | null;
    corpus: string | null;
    activation_mode: string | null;
    top_k: number | null;
    stored_tokens: number | null;
    analysis_tokens: number | null;
    collected_token_rows: number | null;
    collected_texts?: number | null;
    reanalysis_source?: string | null;
    feature_source: string;
  };
  warnings: string[];
  artifacts: {
    name: string;
    status: string;
    rows: number | null;
    detail?: string;
  }[];
  features: Feature[];
  export_policy: Record<string, unknown>;
  coactivation_metadata: Record<string, unknown>;
  links: Record<string, string>;
}
