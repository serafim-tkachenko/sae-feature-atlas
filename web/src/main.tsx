import { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import { Histogram, Scatter, number, numeric, percent } from "./charts";
import type { Example, Feature, Report } from "./types";
import "./style.css";

const human = (text: string) => text.replaceAll("_", " ");
const PAGE_SIZE = 25;
function hashId(): number | null {
  const value = new URLSearchParams(location.hash.slice(1)).get("feature");
  return value !== null && /^\d+$/.test(value) ? Number(value) : null;
}

function Context({ example }: { example: Example }) {
  const center = example.center_token || "";
  return (
    <article className="context-example">
      <div className="context-meta">
        <span>
          {example.source || "Source not recorded"} · text{" "}
          {example.text_id ?? "?"} · token {example.token_pos ?? "?"}
        </span>
        <span className="activation">
          activation {number(example.activation)}
        </span>
      </div>
      <p className="context-text">
        {example.left_context}
        <mark>
          {center.trim()
            ? center
            : `[${center.length ? "whitespace" : "empty token"}]`}
        </mark>
        {example.right_context}
      </p>
      <details className="evidence-reference">
        <summary>Evidence reference</summary>
        <code>{example.evidence_id}</code>
        <p>{example.artifact}</p>
        {example.display_context && (
          <p className="context-text">{example.display_context}</p>
        )}
      </details>
    </article>
  );
}

function FeatureDetail({
  feature,
  report,
  saved,
  onSave,
  onSelect,
  ids,
  notify,
}: {
  feature: Feature;
  report: Report;
  saved: boolean;
  onSave: () => void;
  onSelect: (id: number) => void;
  ids: Set<number>;
  notify: (text: string) => void;
}) {
  const [group, setGroup] = useState<"top" | "low" | "high">("top");
  const [showAll, setShowAll] = useState(false);
  const [copyFallback, setCopyFallback] = useState("");
  const examples = feature.examples[group];
  async function copyLink() {
    const url = new URL(location.href);
    url.hash = `feature=${feature.id}`;
    try {
      await navigator.clipboard.writeText(url.href);
      notify(
        "Feature link copied. Send the HTML report too when sharing a local file.",
      );
    } catch {
      setCopyFallback(url.href);
      notify("Select and copy the feature link below.");
    }
  }
  return (
    <section
      className="feature-detail"
      aria-label={`Feature ${feature.id} evidence`}
    >
      <div className="detail-heading">
        <div>
          <p className="eyebrow">FEATURE EVIDENCE</p>
          <h2>
            Feature <span className="feature-number">{feature.id}</span>
          </h2>
          <p className="muted">
            Triage: {human(feature.label)}
            {feature.priority && ` · ${feature.priority} review priority`}
          </p>
        </div>
        <div className="actions">
          <button onClick={copyLink}>Copy link</button>
          <button
            className={saved ? "primary" : ""}
            aria-pressed={saved}
            onClick={onSave}
          >
            {saved ? "Saved" : "Save feature"}
          </button>
        </div>
      </div>
      {copyFallback && (
        <label className="copy-link">
          Feature link
          <input
            readOnly
            value={copyFallback}
            onFocus={(event) => event.currentTarget.select()}
          />
        </label>
      )}
      <dl className="feature-metrics">
        <div>
          <dt>
            {feature.population === "analysis_activations"
              ? "Eligible-token frequency"
              : "Legacy token frequency"}
          </dt>
          <dd>{percent(feature.frequency)}</dd>
        </div>
        <div>
          <dt>
            {feature.population === "analysis_activations"
              ? "Analysis support"
              : "Recorded support"}
          </dt>
          <dd>{number(feature.support)}</dd>
        </div>
        <div>
          <dt>Texts with support</dt>
          <dd>{number(feature.text_count)}</dd>
        </div>
        <div>
          <dt>p99 activation</dt>
          <dd>{number(feature.p99)}</dd>
        </div>
      </dl>
      <p className="population-note">
        {feature.population === "analysis_activations"
          ? `Frequency denominator: ${number(feature.denominator)} eligible tokens. `
          : "Legacy population: the frequency denominator and eligibility policy are not recorded here. "}
        {report.run.activation_mode === "topk"
          ? "Frequency measures retained top-k membership, not all positive activations."
          : "Values describe the saved collection population."}
      </p>

      <div className="section-heading contexts-heading">
        <div>
          <h3>Activation contexts</h3>
          <p>Read the evidence behind the numbers.</p>
        </div>
      </div>
      <div className="context-tabs" aria-label="Context example group">
        {(["top", "low", "high"] as const).map((value) => (
          <button
            key={value}
            aria-pressed={group === value}
            onClick={() => {
              setGroup(value);
              setShowAll(false);
            }}
          >
            {value === "top"
              ? "Strongest"
              : `${value === "low" ? "Low" : "High"} regime`}{" "}
            <span>{feature.examples[value].length}</span>
          </button>
        ))}
      </div>
      <p className="fine">
        {group === "top"
          ? "Highest-activation saved examples; this is not a representative sample."
          : "Saved component examples. A two-component fit does not establish two meanings."}{" "}
        Up to {String(report.export_policy.examples_per_group)} examples per
        group are included.
      </p>
      {examples.length ? (
        <>
          {(showAll ? examples : examples.slice(0, 4)).map((example, index) => (
            <Context
              key={`${example.evidence_id}-${index}`}
              example={example}
            />
          ))}
          {examples.length > 4 && (
            <button className="show-more" onClick={() => setShowAll(!showAll)}>
              {showAll
                ? "Show fewer contexts"
                : `Show all ${examples.length} contexts`}
            </button>
          )}
        </>
      ) : (
        <div className="empty small">
          No {group === "top" ? "top" : `${group}-regime`} contexts saved for
          this feature. This does not mean the feature never activates.
        </div>
      )}

      <Histogram feature={feature} />
      <div className="neighbors-grid">
        {(["coactivation", "decoder"] as const).map((kind) => (
          <section key={kind} className="neighbor-section">
            <h3>
              {kind === "coactivation"
                ? "Coactivating features"
                : "Decoder neighbors"}
            </h3>
            <p className="fine">
              {kind === "coactivation"
                ? "Same-token overlap · ranked by Jaccard"
                : "Direction similarity · ranked by cosine"}
            </p>
            {feature.neighbors[kind].length ? (
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>{kind === "coactivation" ? "Jaccard" : "Cosine"}</th>
                    {kind === "coactivation" && <th>Support</th>}
                  </tr>
                </thead>
                <tbody>
                  {feature.neighbors[kind].map((neighbor, index) => (
                    <tr key={`${neighbor.id}-${index}`}>
                      <td>
                        {ids.has(neighbor.id) ? (
                          <button
                            className="text-button"
                            onClick={() => onSelect(neighbor.id)}
                          >
                            {neighbor.id} ↗
                          </button>
                        ) : (
                          <span>
                            {neighbor.id} <small>(outside report)</small>
                          </span>
                        )}
                      </td>
                      <td>{number(neighbor.value)}</td>
                      {kind === "coactivation" && (
                        <td>{number(neighbor.support)}</td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className="muted">No saved neighbors.</p>
            )}
            <p className="fine">
              {kind === "coactivation"
                ? "Missing pairs may be unsupported, ineligible, or not retained; absence is not an observed zero."
                : "Similar directions do not establish equivalent meanings or causal effects."}
            </p>
          </section>
        ))}
      </div>
      <details className="diagnostics">
        <summary>Additional diagnostics</summary>
        <dl className="diagnostic-values">
          {Object.entries({
            "Artifact triage score": feature.artifact_score,
            "Interpretability triage score": feature.triage_score,
            ...feature.diagnostics,
          }).map(([key, value]) => (
            <div key={key}>
              <dt>{human(key)}</dt>
              <dd>{number(value)}</dd>
            </div>
          ))}
        </dl>
        <p className="fine">
          Triage scores prioritize manual inspection; they are not semantic
          confidence. PCA and graph alignment are exploratory diagnostics.
        </p>
      </details>
    </section>
  );
}

function RunDetails({ report }: { report: Report }) {
  return (
    <section className="run-details" aria-label="Run details">
      <h2>Run details & evidence availability</h2>
      <dl className="run-values">
        {Object.entries({
          Run: report.run.name,
          Model: report.run.model_name,
          SAE: `${report.run.sae_release ?? "Unknown"} / ${report.run.sae_id ?? "Unknown"}`,
          Hook: report.run.hook_name,
          "Collected token rows": number(report.run.collected_token_rows),
          "Stored-token denominator": number(report.run.stored_tokens),
          "Eligible-token denominator": number(report.run.analysis_tokens),
          "Collection fingerprint": report.run.fingerprints.collection,
          "Analysis fingerprint": report.run.fingerprints.analysis,
          "Source revision": report.run.git.commit_sha,
          "Feature source": report.run.feature_source,
        }).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value == null ? "Not recorded" : String(value)}</dd>
          </div>
        ))}
      </dl>
      <p className="fine">
        Recorded provenance identifies the saved run. It does not independently
        verify that every artifact was regenerated together.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Artifact</th>
              <th>Status</th>
              <th>Rows</th>
            </tr>
          </thead>
          <tbody>
            {report.artifacts.map((item) => (
              <tr key={item.name}>
                <td>
                  {item.name}
                  {item.detail && <p className="error-detail">{item.detail}</p>}
                </td>
                <td>
                  <span className={`status ${item.status}`}>{item.status}</span>
                </td>
                <td>{item.rows === null ? "—" : number(item.rows)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {Object.keys(report.coactivation_metadata).length > 0 && (
        <details>
          <summary>Coactivation support and retention policy</summary>
          <pre>{JSON.stringify(report.coactivation_metadata, null, 2)}</pre>
        </details>
      )}
      {Object.keys(report.links).length > 0 && (
        <div className="resource-links">
          {Object.entries(report.links).map(([name, url]) => (
            <a key={name} href={url}>
              {name} ↗
            </a>
          ))}
        </div>
      )}
    </section>
  );
}

function App({ report }: { report: Report }) {
  const features = report.features;
  const ids = useMemo(() => new Set(features.map((f) => f.id)), [features]);
  const [selected, setSelected] = useState<number | null>(
    hashId() ?? features[0]?.id ?? null,
  );
  const [query, setQuery] = useState("");
  const [label, setLabel] = useState("all");
  const [sort, setSort] = useState("support");
  const [minSupport, setMinSupport] = useState("");
  const [minFrequency, setMinFrequency] = useState("");
  const [maxFrequency, setMaxFrequency] = useState("");
  const [brushed, setBrushed] = useState<number[] | null>(null);
  const brushIds = useMemo(
    () => (brushed === null ? null : new Set(brushed)),
    [brushed],
  );
  const [savedOnly, setSavedOnly] = useState(false);
  const [page, setPage] = useState(0);
  const [details, setDetails] = useState(false);
  const [message, setMessage] = useState("");
  const storageKey = `atlas-saved:${report.run.fingerprints.analysis || `${report.run.name}:${report.run.sae_id}`}`;
  const [saved, setSaved] = useState<number[]>(() => {
    try {
      const stored: unknown = JSON.parse(
        localStorage.getItem(storageKey) || "[]",
      );
      return Array.isArray(stored)
        ? stored.filter((id) => typeof id === "number" && ids.has(id))
        : [];
    } catch {
      return [];
    }
  });
  useEffect(() => {
    const update = () => setSelected(hashId() ?? features[0]?.id ?? null);
    addEventListener("hashchange", update);
    return () => removeEventListener("hashchange", update);
  }, [features]);
  useEffect(() => {
    setPage(0);
  }, [
    query,
    label,
    minSupport,
    minFrequency,
    maxFrequency,
    brushed,
    savedOnly,
    sort,
  ]);
  const searchText = useMemo(
    () =>
      new Map(
        features.map((f) => [
          f.id,
          `${f.id} ${human(f.label)} ${f.examples.top.map((e) => `${e.left_context || ""}${e.center_token || ""}${e.right_context || ""}`).join(" ")}`.toLowerCase(),
        ]),
      ),
    [features],
  );
  const matching = useMemo(
    () =>
      features.filter((f) => {
        if (savedOnly && !saved.includes(f.id)) return false;
        if (label !== "all" && f.label !== label) return false;
        if (
          minSupport !== "" &&
          (!numeric(f.support) || f.support < Number(minSupport))
        )
          return false;
        if (
          minFrequency !== "" &&
          (!numeric(f.frequency) || f.frequency * 100 < Number(minFrequency))
        )
          return false;
        if (
          maxFrequency !== "" &&
          (!numeric(f.frequency) || f.frequency * 100 > Number(maxFrequency))
        )
          return false;
        if (/^\d+$/.test(query.trim())) return f.id === Number(query.trim());
        return (
          !query.trim() ||
          searchText.get(f.id)!.includes(query.trim().toLowerCase())
        );
      }),
    [
      features,
      savedOnly,
      saved,
      label,
      minSupport,
      minFrequency,
      maxFrequency,
      query,
      searchText,
    ],
  );
  const visible = useMemo(
    () =>
      matching
        .filter((f) => brushIds === null || brushIds.has(f.id))
        .sort((a, b) => {
          if (sort === "id") return a.id - b.id;
          const field = sort as "frequency" | "p99" | "support";
          return (
            (b[field] ?? -Infinity) - (a[field] ?? -Infinity) || a.id - b.id
          );
        }),
    [matching, brushIds, sort],
  );
  const feature = features.find((f) => f.id === selected);
  const lastPage = Math.max(0, Math.ceil(visible.length / PAGE_SIZE) - 1);
  const currentPage = Math.min(page, lastPage);
  function selectFeature(id: number, reveal = false) {
    setSelected(id);
    location.hash = `feature=${id}`;
    if (reveal) {
      resetFilters();
      setMessage(`Opened related feature ${id}. Filters cleared.`);
      requestAnimationFrame(() =>
        document
          .querySelector(".evidence-panel")
          ?.scrollIntoView({ block: "start" }),
      );
    }
  }
  function resetFilters() {
    setQuery("");
    setLabel("all");
    setMinSupport("");
    setMinFrequency("");
    setMaxFrequency("");
    setBrushed(null);
    setSavedOnly(false);
  }
  function toggleSave(id: number) {
    const next = saved.includes(id)
      ? saved.filter((value) => value !== id)
      : [...saved, id];
    setSaved(next);
    try {
      localStorage.setItem(storageKey, JSON.stringify(next));
      setMessage(
        next.includes(id)
          ? `Feature ${id} saved in this browser.`
          : `Feature ${id} removed from selection.`,
      );
    } catch {
      setMessage(
        "Selection is available for this session. Export it to keep a copy.",
      );
    }
  }
  function exportSelection() {
    const data = {
      schema_version: report.schema_version,
      kind: "sae-atlas-evidence-selection",
      run: report.run,
      warnings: report.warnings,
      artifacts: report.artifacts,
      export_policy: report.export_policy,
      features: features.filter((f) => saved.includes(f.id)),
      coactivation_metadata: report.coactivation_metadata,
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
    );
    const a = document.createElement("a");
    a.href = url;
    a.download = `${report.run.name}-selection.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setMessage(
      `Exported ${saved.length} features with evidence and provenance.`,
    );
  }
  return (
    <>
      <header className="app-header">
        <div className="brand">
          <span className="brand-symbol" aria-hidden="true">
            A
          </span>
          <div>
            Feature Atlas<span>SAE RESEARCH TOOLKIT</span>
          </div>
        </div>
        <div className="header-actions">
          <button aria-pressed={details} onClick={() => setDetails(!details)}>
            Run details
          </button>
          <button
            aria-pressed={savedOnly}
            onClick={() => setSavedOnly(!savedOnly)}
          >
            Saved ({saved.length})
          </button>
          <button
            className="primary"
            disabled={!saved.length}
            onClick={exportSelection}
          >
            Export selection
          </button>
        </div>
      </header>
      <main>
        <div className="run-heading">
          <div>
            <p className="eyebrow">SINGLE-RUN EXPLORER</p>
            <h1>Explore features</h1>
            <p className="run-name">{report.run.name}</p>
          </div>
          <span
            className={`provenance ${report.run.provenance === "recorded" ? "" : "unverified"}`}
          >
            {report.run.provenance === "recorded"
              ? "Provenance recorded"
              : "Unverified / legacy run"}
          </span>
        </div>
        <div className="run-strip">
          <span>
            <b>{report.run.model_name || "Model not recorded"}</b>
          </span>
          <span>
            Layer {report.run.layer ?? "?"} ·{" "}
            {report.run.corpus || "Corpus unknown"}
          </span>
          <span>{features.length.toLocaleString()} features</span>
          <span>
            {report.run.activation_mode === "topk"
              ? `Top-${report.run.top_k ?? "?"} retained activations`
              : report.run.activation_mode === "positive"
                ? "Positive activations"
                : "Storage mode unknown"}
          </span>
        </div>
        {report.warnings.length > 0 && (
          <div className="notice">
            {report.warnings.map((text) => (
              <p key={text}>{text}</p>
            ))}
          </div>
        )}
        {report.artifacts.some((a) => a.status === "unreadable") && (
          <div className="notice">
            Some artifacts are unreadable.{" "}
            <button className="text-button" onClick={() => setDetails(true)}>
              View evidence availability
            </button>
          </div>
        )}
        {details && <RunDetails report={report} />}
        <div className="explore-overview">
          <div className="search-panel">
            <h2>Find a feature</h2>
            <label>
              Feature ID, triage label, or saved context
              <input
                type="search"
                value={query}
                placeholder="Search features and strongest contexts…"
                onChange={(e) => setQuery(e.target.value)}
              />
            </label>
            <div className="two-fields">
              <label>
                Triage label
                <select
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                >
                  <option value="all">All labels</option>
                  {[...new Set(features.map((f) => f.label))]
                    .sort()
                    .map((value) => (
                      <option key={value} value={value}>
                        {human(value)}
                      </option>
                    ))}
                </select>
              </label>
              <label>
                Minimum support
                <input
                  type="number"
                  min="0"
                  step="1"
                  value={minSupport}
                  placeholder="Any"
                  onChange={(e) => setMinSupport(e.target.value)}
                />
              </label>
            </div>
            <div className="two-fields">
              <label>
                Minimum frequency (%)
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="any"
                  value={minFrequency}
                  placeholder="0"
                  onChange={(e) => setMinFrequency(e.target.value)}
                />
              </label>
              <label>
                Maximum frequency (%)
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="any"
                  value={maxFrequency}
                  placeholder="100"
                  onChange={(e) => setMaxFrequency(e.target.value)}
                />
              </label>
            </div>
            <div className="filter-actions">
              <button onClick={resetFilters}>Reset filters</button>
              <button
                disabled={!visible.length}
                onClick={() =>
                  selectFeature(
                    visible[Math.floor(Math.random() * visible.length)].id,
                  )
                }
              >
                Random feature
              </button>
            </div>
            <p className="fine">
              Labels are inspection cues, not explanations. Search covers
              exported strongest contexts only.
            </p>
          </div>
          <Scatter
            features={matching}
            selected={selected}
            brushed={brushed}
            onSelect={selectFeature}
            onBrush={setBrushed}
          />
        </div>
        <div className="workspace">
          <aside className="feature-browser" aria-label="Feature browser">
            <div className="browser-heading">
              <h2>
                Features <span>({visible.length.toLocaleString()})</span>
              </h2>
              <label className="sort-label">
                Sort by
                <select value={sort} onChange={(e) => setSort(e.target.value)}>
                  <option value="support">Support</option>
                  <option value="frequency">Frequency</option>
                  <option value="p99">p99 activation</option>
                  <option value="id">Feature ID</option>
                </select>
              </label>
            </div>
            <div className="feature-list" aria-label="Matching features">
              {visible
                .slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE)
                .map((f) => (
                  <button
                    key={f.id}
                    className={`feature-row ${f.id === selected ? "active" : ""}`}
                    aria-pressed={f.id === selected}
                    onClick={() => selectFeature(f.id)}
                  >
                    <span className="row-top">
                      <b>#{f.id}</b>
                      <span>{percent(f.frequency)}</span>
                    </span>
                    <span className="row-bottom">
                      {human(f.label)}
                      {saved.includes(f.id) && (
                        <span className="saved-dot"> · saved</span>
                      )}
                    </span>
                  </button>
                ))}
            </div>
            {!visible.length && (
              <div className="empty small">
                <p>
                  {features.length
                    ? "No features match these filters."
                    : "No feature tables are available. Run the features stage to generate them."}
                </p>
                {features.length > 0 && (
                  <button onClick={resetFilters}>Reset filters</button>
                )}
              </div>
            )}
            {visible.length > PAGE_SIZE && (
              <div className="pagination">
                <button
                  disabled={!currentPage}
                  onClick={() => setPage(currentPage - 1)}
                >
                  Previous
                </button>
                <span>
                  {currentPage + 1} / {lastPage + 1}
                </span>
                <button
                  disabled={currentPage === lastPage}
                  onClick={() => setPage(currentPage + 1)}
                >
                  Next
                </button>
              </div>
            )}
          </aside>
          <div className="evidence-panel">
            {feature && !visible.some((f) => f.id === feature.id) && (
              <p className="outside-filter">
                The selected feature is outside the current filters.{" "}
                <button className="text-button" onClick={resetFilters}>
                  Reset filters
                </button>
              </p>
            )}
            {feature ? (
              <FeatureDetail
                key={feature.id}
                feature={feature}
                report={report}
                saved={saved.includes(feature.id)}
                onSave={() => toggleSave(feature.id)}
                onSelect={(id) => selectFeature(id, true)}
                ids={ids}
                notify={setMessage}
              />
            ) : (
              <div className="empty">
                <h2>
                  {selected !== null
                    ? `Feature ${selected} is not in this report`
                    : "No feature selected"}
                </h2>
                <p>Choose a feature from the list to inspect saved evidence.</p>
              </div>
            )}
          </div>
        </div>
        <footer>
          <span>Feature Atlas · portable research report</span>
          <span>
            Evidence comes from saved artifacts. No model inference runs in this
            view.
          </span>
        </footer>
      </main>
      <div
        role="status"
        aria-live="polite"
        className={`toast ${message ? "visible" : ""}`}
      >
        {message}
        {message && (
          <button onClick={() => setMessage("")} aria-label="Dismiss message">
            ×
          </button>
        )}
      </div>
    </>
  );
}

const root = createRoot(document.getElementById("root")!);
try {
  const report: Report = JSON.parse(
    document.getElementById("atlas-data")!.textContent!,
  );
  if (report.schema_version !== 1 || !Array.isArray(report.features))
    throw new Error("Unsupported report schema");
  root.render(<App report={report} />);
} catch {
  root.render(
    <main>
      <h1>Unable to open this report</h1>
      <p>
        The embedded report data is invalid or uses an unsupported version.
        Regenerate it with the matching SAE Feature Atlas version.
      </p>
    </main>,
  );
}
