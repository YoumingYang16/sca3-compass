import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowUpRight,
  Search,
  FileText,
  FlaskConical,
  Download,
  BookOpen,
  Layers3,
} from "lucide-react";
import {
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { researchApi } from "./research-api";
import "./literature.css";

type Article = {
  pmcid: string;
  title: string;
  year: string;
  domain: string;
  design: string;
  population: string;
  boundary: string;
  source_url: string;
  doi: string;
  authors: string[];
  license: string;
  chunks: number;
  snapshot: { sha256: string; url: string };
};
type Passage = {
  id: string;
  pmcid: string;
  section: string;
  heading: string;
  text: string;
  text_sha256: string;
  bioc_offset: number;
  passage_ordinal: number;
  start: number;
  end: number;
  article: Article;
  ranks?: Record<string, number | null>;
  scores?: Record<string, number>;
};
type Point = {
  id: string;
  pmcid: string;
  section: string;
  x: number;
  y: number;
};
type Summary = {
  sha256: string;
  selection: string;
  articles: Article[];
  chunk_count: number;
  semantic_ready: boolean;
  model: {
    model_id: string;
    revision: string;
    device: string;
    dimensions: number;
  };
  verification: { exact_spans_checked: number };
  projection: {
    method: string;
    explained_variance: number[];
    points: Point[];
  } | null;
  not_indexed: Array<{ pmcid: string; reason: string }>;
};
type SearchResult = {
  query: string;
  method: string;
  action: string;
  results: Passage[];
  boundary: string;
  comparison: Record<
    string,
    Array<{ id: string; pmcid: string; section: string }>
  >;
};
type Metric = {
  estimate: number;
  low: number;
  high: number;
  difference_vs_bm25: number;
  difference_low: number;
  difference_high: number;
};
type Benchmark = {
  status: string;
  query_count: number;
  intent_clusters: number;
  purpose: string;
  sha256: string;
  aggregate: Record<string, Record<string, Metric>>;
  by_language: Record<string, Record<string, Record<string, number>>>;
  case_results: Array<{
    intent: string;
    language: string;
    query: string;
    relevant: string[];
    ranked: Record<string, string[]>;
    metrics: Record<string, Record<string, number>>;
  }>;
};
const methodNames: Record<string, string> = {
  bm25: "BM25",
  dense: "Neural",
  rrf: "Hybrid RRF",
  diverse: "Hybrid + MMR",
};
const colors = [
  "#126b77",
  "#2866b2",
  "#8a5aa5",
  "#b4742e",
  "#456e48",
  "#9b4e69",
];
const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
const examples = [
  "中国大陆SCA3队列的SARA年进展速度是多少？",
  "How many human donors were used in spinal cord transcriptomics?",
  "How is detectable change different from meaningful change?",
];

function PassageReader({ passage }: { passage: Passage | null }) {
  if (!passage)
    return (
      <aside className="lit-reader lit-reader-empty">
        <FileText size={32} />
        <h3>Select a passage</h3>
        <p>
          Search the literature, or select a point in the map to inspect its
          exact source text.
        </p>
      </aside>
    );
  return (
    <aside className="lit-reader" aria-label="Original passage reader">
      <div className="lit-reader-top">
        <span className="lit-eyebrow">SOURCE INSPECTION</span>
        <a href={passage.article.source_url} target="_blank" rel="noreferrer">
          Original article <ArrowUpRight size={16} />
        </a>
      </div>
      <h3>{passage.article.title}</h3>
      <p className="lit-citation">
        {passage.article.authors[0]} et al. · {passage.article.year} ·{" "}
        {passage.pmcid}
      </p>
      <div className="lit-pills">
        <span>{passage.article.design}</span>
        <span>{passage.section}</span>
      </div>
      <h4>{passage.heading || "Original passage"}</h4>
      <blockquote lang="en">{passage.text}</blockquote>
      <div className="lit-limit">
        <span>INTERPRETATION LIMIT</span>
        <p>{passage.article.boundary}</p>
      </div>
      <div className="lit-provenance">
        <span>EXACT-SPAN PROVENANCE</span>
        <dl>
          <dt>Population</dt>
          <dd>{passage.article.population}</dd>
          <dt>BioC offset</dt>
          <dd>{passage.bioc_offset} characters</dd>
          <dt>Source span</dt>
          <dd>
            Passage {passage.passage_ordinal}, characters {passage.start}–
            {passage.end}
          </dd>
          <dt>Text SHA-256</dt>
          <dd>
            <code>{passage.text_sha256}</code>
          </dd>
        </dl>
      </div>
      <details>
        <summary>Attribution & reuse conditions</summary>
        <p>{passage.article.license}</p>
        <p>DOI: {passage.article.doi}</p>
        <code>Snapshot {passage.article.snapshot.sha256}</code>
      </details>
      <a className="lit-learning-link" href="#learning">
        <BookOpen size={16} /> Practice distinguishing evidence from inference{" "}
        <ArrowUpRight size={15} />
      </a>
    </aside>
  );
}

function EvidenceMap({
  summary,
  selected,
  retrieved,
  onSelect,
}: {
  summary: Summary;
  selected: string | null;
  retrieved: string[];
  onSelect: (id: string) => void;
}) {
  const [filter, setFilter] = useState("all");
  const [section, setSection] = useState("all");
  const projection = summary.projection;
  const points = projection?.points ?? [];
  const filtered = points.filter(
    (p) =>
      (filter === "all" || p.pmcid === filter) &&
      (section === "all" || p.section === section),
  );
  const shown = new Set(filtered.map((p) => p.id));
  const pick = (entry: Point | { payload?: Point }) => {
    const value = "id" in entry ? entry : entry.payload;
    if (value?.id) onSelect(value.id);
  };
  return (
    <section className="lit-map-card">
      <div className="lit-panel-heading">
        <div>
          <span className="lit-eyebrow">CORPUS GEOMETRY</span>
          <h2>Evidence space</h2>
        </div>
        <label className="lit-select">
          Section
          <select value={section} onChange={(e) => setSection(e.target.value)}>
            <option value="all">All sections</option>
            {[...new Set(points.map((p) => p.section))].sort().map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
        </label>
      </div>
      {projection ? (
        <>
          <div className="lit-map">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart
                margin={{ top: 20, right: 20, bottom: 18, left: 4 }}
              >
                <CartesianGrid stroke="#dce5e9" strokeDasharray="3 5" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="PC1"
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v) => Number(v).toFixed(1)}
                  label={{
                    value: `PC1 · ${pct(projection.explained_variance[0])} variance`,
                    position: "insideBottom",
                    offset: -12,
                    fontSize: 12,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="PC2"
                  width={44}
                  tick={{ fontSize: 12 }}
                  tickFormatter={(v) => Number(v).toFixed(1)}
                />
                <ReferenceLine x={0} stroke="#aabfc8" />
                <ReferenceLine y={0} stroke="#aabfc8" />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  content={({ active, payload }) => {
                    const p = payload?.[0]?.payload as Point | undefined;
                    return active && p ? (
                      <div className="lit-map-tooltip">
                        <strong>{p.pmcid}</strong>
                        <span>
                          {p.section} · {p.id}
                        </span>
                        <span>Select to inspect original text</span>
                      </div>
                    ) : null;
                  }}
                />
                {summary.articles.map((a, i) => (
                  <Scatter
                    key={a.pmcid}
                    name={a.domain}
                    data={filtered.filter((p) => p.pmcid === a.pmcid)}
                    fill={colors[i % colors.length]}
                    fillOpacity={0.65}
                    onClick={pick}
                  />
                ))}
                <Scatter
                  name="Retrieved passages"
                  data={filtered.filter((p) => retrieved.includes(p.id))}
                  fill="#fff"
                  stroke="#112c42"
                  strokeWidth={2}
                  onClick={pick}
                />
                <Scatter
                  name="Selected passage"
                  data={filtered.filter((p) => p.id === selected)}
                  fill="#e5a330"
                  stroke="#112c42"
                  strokeWidth={2}
                  onClick={pick}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <p className="lit-map-note">
            One point = one text fragment. PC2 explains{" "}
            {pct(projection.explained_variance[1])} of embedding variance.
            Proximity describes language, not biological causation or evidence
            quality.{" "}
            {selected &&
              !shown.has(selected) &&
              "The selected passage is outside the active filters."}
          </p>
        </>
      ) : (
        <div className="lit-unavailable">
          <Layers3 />
          <p>
            The semantic map is unavailable until the local embedding index has
            been built. Original sources and lexical retrieval remain
            accessible.
          </p>
        </div>
      )}
      <div className="lit-map-legend" aria-label="Filter map by source">
        <button
          aria-pressed={filter === "all"}
          onClick={() => setFilter("all")}
        >
          All sources
        </button>
        {summary.articles.map((a, i) => (
          <button
            key={a.pmcid}
            aria-pressed={filter === a.pmcid}
            onClick={() => setFilter(filter === a.pmcid ? "all" : a.pmcid)}
          >
            <i style={{ background: colors[i % colors.length] }} />
            {a.domain}
            <small>{a.chunks}</small>
          </button>
        ))}
      </div>
      {filtered.length > 0 && (
        <label className="lit-select lit-accessible-map">
          Inspect a map point (keyboard accessible)
          <select
            value={selected && shown.has(selected) ? selected : ""}
            onChange={(e) => e.target.value && onSelect(e.target.value)}
          >
            <option value="">Select source fragment</option>
            {filtered.map((p) => (
              <option key={p.id} value={p.id}>
                {p.id} · {p.section}
              </option>
            ))}
          </select>
        </label>
      )}
    </section>
  );
}

function BenchmarkPanel({
  report,
  onQuery,
}: {
  report: Benchmark | null;
  onQuery: (query: string) => void;
}) {
  const [metric, setMetric] = useState("ndcg_at_10");
  const [language, setLanguage] = useState("all");
  if (!report || report.status !== "available")
    return (
      <section className="lit-benchmark">
        <FlaskConical />
        <h2>Benchmark not yet run</h2>
        <p>
          No performance values are shown until the frozen query set has been
          evaluated against the local index.
        </p>
      </section>
    );
  const rows = report.case_results.filter(
    (row) => language === "all" || row.language === language,
  );
  return (
    <section className="lit-benchmark">
      <div className="lit-panel-heading">
        <div>
          <span className="lit-eyebrow">METHODS, NOT MARKETING</span>
          <h2>Retrieval comparison</h2>
        </div>
        <a
          className="lit-export"
          href="/api/literature/benchmark"
          target="_blank"
          rel="noreferrer"
        >
          <Download size={16} /> Full result JSON
        </a>
      </div>
      <p>
        {report.query_count} bilingual queries · {report.intent_clusters} paired
        intents · 5,000 intent-cluster bootstrap draws. Labels were authored
        from this corpus; this is a pilot, not an external benchmark.
      </p>
      <label className="lit-select">
        Ranking metric
        <select value={metric} onChange={(e) => setMetric(e.target.value)}>
          <option value="ndcg_at_10">nDCG@10 — ranked relevance</option>
          <option value="mrr_at_10">MRR@10 — first judged match</option>
          <option value="recall_at_5">
            Recall@5 — judged passage coverage
          </option>
        </select>
      </label>
      <div className="lit-score-grid">
        {Object.entries(report.aggregate[metric]).map(([method, value]) => (
          <article key={method}>
            <span>{methodNames[method]}</span>
            <strong>{value.estimate.toFixed(3)}</strong>
            <div
              className="lit-interval"
              role="img"
              aria-label={`95% bootstrap interval ${value.low.toFixed(3)} to ${value.high.toFixed(3)}`}
            >
              <i
                style={{
                  left: `${value.low * 100}%`,
                  width: `${(value.high - value.low) * 100}%`,
                }}
              />
              <b style={{ left: `${value.estimate * 100}%` }} />
            </div>
            <small>
              95% interval {value.low.toFixed(3)}–{value.high.toFixed(3)}
            </small>
            {method !== "bm25" && (
              <p>
                Δ vs BM25 {value.difference_vs_bm25 >= 0 ? "+" : ""}
                {value.difference_vs_bm25.toFixed(3)}
                <br />
                <small>
                  Paired interval {value.difference_low.toFixed(3)} to{" "}
                  {value.difference_high.toFixed(3)}
                </small>
              </p>
            )}
          </article>
        ))}
      </div>
      <div className="lit-language-grid">
        {["en", "zh"].map((lang) => (
          <div key={lang}>
            <h3>{lang === "en" ? "English queries" : "中文问题"}</h3>
            {Object.keys(methodNames).map((method) => (
              <div className="lit-language-row" key={method}>
                <span>{methodNames[method]}</span>
                <div>
                  <i
                    style={{
                      width: `${report.by_language[lang][method][metric] * 100}%`,
                    }}
                  />
                </div>
                <b>{report.by_language[lang][method][metric].toFixed(3)}</b>
              </div>
            ))}
          </div>
        ))}
      </div>
      <details className="lit-case-detail">
        <summary>Inspect every query, including failures</summary>
        <label className="lit-select">
          Language
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
          >
            <option value="all">English + Chinese</option>
            <option value="en">English</option>
            <option value="zh">中文</option>
          </select>
        </label>
        <div className="lit-table-scroll">
          <table>
            <caption>
              {metric}: unlisted passages are unjudged, not proven irrelevant
            </caption>
            <thead>
              <tr>
                <th>Query / rerun</th>
                {Object.values(methodNames).map((n) => (
                  <th key={n}>{n}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.intent}-${row.language}`}>
                  <td>
                    <button onClick={() => onQuery(row.query)}>
                      {row.query}
                      <ArrowUpRight size={14} />
                    </button>
                  </td>
                  {Object.keys(methodNames).map((method) => (
                    <td
                      key={method}
                      style={{
                        background: `rgba(18,107,119,${row.metrics[method][metric] * 0.18})`,
                      }}
                    >
                      {row.metrics[method][metric].toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
      <p className="lit-map-note">
        Neural retrieval is the interface default because it led this pilot,
        not because it was validated on an independent test set. No model
        fine-tuning was performed. Pretraining exposure to these articles is
        unknown. Bootstrap intervals describe this authored query set, not
        clinical safety or user benefit.
      </p>
    </section>
  );
}

export function LiteratureWorkbench() {
  const [summary, setSummary] = useState<Summary | null>(null);
  const [report, setReport] = useState<Benchmark | null>(null);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [query, setQuery] = useState(examples[0]);
  const [method, setMethod] = useState("dense");
  const [result, setResult] = useState<SearchResult | null>(null);
  const [passage, setPassage] = useState<Passage | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"explore" | "compare" | "sources">("explore");
  const requestId = useRef(0);
  const readerId = useRef(0);
  const readerElement = useRef<HTMLDivElement>(null);
  const [readerBusy, setReaderBusy] = useState(false);
  const [readerError, setReaderError] = useState("");
  async function load() {
    setLoading(true);
    setError("");
    try {
      const data = await researchApi<Summary>("/api/literature");
      setSummary(data);
      if (!data.semantic_ready) setMethod("bm25");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to load corpus");
    } finally {
      setLoading(false);
    }
  }
  async function loadBenchmark() {
    setBenchmarkError("");
    try { setReport(await researchApi<Benchmark>("/api/literature/benchmark")); }
    catch (e) { setBenchmarkError(e instanceof Error ? e.message : "Unable to load benchmark"); }
  }
  useEffect(() => { void load(); void loadBenchmark(); }, []);
  function revealReader() {
    requestAnimationFrame(() => readerElement.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start",
    }));
  }
  function showPassage(value: Passage) {
    setPassage(value);
    if (window.matchMedia("(max-width: 1250px)").matches) revealReader();
  }
  const retrieved = useMemo(
    () => result?.results.map((p) => p.id) ?? [],
    [result],
  );
  async function selectPassage(id: string) {
    const ticket = ++readerId.current;
    setReaderBusy(true);
    setReaderError("");
    try {
      const p = await researchApi<Passage>(
        `/api/literature/passage/${encodeURIComponent(id)}`,
      );
      if (ticket === readerId.current) showPassage(p);
    } catch (e) {
      if (ticket === readerId.current)
        setReaderError(
          e instanceof Error ? e.message : "Unable to inspect passage",
        );
    } finally {
      if (ticket === readerId.current) setReaderBusy(false);
    }
  }
  async function search(value = query) {
    if (!value.trim() || busy) return;
    const ticket = ++requestId.current;
    setBusy(true);
    setError("");
    setTab("explore");
    try {
      const data = await researchApi<SearchResult>(
        "/api/literature/search",
        "POST",
        { query: value, method, top_k: 6 },
      );
      if (ticket === requestId.current) {
        setResult(data);
        ++readerId.current;
        setReaderBusy(false);
        setReaderError("");
        setPassage(data.results[0] ?? null);
      }
    } catch (e) {
      if (ticket === requestId.current)
        setError(e instanceof Error ? e.message : "Retrieval failed");
    } finally {
      if (ticket === requestId.current) setBusy(false);
    }
  }
  function rerun(value: string) {
    setQuery(value);
    void search(value);
  }
  if (loading && !summary)
    return (
      <section className="lit-loading" role="status">
        Loading the public literature corpus…
      </section>
    );
  if (!summary)
    return (
      <section className="lit-loading">
        <p role="alert">{error}</p>
        <button onClick={() => void load()}>Retry corpus load</button>
      </section>
    );
  return (
    <div className="literature-workbench">
      <header className="lit-heading">
        <div>
          <span className="lit-eyebrow">01 / EVIDENCE INTELLIGENCE</span>
          <h1>From question to source.</h1>
          <p>
            Explore original passages. Compare retrieval methods. Inspect what
            the evidence can—and cannot—support.
          </p>
        </div>
        <span className="lit-local">
          <span /> LOCAL RESEARCH ONLY
        </span>
      </header>
      <section className="lit-search-panel">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void search();
          }}
        >
          <label htmlFor="literature-question">
            Research question · English or 中文
          </label>
          <div className="lit-search-row">
            <div>
              <Search size={20} />
              <input
                id="literature-question"
                value={query}
                maxLength={600}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask about a study, method or limitation"
              />
            </div>
            <label className="lit-method">
              <span className="lit-sr-only">Retrieval method</span>
              <select
                value={method}
                disabled={busy}
                onChange={(e) => setMethod(e.target.value)}
              >
                {Object.entries(methodNames).map(([key, name]) => (
                  <option
                    key={key}
                    value={key}
                    disabled={key !== "bm25" && !summary.semantic_ready}
                  >
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="lit-run"
              disabled={busy || query.trim().length < 2}
            >
              {busy ? "Retrieving…" : "Retrieve passages"}
              <ArrowUpRight size={17} />
            </button>
          </div>
        </form>
        <div className="lit-examples">
          {examples.map((value, i) => (
            <button key={value} disabled={busy} onClick={() => rerun(value)}>
              {
                [
                  "Progression / 进展",
                  "Donor-level evidence",
                  "Meaningful outcomes",
                ][i]
              }
            </button>
          ))}
        </div>
        <p className="lit-privacy">
          Questions stay on this local server and are not saved by the search
          endpoint. Do not enter patient-identifying information. No medical
          answer is generated.
        </p>
      </section>
      <div className="lit-metrics">
        <div>
          <strong>{summary.articles.length.toString().padStart(2, "0")}</strong>
          <span>PUBLIC ARTICLES</span>
        </div>
        <div>
          <strong>{summary.chunk_count}</strong>
          <span>EXACT-SOURCE FRAGMENTS</span>
        </div>
        <div>
          <strong>
            {summary.semantic_ready ? summary.model.dimensions : "—"}
          </strong>
          <span>NEURAL DIMENSIONS</span>
        </div>
        <div>
          <strong>
            {report?.status === "available" ? report.query_count : "—"}
          </strong>
          <span>PAIRED-LANGUAGE QUERIES</span>
        </div>
      </div>
      <nav className="lit-tabs" aria-label="Evidence workbench views">
        {(["explore", "compare", "sources"] as const).map((t) => (
          <button
            key={t}
            aria-current={tab === t ? "page" : undefined}
            onClick={() => setTab(t)}
          >
            {t === "explore"
              ? "Explore evidence"
              : t === "compare"
                ? "Methods & evaluation"
                : "Sources & provenance"}
          </button>
        ))}
      </nav>
      {error && (
        <p className="lit-error" role="alert">
          {error} {result && "Previous results remain visible."}
        </p>
      )}
      {tab === "explore" && (
        <>
          {result && (
            <div className="lit-result-caption" aria-live="polite">
              <strong>
                {result.action === "safety_abstain"
                  ? "Personal or causal medical claim: search withheld"
                  : `${result.results.length} passages · ${methodNames[result.method]}`}
              </strong>
              <span>For: {result.query}</span>
              {passage && <button className="lit-reader-jump" onClick={revealReader}>Read selected source ↓</button>}
              {(query !== result.query || method !== result.method) && (
                <b>Controls changed — run again to update results.</b>
              )}
            </div>
          )}
          <div className="lit-explore-grid">
            <div className="lit-main-column">
              {result && result.results.length > 0 && (
                <section
                  className="lit-results"
                  aria-label="Retrieved passages"
                >
                  {result.results.map((p, i) => (
                    <button
                      className={`lit-result ${passage?.id === p.id ? "is-selected" : ""}`}
                      key={p.id}
                      onClick={() => {
                        ++readerId.current;
                        setReaderBusy(false);
                        setReaderError("");
                        showPassage(p);
                      }}
                    >
                      <span className="lit-rank">
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <div>
                        <div className="lit-result-meta">
                          <span>{p.article.domain}</span>
                          <span>
                            {p.article.year} / {p.section}
                          </span>
                        </div>
                        <h3>{p.heading || p.article.title}</h3>
                        <p>
                          {p.text.slice(0, 210)}
                          {p.text.length > 210 ? "…" : ""}
                        </p>
                        <div className="lit-rank-strip">
                          {p.ranks &&
                            Object.entries(p.ranks).map(([name, rank]) => (
                              <span key={name}>
                                {methodNames[name]}{" "}
                                <b>{rank ? `#${rank}` : "—"}</b>
                              </span>
                            ))}
                        </div>
                      </div>
                      <ArrowUpRight size={17} />
                    </button>
                  ))}
                </section>
              )}
              {result?.action === "safety_abstain" && (
                <div className="lit-limit">
                  <p>
                    This research tool cannot diagnose an individual, recommend
                    medication or prove treatment benefit. Please discuss
                    personal questions with a qualified clinician or genetic
                    counsellor.
                  </p>
                </div>
              )}
              {result?.action === "no_lexical_match" && (
                <p className="lit-limit">
                  No lexical match. Try an English scientific term or a semantic
                  method; absence of a match is not absence of evidence.
                </p>
              )}
              <EvidenceMap
                summary={summary}
                selected={passage?.id ?? null}
                retrieved={retrieved}
                onSelect={(id) => void selectPassage(id)}
              />
            </div>
            <div className="lit-reader-column" ref={readerElement}>
              {readerBusy && <p role="status">Loading original passage…</p>}
              {readerError && (
                <p role="alert" className="lit-error">
                  {readerError}
                </p>
              )}
              <PassageReader passage={passage} />
            </div>
          </div>
        </>
      )}
      {tab === "compare" && (benchmarkError ? <div className="lit-error" role="alert"><p>{benchmarkError}</p><button onClick={() => void loadBenchmark()}>Retry benchmark load</button></div> : <BenchmarkPanel report={report} onQuery={rerun} />)}
      {tab === "sources" && (
        <section className="lit-sources">
          <div className="lit-panel-heading">
            <div>
              <span className="lit-eyebrow">AUDITABLE INPUTS</span>
              <h2>Source register</h2>
            </div>
            <a href="/api/literature" target="_blank" rel="noreferrer">
              Inspect manifest <ArrowUpRight size={16} />
            </a>
          </div>
          <p>{summary.selection}</p>
          <div className="lit-source-grid">
            {summary.articles.map((a, i) => (
              <article
                key={a.pmcid}
                style={{ borderTopColor: colors[i % colors.length] }}
              >
                <span className="lit-eyebrow">
                  {a.domain} / {a.year}
                </span>
                <h3>
                  <a href={a.source_url} target="_blank" rel="noreferrer">
                    {a.title}
                    <ArrowUpRight size={16} />
                  </a>
                </h3>
                <span className="lit-source-design">
                  {a.design} · {a.chunks} fragments
                </span>
                <p>{a.boundary}</p>
                <details>
                  <summary>Provenance & license</summary>
                  <p>{a.license}</p>
                  <code>{a.snapshot.sha256}</code>
                  <a href={a.snapshot.url} target="_blank" rel="noreferrer">
                    NCBI BioC source
                  </a>
                </details>
              </article>
            ))}
          </div>
          <div className="lit-limit">
            <h3>Versions kept separate</h3>
            {summary.not_indexed.map((s) => (
              <p key={s.pmcid}>
                <strong>{s.pmcid}</strong> — {s.reason}
              </p>
            ))}
          </div>
        </section>
      )}
      <footer className="lit-footer">
        <span>
          {summary.verification.exact_spans_checked} spans checked against
          source snapshots · relevance is not certified
        </span>
        <code>Corpus {summary.sha256.slice(0, 16)}</code>
      </footer>
    </div>
  );
}
