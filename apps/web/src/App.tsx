import { useEffect, useState } from "react";
import {
  Activity,
  ArrowUpRight,
  BarChart3,
  BookOpenCheck,
  Check,
  CheckCircle2,
  CircleHelp,
  Database,
  Dna,
  ExternalLink,
  FlaskConical,
  Gauge,
  Layers3,
  Menu,
  Network,
  Play,
  RefreshCw,
  ShieldCheck,
  TerminalSquare,
  TestTube2,
  X,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { loadDashboard, queryEvidence, runSimulation } from "./api";
import { DesignWorkbench } from "./DesignWorkbench";
import { AssessmentStudio } from "./AssessmentStudio";
import { FhirWorkbench, type ResearchBundle } from "./FhirWorkbench";
import { LiteratureWorkbench } from "./LiteratureWorkbench";
import { MolecularWorkbench } from "./MolecularWorkbench";
import { EvidencePortal } from "./EvidencePortal";
import type {
  BayesianMetaAnalysis,
  EvidenceClaim,
  GeneExpressionAnalysis,
  LearningModule,
  MetaAnalysis,
  Overview,
  SimulationResult,
  TranscriptomicsAnalysis,
  Trial,
  TrialSummary,
} from "./types";

type View =
  | "overview"
  | "evidence"
  | "analytics"
  | "trials"
  | "genomics"
  | "molecular"
  | "fhir"
  | "learning"
  | "governance";

type DashboardData = {
  overview: Overview;
  trialSummary: TrialSummary;
  trials: Trial[];
  meta: MetaAnalysis;
  bayesianMeta: BayesianMetaAnalysis;
  evidence: EvidenceClaim[];
  learning: { modules: LearningModule[]; learning_theories: string[] };
  genomics: Array<Record<string, string | number>>;
  transcriptomics: TranscriptomicsAnalysis;
  geneExpression: GeneExpressionAnalysis;
  governance: Record<string, unknown>;
  fhir: ResearchBundle;
};

const navItems: Array<{
  id: View;
  label: string;
  eyebrow: string;
  icon: typeof Activity;
}> = [
  {
    id: "overview",
    label: "Command center",
    eyebrow: "00 / Overview",
    icon: Gauge,
  },
  {
    id: "evidence",
    label: "Evidence atlas",
    eyebrow: "01 / Knowledge",
    icon: Network,
  },
  {
    id: "analytics",
    label: "Progression lab",
    eyebrow: "02 / Inference",
    icon: BarChart3,
  },
  {
    id: "trials",
    label: "Trial simulator",
    eyebrow: "03 / Design",
    icon: FlaskConical,
  },
  {
    id: "genomics",
    label: "Molecular analysis",
    eyebrow: "04 / Omics",
    icon: Dna,
  },
  {
    id: "molecular",
    label: "Molecular methods",
    eyebrow: "04B / Reliability",
    icon: Network,
  },
  {
    id: "fhir",
    label: "Research exchange",
    eyebrow: "05 / FHIR",
    icon: Layers3,
  },
  {
    id: "learning",
    label: "Learning studio",
    eyebrow: "06 / LDT",
    icon: BookOpenCheck,
  },
  {
    id: "governance",
    label: "Trust center",
    eyebrow: "07 / Governance",
    icon: ShieldCheck,
  },
];

const chartColors = [
  "#176b82",
  "#7b5d8c",
  "#a86b3e",
  "#41755b",
  "#8d4c60",
  "#476c9b",
];

function formatNumber(value: number | undefined, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: digits,
  }).format(value);
}

function statusLabel(value: string) {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function Badge({
  children,
  tone = "neutral",
}: {
  children: string;
  tone?: "neutral" | "blue" | "green" | "amber" | "red" | "purple";
}) {
  return (
    <span className={`badge badge-${tone}`}>
      <i />
      {children}
    </span>
  );
}

function LoadingScreen() {
  return (
    <div className="loading-screen">
      <div className="loading-mark">
        <Dna size={27} />
      </div>
      <span className="mono-label">SCA3 / COMPASS / BOOT</span>
      <h1>Loading the research workspace</h1>
      <div className="loading-line">
        <span />
      </div>
    </div>
  );
}

function Sidebar({
  active,
  onChange,
  open,
  onClose,
}: {
  active: View;
  onChange: (view: View) => void;
  open: boolean;
  onClose: () => void;
}) {
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : ""}`}>
      <button
        className="mobile-close"
        onClick={onClose}
        aria-label="Close navigation"
      >
        <X size={18} />
      </button>
      <div className="brand">
        <div className="brand-symbol">
          <Dna size={19} />
        </div>
        <div>
          <strong>SCA3 / Compass</strong>
          <span>Research workbench</span>
        </div>
      </div>
      <div className="rail-status">
        <span className="status-pulse" />
        <div>
          <strong>PUBLIC DATA MODE</strong>
          <small>Clinical claims locked</small>
        </div>
      </div>
      <nav className="rail-nav" aria-label="Research modules">
        {navItems.map((item) => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={active === item.id ? "active" : ""}
              onClick={() => {
                onChange(item.id);
                onClose();
              }}
            >
              <Icon size={16} strokeWidth={1.7} />
              <span>
                <small>{item.eyebrow}</small>
                <b>{item.label}</b>
              </span>
              {active === item.id && <i className="nav-marker" />}
            </button>
          );
        })}
      </nav>
      <div className="rail-footer">
        <ShieldCheck size={16} />
        <div>
          <strong>Research use only</strong>
          <span>No diagnosis or treatment</span>
        </div>
      </div>
    </aside>
  );
}

function Topbar({
  current,
  onMenu,
  overview,
}: {
  current: View;
  onMenu: () => void;
  overview: Overview;
}) {
  const item = navItems.find((entry) => entry.id === current)!;
  return (
    <header className="topbar">
      <button
        className="menu-button"
        onClick={onMenu}
        aria-label="Open navigation"
      >
        <Menu size={19} />
      </button>
      <div className="crumb">
        <span>{item.eyebrow}</span>
        <strong>{item.label}</strong>
      </div>
      <div className="topbar-meta">
        <span>
          <i className="status-pulse" />
          API ONLINE
        </span>
        <span className="hide-mobile">
          {overview.verified_public_sources}/
          {overview.registered_public_sources} SOURCES VERIFIED
        </span>
        <a href="/api/docs" target="_blank" rel="noreferrer">
          API DOCS <ArrowUpRight size={13} />
        </a>
      </div>
    </header>
  );
}

function SectionHeading({
  eyebrow,
  title,
  note,
  action,
}: {
  eyebrow: string;
  title: string;
  note?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="section-heading">
      <div>
        <span className="mono-label">{eyebrow}</span>
        <h1>{title}</h1>
        {note && <p>{note}</p>}
      </div>
      {action}
    </div>
  );
}

function Panel({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <section className={`panel ${className}`}>{children}</section>;
}

function PanelHeader({
  eyebrow,
  title,
  meta,
}: {
  eyebrow: string;
  title: string;
  meta?: React.ReactNode;
}) {
  return (
    <div className="panel-header">
      <div>
        <span className="mono-label">{eyebrow}</span>
        <h2>{title}</h2>
      </div>
      {meta}
    </div>
  );
}

function Metric({
  value,
  label,
  note,
  tone = "blue",
}: {
  value: string | number;
  label: string;
  note: string;
  tone?: string;
}) {
  return (
    <div className={`metric metric-${tone}`}>
      <small>{label}</small>
      <strong>{value}</strong>
      <span>{note}</span>
    </div>
  );
}

function DataLine({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string | number;
  mono?: boolean;
}) {
  return (
    <div className="data-line">
      <span>{label}</span>
      <b className={mono ? "mono" : ""}>{value}</b>
    </div>
  );
}

function OverviewView({
  data,
  navigate,
}: {
  data: DashboardData;
  navigate: (view: View) => void;
}) {
  const statusData = Object.entries(data.trialSummary.statuses).map(
    ([name, value]) => ({ name: statusLabel(name), value }),
  );
  const geneMetrics = data.geneExpression.metrics;
  const methods = Object.entries(data.overview.research_methods ?? {});
  return (
    <div className="view-stack">
      <SectionHeading
        eyebrow="SCA3 / RESEARCH WORKBENCH / RELEASE 2.0"
        title="A governed workspace for scarce evidence."
        note="Clinical-trial registry signals, donor-aware molecular analysis, statistical inference, trial operating characteristics, interoperability and adaptive learning — connected by explicit provenance."
        action={<Badge tone="green">PUBLIC_REAL ONLY</Badge>}
      />
      <div className="command-strip">
        <div>
          <span className="mono-label">RUN STATUS</span>
          <strong>All core pipelines available</strong>
        </div>
        <div>
          <span className="mono-label">ANALYSIS SCOPE</span>
          <strong>Registry · natural history · post-mortem RNA-seq</strong>
        </div>
        <div>
          <span className="mono-label">HARD BOUNDARY</span>
          <strong>Not a diagnostic or treatment service</strong>
        </div>
        <button onClick={() => navigate("governance")}>
          Inspect lineage <ArrowUpRight size={14} />
        </button>
      </div>
      <div className="metric-grid metric-grid-five">
        <Metric
          value={data.overview.exact_sca3_matches}
          label="Exact SCA3 trial matches"
          note={`${data.overview.trial_candidates} registry candidates`}
        />
        <Metric
          value={`${data.overview.verified_public_sources}/${data.overview.registered_public_sources}`}
          label="Sources verified"
          note="Snapshot and endpoint audit"
          tone="green"
        />
        <Metric
          value={formatNumber(data.meta.pooled_effect)}
          label="Pooled SARA change"
          note="REML points / year"
          tone="amber"
        />
        <Metric
          value={geneMetrics.genes_after_filter ?? "—"}
          label="Genes in model"
          note={`${geneMetrics.donors ?? "—"} inferred donor groups; IDs unresolved`}
          tone="purple"
        />
        <Metric
          value={data.fhir.entry.filter(entry => entry.resource.resourceType === "ResearchStudy").length}
          label="FHIR studies"
          note="ResearchStudy export"
          tone="slate"
        />
      </div>
      <div className="grid-2-1">
        <Panel>
          <PanelHeader
            eyebrow="REGISTRY / LIVE SNAPSHOT"
            title="Study status landscape"
            meta={<Badge>PUBLIC_REAL</Badge>}
          />
          <div className="chart-frame chart-short">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={statusData}
                margin={{ left: -22, right: 8, top: 12, bottom: 18 }}
              >
                <CartesianGrid
                  stroke="#e4e9eb"
                  strokeDasharray="2 4"
                  vertical={false}
                />
                <XAxis
                  dataKey="name"
                  tick={{ fill: "#6d7d84", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  angle={-15}
                  textAnchor="end"
                  height={50}
                />
                <YAxis
                  allowDecimals={false}
                  tick={{ fill: "#6d7d84", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    border: "1px solid #d9e0e3",
                    borderRadius: 2,
                    boxShadow: "0 8px 24px rgba(27,48,57,.10)",
                  }}
                />
                <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                  {statusData.map((_, index) => (
                    <Cell
                      key={index}
                      fill={chartColors[index % chartColors.length]}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
        <Panel className="contract-panel">
          <PanelHeader
            eyebrow="SCIENTIFIC CONTRACT"
            title="The workspace makes four promises"
            meta={<ShieldCheck size={17} />}
          />
          <div className="contract-list">
            {Object.entries(data.overview.governance).map(
              ([key, value], index) => (
                <div className="contract-row" key={key}>
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <div>
                    <strong>{statusLabel(key)}</strong>
                    <p>{value}</p>
                  </div>
                  <Check size={14} />
                </div>
              ),
            )}
          </div>
        </Panel>
      </div>
      <div className="grid-2-1">
        <Panel>
          <PanelHeader
            eyebrow="RESEARCH METHODS / IMPLEMENTED"
            title="Model registry"
            meta={<Badge tone="blue">TRACEABLE</Badge>}
          />
          <div className="method-table">
            <div className="table-head">
              <span>Module</span>
              <span>Method</span>
              <span>Evidence tier</span>
            </div>
            {methods.map(([key, value]) => (
              <div className="table-row" key={key}>
                <strong>{statusLabel(key)}</strong>
                <span>{String(value)}</span>
                <Badge
                  tone={String(value).includes("simulation") ? "amber" : "blue"}
                >
                  {String(value).includes("simulation")
                    ? "SIMULATED"
                    : "DERIVED"}
                </Badge>
              </div>
            ))}
          </div>
        </Panel>
        <Panel className="run-panel">
          <PanelHeader
            eyebrow="LATEST DERIVED ARTIFACT"
            title="GSE309548 gene analysis"
            meta={<Dna size={17} />}
          />
          <DataLine
            label="Inferred donor groups"
            value={`${geneMetrics.sca3_donors ?? "—"} SCA3 / ${geneMetrics.control_donors ?? "—"} CTRL`}
          />
          <DataLine
            label="GENCODE mapping"
            value={`${formatNumber((geneMetrics.abundance_mapping_rate ?? 0) * 100, 1)}% abundance`}
          />
          <DataLine label="Primary model" value="SCA3 + sex / EB variance" />
          <DataLine
            label="Sensitivity"
            value={`${geneMetrics.age_adjusted_fdr_005 ?? "—"} age-adjusted FDR signals`}
          />
          <button className="text-button" onClick={() => navigate("genomics")}>
            Open molecular analysis <ArrowUpRight size={14} />
          </button>
        </Panel>
      </div>
      <Panel>
        <PanelHeader
          eyebrow="PIPELINE / CONTROLLED HANDOFF"
          title="Observation → inference → design → translation"
        />
        <div className="pipeline-track">
          {[
            ["01", "Ingest", "Versioned public snapshots"],
            ["02", "Infer", "Effect + uncertainty"],
            ["03", "Stress-test", "Operating characteristics"],
            ["04", "Exchange", "FHIR + provenance"],
            ["05", "Learn", "BKT + assessment"],
          ].map((step, index) => (
            <div key={step[0]}>
              <span>{step[0]}</span>
              <strong>{step[1]}</strong>
              <small>{step[2]}</small>
              {index < 4 && <i />}
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function EvidenceView({ claims }: { claims: EvidenceClaim[] }) {
  return <div className="view-stack"><LiteratureWorkbench /><details className="wb-legacy"><summary>Original curated claim cards (separate five-claim baseline)</summary><CuratedEvidenceView claims={claims} /></details></div>;
}

function CuratedEvidenceView({ claims }: { claims: EvidenceClaim[] }) {
  const [filter, setFilter] = useState("All");
  const [query, setQuery] = useState(
    "What is the annual SARA progression estimate?",
  );
  const [queryResult, setQueryResult] = useState<Record<string, any> | null>(
    null,
  );
  const [queryRunning, setQueryRunning] = useState(false);
  const [queryError, setQueryError] = useState("");
  const topics = ["All", ...new Set(claims.map((claim) => claim.topic))];
  const visible =
    filter === "All"
      ? claims
      : claims.filter((claim) => claim.topic === filter);
  async function runQuery() {
    setQueryRunning(true);
    setQueryError("");
    try {
      setQueryResult(await queryEvidence(query));
    } catch (reason) {
      setQueryError(reason instanceof Error ? reason.message : "Retrieval failed");
    } finally {
      setQueryRunning(false);
    }
  }
  return (
    <div className="view-stack">
      <SectionHeading
        eyebrow="01 / EVIDENCE ATLAS"
        title="Claims are useful only when their boundary is visible."
        note="The atlas is citation-bound: each statement carries a source, evidence type, provenance tier and an explicit interpretation limit."
        action={<Badge>CLAIM GATE ACTIVE</Badge>}
      />
      <Panel className="query-panel">
        <PanelHeader
          eyebrow="AUDITABLE EVIDENCE INTELLIGENCE"
          title={`Search ${claims.length} curated evidence claims`}
          meta={<Badge tone="blue">BM25 / CITATION-BOUND</Badge>}
        />
        <div className="query-row">
          <input
            aria-label="Evidence query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button
            className="primary-button"
            onClick={runQuery}
            disabled={queryRunning}
          >
            <Network size={14} />
            {queryRunning ? "Retrieving" : "Run retrieval"}
          </button>
        </div>
        {queryError && <p className="wb-error" role="alert">{queryError}</p>}
        {queryResult && (
          <div className="query-result">
            <div>
              <span className="mono-label">ACTION</span>
              <strong>{String(queryResult.action ?? "answer")}</strong>
            </div>
            <div>
              <span className="mono-label">TRACE</span>
              <span>
                {Array.isArray(queryResult.trace)
                  ? queryResult.trace
                      .map((item: any) =>
                        `${item.stage}: ${item.status}`,
                      )
                      .join(" → ")
                  : "policy → retrieval → citation verifier"}
              </span>
            </div>
            <div>
              <span className="mono-label">CITATIONS</span>
              <strong>
                {Array.isArray(queryResult.citations)
                  ? queryResult.citations.length
                  : 0}
              </strong>
            </div>
            <p className="query-answer">
              {String(queryResult.answer ?? "No extractive answer returned.")}
            </p>
            {queryResult.boundary && <p className="query-answer wb-boundary">{String(queryResult.boundary)}</p>}
            <div className="query-citation-list">{Array.isArray(queryResult.citations) && queryResult.citations.map((citation: { claim_id: string; source_url: string; source_title: string; boundary: string }) => <article key={citation.claim_id}><a href={citation.source_url} target="_blank" rel="noreferrer">{citation.source_title} <ExternalLink size={12} /></a><p>{citation.boundary}</p></article>)}<p>Claim-to-source linkage is checked; automated semantic entailment and clinical validity are not established.</p></div>
          </div>
        )}
      </Panel>
      <div className="filter-row">
        {topics.map((topic) => (
          <button
            key={topic}
            className={filter === topic ? "selected" : ""}
            onClick={() => setFilter(topic)}
          >
            {topic}
          </button>
        ))}
      </div>
      <Panel>
        <div className="evidence-table">
          <div className="table-head">
            <span>Claim</span>
            <span>Evidence type</span>
            <span>Source</span>
            <span>Boundary</span>
          </div>
          {visible.map((claim) => (
            <article className="evidence-row" key={claim.id}>
              <div>
                <span className="mono-label">
                  {claim.topic} / {claim.id}
                </span>
                <strong>{claim.title}</strong>
                <p>{claim.plain_language}</p>
              </div>
              <Badge tone="blue">{statusLabel(claim.evidence_type)}</Badge>
              <a href={claim.source_url} target="_blank" rel="noreferrer">
                {claim.source_title}
                <ExternalLink size={13} />
              </a>
              <div className="boundary-inline">
                <CircleHelp size={14} />
                <span>{claim.boundary}</span>
              </div>
            </article>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function ForestPlot({ meta }: { meta: MetaAnalysis }) {
  const min = Math.min(
    0,
    ...meta.forest.map((item) => item.ci_low),
    meta.prediction_low,
  );
  const max = Math.max(
    ...meta.forest.map((item) => item.ci_high),
    meta.prediction_high,
  );
  const position = (value: number) => `${((value - min) / (max - min)) * 100}%`;
  return (
    <div className="forest">
      <div className="forest-grid">
        <span>COHORT / WEIGHT</span>
        <span>ESTIMATE / 95% CI</span>
        <span>VALUE</span>
      </div>
      {meta.forest.map((item) => (
        <a
          className="forest-row"
          href={item.source_url}
          target="_blank"
          rel="noreferrer"
          key={item.study_id}
        >
          <div className="forest-label">
            <strong>{item.cohort}</strong>
            <span>
              {item.sample_size ? `n=${item.sample_size}` : "n not extracted"} ·{" "}
              {item.weight_percent.toFixed(1)}%
            </span>
          </div>
          <div className="forest-scale">
            <i className="forest-zero" style={{ left: position(0) }} />
            <span
              className="forest-ci"
              style={{
                left: position(item.ci_low),
                width: `calc(${position(item.ci_high)} - ${position(item.ci_low)})`,
              }}
            />
            <b
              className="forest-point"
              style={{ left: position(item.effect) }}
            />
          </div>
          <div className="forest-value">
            {item.effect.toFixed(2)}{" "}
            <span>
              [{item.ci_low.toFixed(2)}, {item.ci_high.toFixed(2)}]
            </span>
          </div>
        </a>
      ))}
      <div className="forest-row pooled-row">
        <div className="forest-label">
          <strong>REML pooled</strong>
          <span>
            Prediction interval: {meta.prediction_low.toFixed(2)} to{" "}
            {meta.prediction_high.toFixed(2)}
          </span>
        </div>
        <div className="forest-scale">
          <i className="forest-zero" style={{ left: position(0) }} />
          <span
            className="forest-ci pooled"
            style={{
              left: position(meta.ci_low),
              width: `calc(${position(meta.ci_high)} - ${position(meta.ci_low)})`,
            }}
          />
          <b
            className="forest-diamond"
            style={{ left: position(meta.pooled_effect) }}
          />
        </div>
        <div className="forest-value">
          {meta.pooled_effect.toFixed(2)}{" "}
          <span>
            [{meta.ci_low.toFixed(2)}, {meta.ci_high.toFixed(2)}]
          </span>
        </div>
      </div>
      <div className="forest-axis">
        <span>{min.toFixed(1)}</span>
        <span>Annual SARA change / points per year</span>
        <span>{max.toFixed(1)}</span>
      </div>
    </div>
  );
}

function AnalyticsView({
  meta,
  bayesian,
}: {
  meta: MetaAnalysis;
  bayesian: BayesianMetaAnalysis;
}) {
  const influenceData = meta.leave_one_out.map((row) => ({
    name: row.omitted_study_id.replace("_", " "),
    estimate: Number(row.pooled_effect.toFixed(3)),
  }));
  return (
    <div className="view-stack">
      <SectionHeading
        eyebrow="02 / PROGRESSION LAB"
        title="Estimate the signal. Keep the uncertainty."
        note="Publication-level natural-history estimates are synthesized with frequentist REML and an independently implemented normal–normal hierarchical posterior. Neither is an individual forecast."
        action={<Badge tone="blue">PUBLIC_REAL_DERIVED</Badge>}
      />
      <div className="metric-grid metric-grid-four dark-metrics">
        <Metric
          value={meta.pooled_effect.toFixed(2)}
          label="REML pooled effect"
          note="SARA points / year"
        />
        <Metric
          value={bayesian.pooled_effect.median.toFixed(2)}
          label="Posterior median"
          note={`95% CrI ${bayesian.pooled_effect.low.toFixed(2)} to ${bayesian.pooled_effect.high.toFixed(2)}`}
          tone="slate"
        />
        <Metric
          value={bayesian.heterogeneity_tau.median.toFixed(2)}
          label="Posterior tau"
          note={`95% CrI ${bayesian.heterogeneity_tau.low.toFixed(2)} to ${bayesian.heterogeneity_tau.high.toFixed(2)}`}
          tone="purple"
        />
        <Metric
          value={`${(bayesian.probability_progression_above_one * 100).toFixed(1)}%`}
          label="P(progress > 1)"
          note="Prior-sensitive posterior"
          tone="amber"
        />
      </div>
      <Panel>
        <PanelHeader
          eyebrow="PRIMARY INDEPENDENT COHORT ESTIMATES"
          title="Forest plot"
          meta={<span className="method-chip">{meta.method}</span>}
        />
        <ForestPlot meta={meta} />
        <div className="boundary-note">
          <CircleHelp size={14} />
          <span>{meta.interpretation_boundary}</span>
        </div>
      </Panel>
      <div className="grid-1-1">
        <Panel>
          <PanelHeader
            eyebrow="BAYESIAN HIERARCHICAL MODEL"
            title="Posterior and new-setting prediction"
            meta={<Badge tone="purple">30,000 DRAWS</Badge>}
          />
          <div className="posterior-grid">
            <DataLine
              label="Posterior mean"
              value={bayesian.pooled_effect.mean.toFixed(3)}
            />
            <DataLine
              label="New-setting median"
              value={bayesian.new_setting_prediction.median.toFixed(3)}
            />
            <DataLine
              label="New-setting 95% interval"
              value={`${bayesian.new_setting_prediction.low.toFixed(2)} to ${bayesian.new_setting_prediction.high.toFixed(2)}`}
            />
            <DataLine
              label="Prior"
              value={bayesian.priors.tau ?? "HalfNormal"}
            />
          </div>
          <div className="sensitivity-table">
            <div>
              <span>tau prior</span>
              <span>posterior median</span>
              <span>95% interval</span>
            </div>
            {bayesian.prior_sensitivity.map((row) => (
              <div key={row.tau_prior_scale}>
                <strong>{row.tau_prior_scale.toFixed(2)}</strong>
                <span>{row.pooled_median.toFixed(2)}</span>
                <span>
                  {row.pooled_low.toFixed(2)} to {row.pooled_high.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <div className="boundary-note compact">
            <CircleHelp size={14} />
            <span>{bayesian.interpretation_boundary}</span>
          </div>
        </Panel>
        <Panel>
          <PanelHeader
            eyebrow="INFLUENCE DIAGNOSTIC"
            title="Leave one cohort out"
          />
          <div className="chart-frame chart-medium">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={influenceData}
                layout="vertical"
                margin={{ left: 26, right: 20 }}
              >
                <CartesianGrid
                  stroke="#e4e9eb"
                  strokeDasharray="2 4"
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  tick={{ fill: "#6d7d84", fontSize: 10 }}
                  axisLine={false}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={94}
                  tick={{ fill: "#6d7d84", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{
                    border: "1px solid #d9e0e3",
                    borderRadius: 2,
                  }}
                />
                <Bar dataKey="estimate" fill="#176b82" radius={[0, 2, 2, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>
      </div>
      <Panel>
        <PanelHeader
          eyebrow="ANALYSIS CONTRACT"
          title="Review before interpretation"
        />
        <div className="check-list">
          {[
            "Overlapping follow-up excluded from primary synthesis",
            "Frequentist and Bayesian estimates shown side by side",
            "Prior sensitivity and new-setting prediction retained",
            "Study influence reported, not hidden",
            "Cohort average never reframed as personal prognosis",
          ].map((text) => (
            <div key={text}>
              <CheckCircle2 size={15} />
              <span>{text}</span>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  suffix,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  suffix: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="slider-field">
      <span>
        <b>{label}</b>
        <output>
          {value}
          {suffix}
        </output>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function TrialLabView(props: { meta: MetaAnalysis; trials: Trial[] }) {
  return <div className="view-stack"><DesignWorkbench /><details className="wb-card wb-disclosure"><summary>Baseline comparison: original fixed-parameter parallel simulator and registry context</summary><SimpleTrialLabView {...props} /></details></div>;
}

function SimpleTrialLabView({ trials }: { meta: MetaAnalysis; trials: Trial[] }) {
  const [settings, setSettings] = useState({
    participants_per_arm: 60,
    followup_months: 24,
    treatment_reduction: 0.3,
    slope_sd: 1.5,
    measurement_sd: 0.7,
    attrition_rate: 0.1,
    simulations: 1000,
    seed: 202709,
  });
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [running, setRunning] = useState(false);
  const [simulationError, setSimulationError] = useState("");
  const update = (key: string, value: number) =>
    setSettings((current) => ({ ...current, [key]: value }));
  async function run() {
    setRunning(true);
    setSimulationError("");
    try {
      setResult(await runSimulation(settings));
    } catch (reason) {
      setSimulationError(reason instanceof Error ? reason.message : "Simulation failed");
    } finally {
      setRunning(false);
    }
  }
  return (
    <div className="view-stack">
      <SectionHeading
        eyebrow="03 / TRIAL DESIGN LABORATORY"
        title="Stress-test a design before a participant is enrolled."
        note="Operating characteristics are simulated from public natural-history estimates. A simulated output is never an observed patient result."
        action={<Badge tone="amber">PUBLIC INPUTS / SIMULATED OUTPUT</Badge>}
      />
      <div className="sim-layout">
        {simulationError && <p className="wb-error" role="alert">{simulationError}</p>}
        <Panel className="control-panel">
          <PanelHeader
            eyebrow="DESIGN CONTROLS"
            title="Parallel two-arm study"
            meta={<TestTube2 size={17} />}
          />
          <SliderField
            label="Participants per arm"
            value={settings.participants_per_arm}
            min={10}
            max={250}
            step={5}
            suffix=""
            onChange={(value) => update("participants_per_arm", value)}
          />
          <SliderField
            label="Follow-up"
            value={settings.followup_months}
            min={6}
            max={48}
            step={6}
            suffix=" mo"
            onChange={(value) => update("followup_months", value)}
          />
          <SliderField
            label="Progression reduction"
            value={Math.round(settings.treatment_reduction * 100)}
            min={5}
            max={60}
            step={5}
            suffix="%"
            onChange={(value) => update("treatment_reduction", value / 100)}
          />
          <SliderField
            label="Attrition"
            value={Math.round(settings.attrition_rate * 100)}
            min={0}
            max={35}
            step={5}
            suffix="%"
            onChange={(value) => update("attrition_rate", value / 100)}
          />
          <button
            className="primary-button run-button"
            onClick={run}
            disabled={running}
          >
            {running ? (
              <RefreshCw className="spin" size={15} />
            ) : (
              <Play size={15} fill="currentColor" />
            )}
            {running
              ? "Running 1,000 replicates"
              : "Run reproducible simulation"}
          </button>
          <span className="seed-label">
            <TerminalSquare size={13} /> deterministic seed / {settings.seed}
          </span>
        </Panel>
        <Panel className="result-panel">
          {!result ? (
            <div className="empty-state">
              <div className="empty-icon">
                <Activity size={22} />
              </div>
              <span className="mono-label">NO RUN IN THIS SESSION</span>
              <h2>Operating characteristics will appear here.</h2>
              <p>
                Change a design assumption, run the simulation, and inspect
                power, bias, uncertainty and analyzed sample size.
              </p>
            </div>
          ) : (
            <>
              <PanelHeader
                eyebrow="SIMULATION RESULT"
                title="Design operating characteristics"
                meta={<Badge tone="amber">SIMULATED</Badge>}
              />
              <div className="result-main">
                <div className="power-ring">
                  <strong>
                    {Math.round(result.power * 100)}
                    <small>%</small>
                  </strong>
                  <span>POWER</span>
                </div>
                <div className="result-copy">
                  <span className="mono-label">TARGET MEAN DIFFERENCE</span>
                  <strong>{result.target_mean_difference.toFixed(3)}</strong>
                  <p>{result.interpretation_boundary}</p>
                </div>
              </div>
              <div className="result-grid">
                <DataLine
                  label="Mean estimated difference"
                  value={result.mean_estimated_difference.toFixed(3)}
                />
                <DataLine label="Bias" value={result.bias.toFixed(4)} />
                <DataLine
                  label="Mean standard error"
                  value={result.mean_standard_error.toFixed(3)}
                />
                <DataLine
                  label="Analyzed participants"
                  value={formatNumber(result.mean_analyzed_participants, 1)}
                />
              </div>
            </>
          )}
        </Panel>
      </div>
      <Panel>
        <PanelHeader
          eyebrow="REGISTRY CONTEXT"
          title="Exact SCA3 trial records used for design context"
          meta={<Badge>PUBLIC_REAL</Badge>}
        />
        <div className="trial-table">
          <div className="table-head">
            <span>Identifier</span>
            <span>Study</span>
            <span>Status</span>
            <span>n</span>
          </div>
          {trials.slice(0, 6).map((trial) => (
            <a
              className="table-row"
              href={trial.record_url}
              target="_blank"
              rel="noreferrer"
              key={trial.nct_id}
            >
              <strong className="mono">{trial.nct_id}</strong>
              <span>{trial.brief_title}</span>
              <Badge
                tone={
                  trial.overall_status === "RECRUITING" ? "green" : "neutral"
                }
              >
                {statusLabel(trial.overall_status)}
              </Badge>
              <b>{trial.enrollment ?? "—"}</b>
            </a>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function PcaPlot({ analysis }: { analysis: GeneExpressionAnalysis }) {
  const sca3 = analysis.pca.filter((item) => item.genotype === "SCA3");
  const control = analysis.pca.filter((item) => item.genotype !== "SCA3");
  return (
    <div className="pca-wrap">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 12, right: 20, bottom: 24, left: 4 }}>
          <CartesianGrid stroke="#e4e9eb" strokeDasharray="2 4" />
          <XAxis
            type="number"
            dataKey="pc1"
            name="PC1"
            tick={{ fill: "#6d7d84", fontSize: 10 }}
            axisLine={{ stroke: "#bfcbd0" }}
          />
          <YAxis
            type="number"
            dataKey="pc2"
            name="PC2"
            tick={{ fill: "#6d7d84", fontSize: 10 }}
            axisLine={{ stroke: "#bfcbd0" }}
          />
          <Tooltip
            cursor={{ stroke: "#91a6ae", strokeDasharray: "3 3" }}
            contentStyle={{ border: "1px solid #d9e0e3", borderRadius: 2 }}
            formatter={(value) => Number(value).toFixed(2)}
          />
          <Scatter name="SCA3" data={sca3} fill="#176b82" />
          <Scatter name="Control" data={control} fill="#a86b3e" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

function GenomicsView({
  datasets,
  transcriptomics,
  analysis,
}: {
  datasets: Array<Record<string, string | number>>;
  transcriptomics: TranscriptomicsAnalysis;
  analysis: GeneExpressionAnalysis;
}) {
  const [mode, setMode] = useState<"gene" | "transcript">("gene");
  const metrics = analysis.metrics;
  const variance = metrics.pca_variance_percent ?? [];
  const rows =
    mode === "gene" ? analysis.top_signals : transcriptomics.top_signals;
  return (
    <div className="view-stack">
      <SectionHeading
        eyebrow="04 / MOLECULAR ANALYSIS"
        title="Donor-aware molecular evidence, with replication boundaries intact."
        note="Exploratory GENCODE-mapped aggregates, not a validated differential-expression pipeline. The eight groups were inferred from demographics, not certified source donor IDs. Transcript-level TPM is a separate exploratory view. No disease mechanism is established."
        action={<Badge tone="green">PUBLIC_REAL_DERIVED</Badge>}
      />
      <div className="metric-grid metric-grid-six">
        <Metric
          value={metrics.donors ?? "—"}
          label="Inferred donor groups"
          note={`${metrics.sca3_donors ?? "—"} SCA3 · ${metrics.control_donors ?? "—"} CTRL`}
        />
        <Metric
          value={formatNumber(metrics.genes_after_filter, 0)}
          label="Genes in model"
          note={`of ${formatNumber(metrics.genes_before_filter, 0)} mapped`}
          tone="purple"
        />
        <Metric
          value={`${formatNumber((metrics.abundance_mapping_rate ?? 0) * 100, 1)}%`}
          label="Abundance mapped"
          note="Transcript → gene"
          tone="green"
        />
        <Metric
          value={metrics.fdr_005 ?? "—"}
          label="Exploratory FDR < .05"
          note="SCA3 + sex"
          tone="amber"
        />
        <Metric
          value={metrics.age_adjusted_fdr_005 ?? "—"}
          label="Age-adjusted FDR"
          note="Sensitivity model"
          tone="slate"
        />
        <Metric
          value={`${variance[0] ?? "—"}% / ${variance[1] ?? "—"}%`}
          label="PCA variance"
          note="PC1 / PC2"
          tone="blue"
        />
      </div>
      <div className="grid-3-2">
        <Panel>
          <PanelHeader
            eyebrow="DONOR-LEVEL PCA / FILTERED GENES"
            title="Eight people, not fourteen independent files"
            meta={<Badge>DONOR UNIT</Badge>}
          />
          <div className="pca-chart">
            <PcaPlot analysis={analysis} />
          </div>
          <div className="legend">
            <span>
              <i className="dot dot-blue" />
              SCA3 donor
            </span>
            <span>
              <i className="dot dot-amber" />
              Control donor
            </span>
          </div>
        </Panel>
        <Panel>
          <PanelHeader
            eyebrow="MODEL DIAGNOSTICS"
            title="Inferential contract"
            meta={<ShieldCheck size={17} />}
          />
          <DataLine
            label="Primary design"
            value={metrics.primary_model?.design?.join(" + ") ?? "—"}
          />
          <DataLine
            label="Residual df"
            value={metrics.primary_model?.residual_df ?? "—"}
          />
          <DataLine
            label="EB prior df"
            value={formatNumber(
              metrics.primary_model?.empirical_bayes_prior_df,
              2,
            )}
          />
          <DataLine
            label="Condition number"
            value={formatNumber(metrics.primary_model?.condition_number, 2)}
          />
          <DataLine
            label="Label assignments"
            value={`${metrics.assignment_calibration?.enumerated_assignments ?? "—"} exact 4:4`}
          />
          <DataLine label="Assignment max-statistic tail" value={formatNumber(metrics.assignment_calibration?.max_statistic_tail_probability, 3)} />
          <div className="boundary-note compact">
            <CircleHelp size={14} />
            <span>
              Nonrandom observational labels are not exchangeable by design: enumeration is a sensitivity diagnostic, not a causal randomization test. Age and tissue-region imbalance remain.
            </span>
          </div>
        </Panel>
      </div>
      <Panel>
        <PanelHeader
          eyebrow="DIFFERENTIAL EXPRESSION"
          title="Top feature signals"
          meta={
            <div className="segmented">
              <button
                className={mode === "gene" ? "selected" : ""}
                onClick={() => setMode("gene")}
              >
                GENE MODEL
              </button>
              <button
                className={mode === "transcript" ? "selected" : ""}
                onClick={() => setMode("transcript")}
              >
                TRANSCRIPT EXPLORATORY
              </button>
            </div>
          }
        />
        <div className="signal-table">
          <div className="table-head">
            <span>Feature</span>
            <span>log2 effect</span>
            <span>FDR</span>
            <span>Age sensitivity</span>
          </div>
          {rows.slice(0, 14).map((signal) => {
            const item = signal as typeof signal & {
              gene_id?: string;
              gene_type?: string;
              age_adjusted_log2_fold_change?: number;
              age_adjusted_q_value?: number;
              direction_concordant?: boolean;
              transcript_id?: string;
            };
            const identifier = (mode === "gene" ? item.gene_id : item.transcript_id) ?? "feature";
            const effect = item.log2_fold_change;
            const q = item.q_value;
            return (
              <a
                className="signal-row"
                key={identifier}
                href={`https://www.ensembl.org/Homo_sapiens/Search/Results?q=${identifier}`}
                target="_blank"
                rel="noreferrer"
              >
                <div>
                  <strong>{item.gene_name ?? identifier}</strong>
                  <small>{item.gene_type ?? identifier}</small>
                </div>
                <b className={effect >= 0 ? "effect-up" : "effect-down"}>
                  {effect >= 0 ? "+" : ""}
                  {effect.toFixed(2)}
                </b>
                <span className={q <= 0.05 ? "significant" : ""}>
                  {q <= 0.001 ? "<.001" : q.toFixed(3)}
                </span>
                <span>
                  {mode === "gene"
                    ? item.direction_concordant
                      ? "concordant"
                      : "flip"
                    : "TPM only"}
                </span>
              </a>
            );
          })}
        </div>
      </Panel>
      <Panel>
        <PanelHeader
          eyebrow="REPRODUCIBLE DATA ASSETS"
          title="Public datasets and analysis boundary"
          meta={<Database size={17} />}
        />
        <div className="dataset-table">
          <div className="table-head">
            <span>Accession</span>
            <span>Model</span>
            <span>Files</span>
            <span>Status</span>
          </div>
          {datasets.map((item) => (
            <a
              className="table-row"
              href={String(item.url)}
              target="_blank"
              rel="noreferrer"
              key={String(item.accession)}
            >
              <strong className="mono">{String(item.accession)}</strong>
              <span>{String(item.model)}</span>
              <b>{String(item.files)}</b>
              <span>
                <Badge tone="green">{String(item.status)}</Badge>
              </span>
            </a>
          ))}
        </div>
        <div className="boundary-note">
          <CircleHelp size={14} />
          <span>
            {analysis.interpretation_boundary ??
              "Analysis boundary unavailable."}
          </span>
        </div>
        <div className="hash-row">
          <span>GEO SHA-256</span>
          <code>
            {analysis.sources?.geo_archive?.sha256?.slice(0, 20) ??
              transcriptomics.source?.sha256?.slice(0, 20) ??
              "pending"}
            …
          </code>
          <span>GENCODE v50 SHA-256</span>
          <code>
            {analysis.sources?.gencode_annotation?.sha256?.slice(0, 20) ??
              "pending"}
            …
          </code>
        </div>
      </Panel>
    </div>
  );
}

function LearningView({
  learning,
}: {
  learning: { modules: LearningModule[]; learning_theories: string[] };
}) {
  const [selected, setSelected] = useState(0);
  const module = learning.modules[selected];
  return (
    <div className="view-stack">
      <AssessmentStudio />
      <details className="wb-card wb-disclosure">
      <summary>Lesson design notes and learning objectives (not a completion record)</summary>
      <SectionHeading
        eyebrow="06 / LEARNING DESIGN STUDIO"
        title="Assessment is part of the evidence system."
        note="A prerequisite-constrained Bayesian Knowledge Tracing prototype turns evidence literacy into an inspectable learning sequence. Parameters remain design assumptions until learner data exists."
        action={<Badge tone="purple">LDT / BKT PROTOTYPE</Badge>}
      />
      <div className="learning-layout">
        <aside className="module-index">
          <span className="mono-label">
            MODULE INDEX / {learning.modules.length}
          </span>
          {learning.modules.map((item, index) => (
            <button
              key={item.id}
              className={selected === index ? "selected" : ""}
              onClick={() => setSelected(index)}
            >
              <span>{item.number}</span>
              <div>
                <strong>{item.title}</strong>
                <small>
                  {item.estimated_minutes} min · threshold{" "}
                  {Math.round(item.mastery_threshold * 100)}%
                </small>
              </div>
            </button>
          ))}
        </aside>
        <article className="lesson-canvas">
          <span className="mono-label">
            LESSON {module.number} / {module.id}
          </span>
          <h2>{module.title}</h2>
          <p className="lesson-description">{module.description}</p>
          <div className="objective-box">
            <span className="mono-label">LEARNING OBJECTIVES</span>
            {module.objectives.map((objective) => (
              <div key={objective}>
                <Check size={14} />
                {objective}
              </div>
            ))}
          </div>
          <div className="teachback">
            <BookOpenCheck size={18} />
            <div>
              <span className="mono-label">TEACH-BACK PROMPT</span>
              <h3>Explain the boundary before the result.</h3>
              <p>
                What evidence would change your confidence, and what would
                remain unknown?
              </p>
            </div>
          </div>
          <button
            className="primary-button"
            onClick={() =>
              setSelected((selected + 1) % learning.modules.length)
            }
          >
            Continue to next module <ArrowUpRight size={14} />
          </button>
          <div className="theory-row">
            <span className="mono-label">THEORY REGISTER</span>
            {learning.learning_theories.map((theory) => (
              <Badge key={theory} tone="purple">
                {theory}
              </Badge>
            ))}
          </div>
        </article>
      </div>
      </details>
    </div>
  );
}

function GovernanceView({
  governance,
}: {
  governance: Record<string, unknown>;
}) {
  const sources = Array.isArray(governance.sources)
    ? (governance.sources as Array<Record<string, unknown>>)
    : [];
  const audit = governance.audit as Record<string, unknown> | undefined;
  const claimGate = Array.isArray(governance.claim_gate)
    ? (governance.claim_gate as string[])
    : [];
  const prohibited = Array.isArray(governance.prohibited)
    ? (governance.prohibited as string[])
    : [];
  return (
    <div className="view-stack">
      <SectionHeading
        eyebrow="07 / TRUST CENTER"
        title="Trust is a property of the pipeline, not a decorative label."
        note="Every scientific output is expected to carry provenance, transformation logic, uncertainty and a visible boundary. Governance is inspectable at runtime."
        action={<Badge tone="green">AUDITABLE</Badge>}
      />
      <div className="trust-banner">
        <div className="trust-seal">
          <ShieldCheck size={25} />
        </div>
        <div>
          <span className="mono-label">
            POLICY VERSION / {String(governance.policy_version ?? "1.0")}
          </span>
          <h2>PUBLIC_REAL claim gate</h2>
          <p>
            Only registered sources, digested snapshots, deterministic
            transformations, tested models, uncertainty and interpretation
            boundaries can enter the research views.
          </p>
        </div>
        <div className="trust-score">
          <strong>{String(audit?.passed ?? "—")}</strong>
          <span>verified sources</span>
        </div>
      </div>
      <div className="grid-1-1">
        <Panel>
          <PanelHeader
            eyebrow="CLAIM GATE"
            title="Required before publication"
          />
          {claimGate.map((item, index) => (
            <div className="gate-row" key={item}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <strong>{item}</strong>
              <CheckCircle2 size={15} />
            </div>
          ))}
        </Panel>
        <Panel className="prohibited-panel">
          <PanelHeader
            eyebrow="HARD PROHIBITIONS"
            title="What this system will not infer"
          />
          {prohibited.map((item) => (
            <div className="prohibited-row" key={item}>
              <ShieldCheck size={14} />
              <span>{item}</span>
            </div>
          ))}
        </Panel>
      </div>
      <Panel>
        <PanelHeader
          eyebrow="SOURCE REGISTRY"
          title={`${sources.length} registered source records`}
          meta={
            <a
              className="text-button"
              href="/api/governance/audit-ledger"
              target="_blank"
              rel="noreferrer"
            >
              Open audit ledger <ExternalLink size={13} />
            </a>
          }
        />
        <div className="source-table">
          <div className="table-head">
            <span>Source</span>
            <span>Tier</span>
            <span>Role</span>
            <span>Verified</span>
          </div>
          {sources.map((source, index) => (
            <div className="table-row" key={String(source.id ?? index)}>
              <strong>
                {String(source.title ?? source.id ?? "Registered source")}
              </strong>
              <Badge tone="blue">
                {String(source.tier ?? "UNCLASSIFIED")}
              </Badge>
              <span>
                {String(source.category ?? "Public source")}
              </span>
              <CheckCircle2 size={14} />
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function LocalWorkbenchApp() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [view, setView] = useState<View>(() => navItems.find(item => item.id === window.location.hash.slice(1))?.id ?? "overview");
  const [menuOpen, setMenuOpen] = useState(false);
  useEffect(() => {
    const change = () => setView(navItems.find(item => item.id === window.location.hash.slice(1))?.id ?? "overview");
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => { window.history.replaceState(null, "", `#${view}`); window.scrollTo(0, 0); }, [view]);
  useEffect(() => {
    loadDashboard()
      .then(setData)
      .catch((reason) =>
        setError(
          reason instanceof Error
            ? reason.message
            : "Unable to load research data",
        ),
      );
  }, []);
  if (error)
    return (
      <div className="error-screen">
        <ShieldCheck size={28} />
        <span className="mono-label">WORKSPACE ERROR</span>
        <h1>Research data could not be loaded.</h1>
        <p>{error}</p>
        <button
          className="primary-button"
          onClick={() => window.location.reload()}
        >
          Retry
        </button>
      </div>
    );
  if (!data) return <LoadingScreen />;
  const page =
    view === "overview" ? (
      <OverviewView data={data} navigate={setView} />
    ) : view === "evidence" ? (
      <EvidenceView claims={data.evidence} />
    ) : view === "analytics" ? (
      <AnalyticsView meta={data.meta} bayesian={data.bayesianMeta} />
    ) : view === "trials" ? (
      <TrialLabView meta={data.meta} trials={data.trials} />
    ) : view === "genomics" ? (
      <GenomicsView
        datasets={data.genomics}
        transcriptomics={data.transcriptomics}
        analysis={data.geneExpression}
      />
    ) : view === "molecular" ? (
      <MolecularWorkbench />
    ) : view === "fhir" ? (
      <FhirWorkbench bundle={data.fhir} />
    ) : view === "learning" ? (
      <LearningView learning={data.learning} />
    ) : (
      <GovernanceView governance={data.governance} />
    );
  return (
    <div className="app-shell">
      <Sidebar
        active={view}
        onChange={setView}
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
      />
      {menuOpen && (
        <button
          className="overlay"
          onClick={() => setMenuOpen(false)}
          aria-label="Close navigation"
        />
      )}
      <main className="main-shell">
        <Topbar
          current={view}
          onMenu={() => setMenuOpen(true)}
          overview={data.overview}
        />
        <div className="content">
          <div className="workspace-ruler">
            <span>CASE / SCA3-COMPASS</span>
            <span>
              UTC {new Date().toISOString().slice(0, 16).replace("T", " ")}
            </span>
            <span>VERSION 4.0 / RESEARCH</span>
          </div>
          {page}
        </div>
        <footer>
          <span>SCA3 COMPASS / PUBLIC-DATA RESEARCH WORKBENCH</span>
          <span>
            Research use only · no diagnosis · no treatment recommendation
          </span>
        </footer>
      </main>
    </div>
  );
}

function App() {
  const [hash, setHash] = useState(() => window.location.hash);
  useEffect(() => {
    const update = () => setHash(window.location.hash);
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  const publicOnly = import.meta.env.VITE_PUBLIC_REVIEW_ONLY === "true";
  if (publicOnly || !hash || hash === "#research" || hash === "#family" || hash === "#ep-content") {
    return <EvidencePortal family={hash === "#family"} publicOnly={publicOnly} />;
  }
  return <LocalWorkbenchApp />;
}

export default App;
