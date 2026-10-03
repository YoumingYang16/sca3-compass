import { useEffect, useState } from "react";
import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import "./molecular.css";
import { AnimalBlockedValidation, CalibrationUncertainty, ExternalTransport } from "./MolecularValidation";

type Dataset = { accession: string; role: string; family: string; organism: string; title: string; samples: number | null; resolved_units: number | null; features: number | null; qc_status: string; source_url: string; issues: string[] };
type Overview = { datasets: Dataset[]; relationships: { source: string; target: string; recorded_shared_unit_ids: number | null }[]; acquisition_summary: { download_complete: number; sealed_candidates: number }; qc_summary: { matrix_qc_complete: number; external_validation_certified: number }; registry_current: boolean; qc_current: boolean; release_gates: { name: string; status: string; detail?: string }[]; protocol: { protocol_id: string; alpha: number }; boundary: string };
type Metric = { mean: number; mc_se: number; ci95: number[] };
type Result = { scenario: string; method: string; fdr: Metric; power: Metric; repetitions: number; within_candidate_assumptions: boolean; paired_power_vs_bonferroni: Metric };
type Benchmark = { rows: Result[]; current: boolean; phase: string; uncertainty: string; run_digest: string; calibration_minimum_p: number; limits: string[] };
type Profile = { registration: { files: { kind: string; sha256: string; bytes: number; url?: string }[] }; qc: { status: string; issues: string[]; pca?: { sample_id: string; x: number; y: number; tissue: string; genotype: string }[]; pca_variance?: number[]; samples?: { sample_id: string; unit_id: string | null; tissue: string; genotype: string; library_total: number }[] } };
type Feature = { feature: string; effects: number[]; standard_errors: number[]; posterior_mean: number[]; posterior_sd: number[]; lfsr: number[] };
type Shrinkage = { genes: number; independent_animals: number; regions: string[]; em: { converged: boolean; iterations: number; simplex_kkt_residual: number }; mixture: { component: string; weight: number }[]; examples: Feature[]; estimated_noise_correlation: number[][]; overlap_matrix: number[][]; current: boolean; simulation: { rows: { method: string; mse: number; training_converged?: boolean }[] }; limits: string[] };

async function get<T>(path: string, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok) throw new Error(`Research endpoint unavailable (${response.status})`);
  return response.json() as Promise<T>;
}
const clean = (text: string) => text.replaceAll("_", " ");
const pct = (value: number) => `${(value * 100).toFixed(2)}%`;
const colors = ["#a0563c", "#879695", "#a48750", "#235b75", "#785c89"];
const methods = ["naive_min_PC_BH", "pipeline_bonferroni_PC_BY", "fixed_pipeline_PC_BY", "maxT_PC_BY", "mixture_e_PC_eBH"];

function EffectPlot({ feature, regions }: { feature: Feature; regions: string[] }) {
  const extent = Math.max(0.5, ...feature.effects.map((v, i) => Math.abs(v) + feature.standard_errors[i]), ...feature.posterior_mean.map((v, i) => Math.abs(v) + feature.posterior_sd[i])) * 1.15;
  const x = (v: number) => 420 + v / extent * 210;
  return <svg className="mol-effects" viewBox="0 0 700 270" role="img" aria-label="Observed effects plus or minus standard error, posterior effects plus or minus standard deviation">
    <line x1={x(0)} x2={x(0)} y1={20} y2={225} stroke="#aebac1" strokeDasharray="4 4" />
    {regions.map((region, i) => <g key={region}>
      <text x="12" y={44 + 48 * i} fill="#304857" fontSize="13">{region}</text>
      <line x1={x(feature.effects[i] - feature.standard_errors[i])} x2={x(feature.effects[i] + feature.standard_errors[i])} y1={34 + 48 * i} y2={34 + 48 * i} stroke="#ac916e" strokeWidth="2" />
      <circle cx={x(feature.effects[i])} cy={34 + 48 * i} r="4" fill="#ac916e" />
      <line x1={x(feature.posterior_mean[i] - feature.posterior_sd[i])} x2={x(feature.posterior_mean[i] + feature.posterior_sd[i])} y1={47 + 48 * i} y2={47 + 48 * i} stroke="#235b75" strokeWidth="3" />
      <circle cx={x(feature.posterior_mean[i])} cy={47 + 48 * i} r="4" fill="#235b75" />
    </g>)}
    {[-extent, 0, extent].map(tick => <text key={tick} x={x(tick)} y="245" textAnchor="middle" fill="#667985" fontSize="11">{tick.toFixed(2)}</text>)}
    <text x="420" y="265" textAnchor="middle" fill="#667985" fontSize="11">Difference in mean log2(sum-scaled expression + 0.5)</text>
  </svg>;
}

export function MolecularWorkbench() {
  const [overview, setOverview] = useState<Overview>();
  const [benchmark, setBenchmark] = useState<Benchmark>();
  const [shrinkage, setShrinkage] = useState<Shrinkage>();
  const [profile, setProfile] = useState<Profile>();
  const [tab, setTab] = useState("Data & provenance");
  const [accession, setAccession] = useState("GSE107958");
  const [scenario, setScenario] = useState("shared_controls");
  const [phase, setPhase] = useState("validation");
  const [featureIndex, setFeatureIndex] = useState(0);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([get<Overview>("/api/molecular/overview", controller.signal), get<Shrinkage>("/api/molecular/shrinkage", controller.signal)])
      .then(([a, b]) => { setOverview(a); setShrinkage(b); }).catch(e => { if (e.name !== "AbortError") setError(String(e)); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    setBenchmark(undefined);
    get<Benchmark>(`/api/molecular/benchmark?phase=${phase}`, controller.signal).then(setBenchmark).catch(e => { if (e.name !== "AbortError") setError(String(e)); });
    return () => controller.abort();
  }, [phase]);
  useEffect(() => {
    const controller = new AbortController();
    setProfile(undefined);
    get<Profile>(`/api/molecular/datasets/${accession}`, controller.signal).then(setProfile).catch(e => { if (e.name !== "AbortError") setError(String(e)); });
    return () => controller.abort();
  }, [accession]);
  if (error) return <section className="mol-workbench"><h1>Research artifacts unavailable</h1><p role="alert">{error}</p><p>No missing result is treated as a successful check.</p></section>;
  if (!overview || !shrinkage) return <section className="mol-workbench" aria-live="polite">Loading verified research artifacts…</section>;
  const selected = overview.datasets.find(d => d.accession === accession)!;
  const rows = benchmark?.rows.filter(row => row.scenario === scenario) ?? [];
  const feature = shrinkage.examples[featureIndex];
  return <div className="mol-workbench">
    <header className="mol-heading"><span className="mol-eyebrow">SCA3 COMPASS / METHODS RESEARCH</span><h1>Molecular reliability</h1><p>Sample provenance, dependence-aware inference and multivariate effect estimation.</p><div className="mol-status">Research development · Original contribution not yet established · No clinical use</div></header>
    {(!overview.registry_current || !overview.qc_current || (benchmark && !benchmark.current) || !shrinkage.current) && <div className="mol-warning" role="alert">One or more results are stale relative to their source or protocol. Rerun the registered workflow before interpreting them.</div>}
    <div className="mol-summary"><div><strong>{overview.acquisition_summary.download_complete}</strong><span>development / reserve datasets</span></div><div><strong>{overview.qc_summary.matrix_qc_complete}</strong><span>development matrices checked</span></div><div><strong>{overview.acquisition_summary.sealed_candidates}</strong><span>reserve expression set sealed</span></div><div><strong>0</strong><span>clinical validations · research only</span></div></div>
    <nav className="mol-tabs" aria-label="Molecular research sections">{["Data & provenance", "Operating characteristics", "Calibration uncertainty", "Multiregion model", "External cohort", "Research gates"].map(label => <button key={label} aria-current={tab === label ? "page" : undefined} onClick={() => setTab(label)}>{label}</button>)}</nav>
    {tab === "Calibration uncertainty" && <CalibrationUncertainty />}
    {tab === "External cohort" && <ExternalTransport />}
    {tab === "Data & provenance" && <>
      <section className="mol-panel"><div className="mol-panel-head"><h2>Registered evidence</h2><span>Dataset ≠ sample ≠ independent subject</span></div><div className="mol-scroll"><table><thead><tr><th>Accession</th><th>Organism</th><th>Samples</th><th>Resolved units</th><th>Role</th><th>QC state</th></tr></thead><tbody>{overview.datasets.map(d => <tr key={d.accession} className={accession === d.accession ? "selected" : ""}><td><button className="mol-link" onClick={() => setAccession(d.accession)}>{d.accession}</button></td><td>{d.organism}</td><td>{d.samples ?? "Unknown"}</td><td>{d.resolved_units ?? "Unresolved"}</td><td>{clean(d.role)}</td><td>{clean(d.qc_status)}</td></tr>)}</tbody></table></div></section>
      <div className="mol-columns"><section className="mol-panel"><span className="mol-eyebrow">{accession}</span><h2>{selected.title}</h2><a href={selected.source_url} target="_blank" rel="noreferrer">Open original GEO record ↗</a><ul className="mol-issues">{selected.issues.map(issue => <li key={issue}>{issue}</li>)}</ul><h3>Documented relationships</h3>{overview.relationships.map(link => <p className="mol-relationship" key={`${link.source}-${link.target}`}><b>{link.source} — {link.target}</b><span>{link.recorded_shared_unit_ids === null ? "Related family; overlap unresolved" : `${link.recorded_shared_unit_ids} shared animal IDs in source titles`}</span></p>)}<p className="mol-note">Missing identities remain unresolved. Similar demographics are not certified donor identifiers.</p></section>
      <section className="mol-panel"><h2>Expression-space QC</h2>{profile?.qc.pca ? <><div className="mol-chart"><ResponsiveContainer><ScatterChart><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" dataKey="x" name="PC1" tick={{ fontSize: 11 }} /><YAxis type="number" dataKey="y" name="PC2" tick={{ fontSize: 11 }} /><Tooltip content={({ payload }) => { const item = payload?.[0]?.payload; return item ? <div className="mol-tooltip"><b>{item.sample_id}</b><p>{item.tissue}</p><p>{item.genotype}</p></div> : null; }} />{[...new Set(profile.qc.pca.map(p => p.tissue))].map((tissue, i) => <Scatter key={tissue} name={tissue} data={profile.qc.pca!.filter(p => p.tissue === tissue)} fill={colors[i % colors.length]} />)}</ScatterChart></ResponsiveContainer></div><p className="mol-note">PC1 {pct(profile.qc.pca_variance?.[0] ?? 0)} · PC2 {pct(profile.qc.pca_variance?.[1] ?? 0)}. Label-free QC, not diagnostic separation.</p></> : <div className="mol-empty">{profile ? clean(profile.qc.status) : "Loading…"}<p>No expression plot is produced for sealed or unprocessed data.</p></div>}</section></div>
      <section className="mol-panel"><h2>Source integrity</h2>{profile?.registration.files.map(file => <div className="mol-receipt" key={file.sha256}><b>{file.kind}</b><span>{(file.bytes / 1e6).toFixed(2)} MB</span><code>{file.sha256}</code></div>)}</section>
    </>}
    {tab === "Operating characteristics" && <>
      <section className="mol-panel"><div className="mol-controls"><label>Run phase<select value={phase} onChange={e => setPhase(e.target.value)}><option value="validation">Separate-seed simulation validation</option><option value="development">Development simulation</option></select></label><label>Scenario<select value={scenario} onChange={e => setScenario(e.target.value)}>{[...new Set(benchmark?.rows.map(row => row.scenario) ?? [])].map(name => <option key={name} value={name}>{clean(name)}</option>)}</select></label></div><p className="mol-note">Synthetic truth only. These are 2G directional hypotheses. Separate random seeds do not constitute external biological validation.</p>{rows.length > 0 && !rows[0].within_candidate_assumptions && <p className="mol-warning">Stress test outside the candidate's stated Gaussian/covariance assumptions. A favorable result does not extend its theoretical guarantee.</p>}
      <div className="mol-chart"><ResponsiveContainer><ScatterChart margin={{ left: 15, right: 30, top: 15, bottom: 20 }}><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" dataKey="powerValue" domain={[0, 1]} padding={{ left: 6, right: 10 }} name="Power" tickFormatter={pct} label={{ value: "Power", position: "insideBottom", offset: -12 }} tick={{ fontSize: 11 }} /><YAxis type="number" dataKey="fdrValue" domain={[0, 0.12]} name="Mean FDP" tickFormatter={pct} tick={{ fontSize: 11 }} /><ReferenceLine y={0.05} stroke="#a0563c" strokeDasharray="6 4" label={{ value: "5% target", position: "insideTopLeft", fontSize: 11 }} /><Tooltip content={({ payload }) => { const item = payload?.[0]?.payload; return item ? <div className="mol-tooltip"><b>{clean(item.method)}</b><p>Mean FDP: {pct(item.fdrValue)}</p><p>Power: {pct(item.powerValue)}</p></div> : null; }} />{methods.map((method, i) => <Scatter key={method} name={method} fill={colors[i]} data={rows.filter(r => r.method === method).map(r => ({ ...r, powerValue: r.power.mean, fdrValue: r.fdr.mean }))} />)}</ScatterChart></ResponsiveContainer></div>
      <div className="mol-scroll"><table><thead><tr><th>Method</th><th>Mean FDP</th><th>Pointwise 95% MC interval</th><th>Power</th><th>Runs</th></tr></thead><tbody>{rows.map(row => <tr key={row.method}><td><i className="mol-dot" style={{ background: colors[methods.indexOf(row.method)] }} />{clean(row.method)}</td><td>{pct(row.fdr.mean)}</td><td>{row.fdr.ci95.map(pct).join(" – ")}</td><td>{pct(row.power.mean)}</td><td>{row.repetitions}</td></tr>)}</tbody></table></div><p className="mol-note">{benchmark?.uncertainty}</p><p className="mol-note">maxT is calibrated using specified covariance, not an independently learned real-world covariance. No existing component is presented as a new invention.</p><code className="mol-digest">{benchmark?.run_digest}</code></section>
    </>}
    {tab === "Multiregion model" && <>
      <section className="mol-panel"><span className="mol-eyebrow">MULTIVARIATE EMPIRICAL BAYES / REFERENCE MODEL</span><h2>Shared and region-specific effects</h2><p>{shrinkage.genes.toLocaleString()} genes across four brain regions, from {shrinkage.independent_animals} mice. A covariance-mixture prior jointly estimates effect patterns; pooled residual correlation is a plug-in estimate.</p><p className="mol-equation">β<sub>g</sub> ∼ ∑ π<sub>k</sub> N(0, U<sub>k</sub>) &nbsp; · &nbsp; b̂<sub>g</sub> | β<sub>g</sub> ∼ N(β<sub>g</sub>, V<sub>g</sub>)</p><p className="mol-note">mash-inspired reference implementation, not a new mash method. EM plus constrained optimization: {shrinkage.em.converged ? "converged" : "NOT CONVERGED"}; KKT residual {shrinkage.em.simplex_kkt_residual.toExponential(2)}.</p></section>
      <div className="mol-columns"><section className="mol-panel"><label className="mol-feature">Feature example<select value={featureIndex} onChange={e => setFeatureIndex(Number(e.target.value))}>{shrinkage.examples.map((item, i) => <option key={item.feature} value={i}>{item.feature}</option>)}</select></label><EffectPlot feature={feature} regions={shrinkage.regions} /><p className="mol-note"><span style={{ color: "#ac916e" }}>Observed ± standard error</span> · <span style={{ color: "#235b75" }}>Posterior mean ± standard deviation</span>. These bars are not 95% confidence intervals. Examples are selected by fitted effect magnitude, not validated biological importance.</p></section>
      <section className="mol-panel"><h2>Animal overlap by region</h2><div className="mol-scroll"><table className="mol-matrix"><thead><tr><th>Animals</th>{shrinkage.regions.map(r => <th key={r}>{r}</th>)}</tr></thead><tbody>{shrinkage.overlap_matrix.map((row, i) => <tr key={i}><th>{shrinkage.regions[i]}</th>{row.map((value, j) => <td key={j} style={{ background: `rgba(35,91,117,${0.04 + value / 14 * 0.2})` }}>{value}</td>)}</tr>)}</tbody></table></div><p className="mol-warning">These regions share animals. Four regions must not be described as four independent replication cohorts.</p><h3>Frozen-generator simulation</h3>{shrinkage.simulation.rows.map(row => <p className="mol-line" key={row.method}><span>{clean(row.method)}</span><b>MSE {row.mse.toFixed(4)}{row.training_converged === false ? " · unconverged" : ""}</b></p>)}<p className="mol-note">Synthetic training/test sets; this generator was also used for numerical debugging. Not an external validation of the SCA3 findings.</p></section></div>
      <AnimalBlockedValidation />
      <section className="mol-panel"><h2>Model limits</h2><ul className="mol-issues">{shrinkage.limits.map(limit => <li key={limit}>{limit}</li>)}</ul></section>
    </>}
    {tab === "Research gates" && <section className="mol-panel"><h2>Research claim ledger</h2>{overview.release_gates.map(gate => <div className="mol-gate" key={gate.name}><h3>{gate.name}</h3><span>{clean(gate.status)}</span>{gate.detail && <p>{gate.detail}</p>}</div>)}<p className="mol-warning">Author-function parity, finite-calibration experiments, animal-blocked checks and a frozen public-cohort evaluation are available. None alone establishes a new method's priority, universal superiority or clinical validity.</p><p>{overview.boundary}</p></section>}
  </div>;
}
