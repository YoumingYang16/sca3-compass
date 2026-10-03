import { useEffect, useRef, useState } from 'react'
import { ArrowDownToLine, FlaskConical, Play, RefreshCw } from 'lucide-react'
import { researchApi, type DesignRun, type Probability } from './research-api'
import './workbench.css'

const pct = (value: number) => `${(100 * value).toFixed(1)}%`
const interval = (value: Probability) => `${pct(value.low)}–${pct(value.high)}`

export function DesignWorkbench() {
  const [settings, setSettings] = useState({ treatment_reduction: .3, effect_prior_concentration: 30, slope_sd: 1.5, attrition_rate: .15, simulations: 20000, seed: 202709 })
  const [run, setRun] = useState<DesignRun | null>(null)
  const [selected, setSelected] = useState('n80-m24')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dirty, setDirty] = useState(false)
  const [restored, setRestored] = useState(false)
  const runGeneration = useRef(0)
  const inputGeneration = useRef(0)
  useEffect(() => {
    let active = true
    researchApi<{ items: Array<{ run_id: string }> }>('/api/experiments')
      .then(async (index) => index.items[0] ? researchApi<DesignRun>(`/api/experiments/${index.items[0].run_id}`) : null)
      .then((value) => { if (active && value && runGeneration.current === 0) { setRun(value); setRestored(true) } })
      .catch(() => { /* An empty workbench remains usable if history cannot load. */ })
    return () => { active = false }
  }, [])
  const update = (key: keyof typeof settings, value: number) => { inputGeneration.current += 1; setSettings((current) => ({ ...current, [key]: value })); setDirty(true) }
  async function execute() {
    runGeneration.current += 1
    const submittedInputs = inputGeneration.current
    setBusy(true); setError('')
    try {
      const result = await researchApi<DesignRun>('/api/analytics/design-assurance', 'POST', settings)
      setRun(result); setDirty(inputGeneration.current !== submittedInputs); setRestored(false)
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Experiment failed') }
    finally { setBusy(false) }
  }
  const cell = run?.result.cells.find((item) => item.id === selected) ?? run?.result.cells[0]
  const calibration = run?.result.calibration
  return <section className="research-workbench" aria-label="Bayesian trial design workbench">
    <header className="wb-title"><span className="wb-kicker">03 / DECISION SCIENCE</span><h1>Design for uncertainty.</h1><p>From public natural-history evidence to an inspectable trial-design experiment. Compare fixed-assumption power, posterior predictive assurance and sequential monitoring.</p></header>
    <div className="wb-pipeline" aria-label="Analysis sequence">
      <div><span>01 / OBSERVED</span><strong>Published cohort estimates</strong><small>Study-level evidence, not patient rows</small></div>
      <div><span>02 / INFERRED</span><strong>New-setting posterior</strong><small>Heterogeneity + prior sensitivity</small></div>
      <div><span>03 / ASSUMED</span><strong>Treatment & design priors</strong><small>Investigator-specified, not efficacy data</small></div>
      <div><span>04 / SIMULATED</span><strong>Assurance & error control</strong><small>Independent null-validation stream</small></div>
    </div>
    <div className="wb-layout">
      <form className="wb-card wb-controls" onSubmit={(event) => { event.preventDefault(); void execute() }}>
        <div className="wb-card-heading"><div><span className="wb-kicker">DESIGN ASSUMPTIONS</span><h2>Experiment specification</h2></div><FlaskConical size={20} /></div>
        <label>Mean progression reduction <output>{pct(settings.treatment_reduction)}</output><input type="range" min={0} max={.6} step={.05} value={settings.treatment_reduction} onChange={(e) => update('treatment_reduction', +e.target.value)} /></label>
        <label>Effect-prior concentration<select value={settings.effect_prior_concentration} onChange={(e) => update('effect_prior_concentration', +e.target.value)}><option value={10}>10 / wider uncertainty</option><option value={30}>30 / moderate uncertainty</option><option value={100}>100 / tighter assumption</option></select></label>
        <p className="wb-small">Beta prior on proportional reduction. This optimistic prior excludes harmful effects; it is not learned from a treatment study.</p>
        <label>Individual slope SD<input type="number" min={.1} max={10} step={.1} required value={settings.slope_sd} onChange={(e) => update('slope_sd', +e.target.value)} /></label>
        <label>Information loss to attrition <output>{pct(settings.attrition_rate)}</output><input type="range" min={0} max={.4} step={.05} value={settings.attrition_rate} onChange={(e) => update('attrition_rate', +e.target.value)} /></label>
        <label>Posterior predictive replicates<select value={settings.simulations} onChange={(e) => update('simulations', +e.target.value)}><option value={5000}>5,000 / exploration</option><option value={20000}>20,000 / standard</option><option value={100000}>100,000 / precision</option></select></label>
        <label>Random seed<input type="number" min={0} max={2147483647} step={1} required value={settings.seed} onChange={(e) => update('seed', +e.target.value)} /></label>
        <div className="wb-mini-spec"><span>18 candidate designs</span><b>40–180 per arm · 12 / 24 / 36 months</b><span>12 sensitivity scenarios per design</span><b>4 heterogeneity priors × 3 slope SDs</b></div>
        <button className="primary-button wb-run" type="submit" disabled={busy}>{busy ? <RefreshCw size={16} className="spin" /> : <Play size={16} />}{busy ? 'Computing and validating…' : 'Run design experiment'}</button>
        {error && <p role="alert" className="wb-error">{error}</p>}
      </form>
      <div className="wb-results" aria-busy={busy}>
        {!run ? <div className="wb-card wb-empty"><span className="wb-kicker">AN EXPERIMENT, NOT A DEMO NUMBER</span><h2>Which assumptions change the decision?</h2><p>Run a reproducible design grid. Every result includes source estimates, numerical assumptions, Monte Carlo uncertainty and a content-addressed record.</p><div className="wb-equation">Assurance = E<sub>design prior</sub> [ P(reject H₀ | θ, design) ]</div><p>No patient records are generated and presented as observations.</p></div> : <>
          <div className="wb-result-status" role="status"><span>{busy ? 'New run in progress; previous result remains visible.' : dirty ? 'Inputs changed — these results belong to the previous run.' : restored ? 'Restored saved experiment; controls are defaults until you run again.' : 'Experiment complete — local artifact integrity verified.'}</span><a href={`/api/experiments/${run.run_id}`} download={`sca3-design-${run.run_id}.json`}><ArrowDownToLine size={14} /> Export run</a></div>
          <article className="wb-card">
            <div className="wb-card-heading"><div><span className="wb-kicker">POSTERIOR PREDICTIVE DESIGN SPACE</span><h2>Fixed-design assurance</h2></div><span className="wb-small">Click a cell to inspect</span></div>
            <div className="wb-table-scroll"><table className="wb-heatmap"><caption>Probability of rejecting the null under the stated design prior. Columns: participants per arm.</caption><thead><tr><th scope="col">Follow-up</th>{run.result.request.sample_sizes.map(n => <th key={n} scope="col">{n} / arm</th>)}</tr></thead><tbody>{run.result.request.durations.map(months => <tr key={months}><th scope="row">{months} months</th>{run.result.request.sample_sizes.map(n => { const item = run.result.cells.find(c => c.participants_per_arm === n && c.duration_months === months)!; return <td key={item.id}><button type="button" aria-pressed={cell?.id === item.id} aria-label={`${n} per arm, ${months} months, assurance ${pct(item.fixed_assurance)}`} style={{ backgroundColor: `rgba(26, 109, 117, ${.05 + item.fixed_assurance * .22})` }} onClick={() => setSelected(item.id)}><strong>{pct(item.fixed_assurance)}</strong><small>MC ±{(1.96 * item.fixed_assurance_mc_se * 100).toFixed(1)} pp</small></button></td> })}</tr>)}</tbody></table></div>
            <p className="wb-small">MC ± values summarize numerical integration error only. Clinical and modeling uncertainty is represented by the prior and scenarios, not by these tiny bands.</p>
          </article>
          {cell && <article className="wb-card wb-design-detail">
            <div className="wb-card-heading"><div><span className="wb-kicker">SELECTED / {cell.id.toUpperCase()}</span><h2>{cell.maximum_participants} participants · {cell.duration_months} months</h2></div><span className="wb-small">{cell.pareto_efficient ? 'Grid Pareto-efficient' : 'Dominated within this grid'}</span></div>
            <div className="wb-stat-grid"><div><span>Plug-in power</span><strong>{pct(cell.plug_in_power)}</strong><small>Fix all effects at their means</small></div><div><span>Fixed-design assurance</span><strong>{pct(cell.fixed_assurance)}</strong><small>Integrate effect & setting uncertainty</small></div><div><span>Sequential assurance</span><strong>{pct(cell.sequential_assurance.estimate)}</strong><small>95% MC interval {interval(cell.sequential_assurance)}</small></div></div>
            <div className="wb-detail-lines"><div><span>Minimum fixed assurance across 12 scenarios</span><b>{pct(cell.minimum_scenario_fixed_assurance)}</b></div><div><span>Early efficacy stopping probability</span><b>{pct(cell.early_efficacy_stop_probability)}</b></div><div><span>Expected fraction of maximum information</span><b>{pct(cell.expected_information_fraction)}</b></div><div><span>Assumed slope-difference standard error</span><b>{cell.slope_difference_se.toFixed(3)} points/year</b></div></div>
            <p className="wb-small">Information fractions are not calendar duration or recruited sample size. Canonical-normal monitoring is an approximation; informative dropout is not corrected here.</p>
            <details className="wb-disclosure"><summary>Inspect all sensitivity scenarios</summary><div className="wb-table-scroll"><table className="wb-data-table"><thead><tr><th>Tau prior scale</th><th>Slope SD multiplier</th><th>Fixed assurance</th><th>MC SE</th></tr></thead><tbody>{cell.sensitivity.map(row => <tr key={`${row.tau_prior_scale}-${row.slope_sd_multiplier}`}><td>{row.tau_prior_scale}</td><td>× {row.slope_sd_multiplier.toFixed(1)}</td><td>{pct(row.fixed_assurance)}</td><td>{pct(row.mc_se)}</td></tr>)}</tbody></table></div></details>
          </article>}
          {calibration && <article className="wb-card">
            <div className="wb-card-heading"><div><span className="wb-kicker">INDEPENDENT NULL EXPERIMENT</span><h2>Does repeated testing inflate false positives?</h2></div></div>
            <div className="wb-null-chart">{[{ label: 'Naive repeated testing', value: calibration.naive_repeated_testing_type1, tone: 'warning' }, { label: 'Calibrated sequential boundary', value: calibration.null_type1, tone: 'calibrated' }].map(row => <div className={`wb-null-row ${row.tone}`} key={row.label}><span>{row.label}</span><div className="wb-bar-track"><i style={{ width: `${Math.min(row.value.estimate / .1, 1) * 100}%` }} /><em style={{ left: '25%' }} title="Nominal one-sided alpha = 2.5%" /></div><b>{pct(row.value.estimate)}</b><small>95% MC {interval(row.value)}</small></div>)}</div>
            <p className="wb-small">Marker: nominal one-sided α = 2.5%. {calibration.quadrature_nodes ? `Boundary solved by ${calibration.quadrature_nodes}-node Gauss–Legendre integration, then checked at double resolution. ` : 'Saved run used a simulation-calibrated boundary. '}{calibration.calibration_draws.toLocaleString()} boundary cross-check draws and {calibration.independent_validation_draws.toLocaleString()} independent null-validation draws.</p>
            {calibration.quadrature_nodes && <div className="wb-detail-lines"><div><span>Integrated type-I rejection probability</span><b>{pct(calibration.integrated_null_rejection ?? 0)}</b></div><div><span>Double-resolution discrepancy</span><b>{calibration.doubled_resolution_alpha_discrepancy?.toExponential(2)}</b></div></div>}
            <div className="wb-boundaries">{calibration.information_fractions.map((fraction, i) => <div key={fraction}><span>Look {i+1} / {pct(fraction)} information</span><strong>Z &gt; {calibration.z_boundaries[i].toFixed(3)}</strong></div>)}</div>
          </article>}
          <article className="wb-card"><span className="wb-kicker">DECISION & REPRODUCIBILITY</span><h2>{run.result.candidate ? `Grid candidate: ${run.result.candidate.design_id}` : 'No scenario-robust candidate at the target.'}</h2><p>{run.result.candidate?.rule ?? run.result.candidate_message}</p><p className="wb-small">“Scenario-robust” applies only to these 12 assumptions and the fixed design, not every plausible model. Pareto status is not clinical validation.</p><details className="wb-disclosure"><summary>Sources, assumptions & computation manifest</summary><h3>Published inputs</h3>{run.result.sources.map(source => <p key={source.study_id}><a href={source.source_url} target="_blank" rel="noreferrer">{source.cohort}</a> · {source.effect.toFixed(2)} (SE {source.standard_error.toFixed(3)}) SARA/year</p>)}<h3>Explicit assumptions</h3><ul>{run.result.assumptions.map(item => <li key={item}>{item}</li>)}</ul><h3>Local artifact</h3><p>{new Date(run.created_at).toLocaleString()} · Seed {run.result.request.seed} · {run.result.request.simulations.toLocaleString()} predictive draws</p><code className="wb-digest">SHA-256 {run.sha256}</code><p>Python {run.reproducibility.python} · NumPy {run.reproducibility.numpy} · SciPy {run.reproducibility.scipy}</p><p className="wb-small">Hashes establish local integrity, not external timestamping or preregistration. Full source hashes are included in the exported JSON.</p></details><p className="wb-boundary">{run.result.interpretation_boundary}</p></article>
        </>}
      </div>
    </div>
  </section>
}
