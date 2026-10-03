import { useState } from 'react'
import { CheckCircle2, ExternalLink, Play } from 'lucide-react'
import { researchApi } from './research-api'
import './workbench.css'

export type ResearchBundle = { resourceType: string; type: string; entry: Array<{ fullUrl: string; resource: { resourceType: string; id: string; [key: string]: unknown } }> }
type Outcome = { issue: Array<{ severity: string; diagnostics: string; expression?: string[] }> }

export function FhirWorkbench({ bundle }: { bundle: ResearchBundle }) {
  const [outcome, setOutcome] = useState<Outcome | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState(bundle.entry[0]?.fullUrl ?? '')
  const counts = bundle.entry.reduce<Record<string, number>>((acc, entry) => { acc[entry.resource.resourceType] = (acc[entry.resource.resourceType] ?? 0) + 1; return acc }, {})
  const entry = bundle.entry.find(row => row.fullUrl === selected)
  const passed = outcome && outcome.issue.every(issue => !['error','fatal'].includes(issue.severity))
  async function validate() {
    setBusy(true); setError('')
    try { setOutcome(await researchApi<Outcome>('/api/fhir/$validate', 'POST', bundle)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Validation failed') }
    finally { setBusy(false) }
  }
  return <section className="research-workbench">
    <header className="wb-title"><span className="wb-kicker">05 / RESEARCH INFORMATICS</span><h1>Exchange evidence. Preserve meaning.</h1><p>Inspect public registry and evidence resources through a read-only FHIR R5 research projection. Structure, study progress and source lineage remain separate and testable.</p></header>
    <div className="wb-pipeline">{Object.entries(counts).map(([type, count]) => <div key={type}><span>R5 RESOURCE</span><strong>{count} {type}</strong><small>Counted by resource type, not bundle size</small></div>)}<div><span>RESOURCE COLLECTION</span><strong>{bundle.entry.length} total entries</strong><small>No patient-level clinical records</small></div></div>
    <div className="grid-1-1"><article className="wb-card"><span className="wb-kicker">INDEPENDENT STRUCTURAL VALIDATOR</span><h2>Validate the actual exported bundle.</h2><p>Uses fhir.resources 8.1.0 (FHIR 5.0.0), plus local vocabulary and invariant checks. No fabricated “valid” status appears before the check runs.</p><button className="primary-button" disabled={busy} onClick={validate}><Play size={14} />{busy ? 'Validating…' : 'Run R5 validation'}</button>{outcome && <div className={passed ? 'wb-boundary' : 'wb-error'} role="status"><strong>{passed ? 'Structural and focused checks passed' : 'Validation errors detected'}</strong>{outcome.issue.map((issue, index) => <p key={index}>{issue.diagnostics}{issue.expression && <code className="wb-digest">{issue.expression.join(', ')}</code>}</p>)}</div>}{error && <p className="wb-error" role="alert">{error}</p>}</article><article className="wb-card"><span className="wb-kicker">SEMANTIC CONTRACT</span><h2>Publication status ≠ trial progress.</h2><div className="wb-detail-lines"><div><span>ResearchStudy.status</span><b>Resource publication: active</b></div><div><span>ResearchStudy.progressStatus</span><b>Original registry status</b></div><div><span>Reported enrollment</span><b>Note; type not inferred</b></div><div><span>CapabilityStatement</span><b>Implemented operations only</b></div></div><p className="wb-small">Full terminology expansion, all FHIRPath invariants and external reference resolution are outside this validator. This is not HL7 certification or an EHR deployment.</p><a className="text-button" href="/api/fhir/metadata" target="_blank" rel="noreferrer">Inspect capability statement <ExternalLink size={14} /></a></article></div>
    <article className="wb-card"><div className="wb-card-heading"><div><span className="wb-kicker">RESOURCE & PROVENANCE INSPECTOR</span><h2>Read the interoperable record.</h2></div><a className="text-button" href="/api/fhir/research-bundle" target="_blank" rel="noreferrer">Export full bundle <ExternalLink size={14} /></a></div><label className="wb-resource-select">Resource<select value={selected} onChange={e => setSelected(e.target.value)}>{bundle.entry.map(row => <option key={row.fullUrl} value={row.fullUrl}>{row.resource.resourceType} / {row.resource.id}</option>)}</select></label>{entry && <><div className="wb-result-status"><span><CheckCircle2 size={12} /> Public-source projection</span><a href={entry.fullUrl} target="_blank" rel="noreferrer">Read endpoint <ExternalLink size={13} /></a></div><pre className="wb-resource-json">{JSON.stringify(entry.resource, null, 2)}</pre></>}</article>
  </section>
}
