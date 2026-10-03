import type {
  EvidenceClaim,
  BayesianMetaAnalysis,
  GeneExpressionAnalysis,
  LearningModule,
  MetaAnalysis,
  Overview,
  SimulationResult,
  Trial,
  TrialSummary,
  TranscriptomicsAnalysis,
} from './types'
import type { ResearchBundle } from './FhirWorkbench'

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`)
  return response.json() as Promise<T>
}

export async function loadDashboard() {
  const [overview, trialSummary, trials, meta, bayesianMeta, evidence, learning, genomics, transcriptomics, geneExpression, governance, fhir] =
    await Promise.all([
      getJson<Overview>('/api/overview'),
      getJson<TrialSummary>('/api/trials/summary'),
      getJson<{ items: Trial[] }>('/api/trials?limit=100'),
      getJson<MetaAnalysis>('/api/analytics/meta-analysis'),
      getJson<BayesianMetaAnalysis>('/api/analytics/bayesian-meta-analysis'),
      getJson<{ claims: EvidenceClaim[] }>('/api/evidence'),
      getJson<{ modules: LearningModule[]; learning_theories: string[] }>('/api/learning/modules'),
      getJson<{ items: Array<Record<string, string | number>> }>('/api/genomics/datasets'),
      getJson<TranscriptomicsAnalysis>('/api/genomics/gse309548-analysis'),
      getJson<GeneExpressionAnalysis>('/api/genomics/gse309548-gene-analysis'),
      getJson<Record<string, unknown>>('/api/governance'),
      getJson<ResearchBundle>('/api/fhir/research-bundle'),
    ])
  return {
    overview,
    trialSummary,
    trials: trials.items,
    meta,
    bayesianMeta,
    evidence: evidence.claims,
    learning,
    genomics: genomics.items,
    transcriptomics,
    geneExpression,
    governance,
    fhir,
  }
}

export async function runSimulation(payload: Record<string, number>): Promise<SimulationResult> {
  const response = await fetch('/api/analytics/trial-simulation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<SimulationResult>
}

export async function loadBayesianMeta(): Promise<BayesianMetaAnalysis> {
  return getJson<BayesianMetaAnalysis>('/api/analytics/bayesian-meta-analysis')
}

export async function queryEvidence(query: string, top_k = 3): Promise<Record<string, any>> {
  const response = await fetch('/api/evidence/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, top_k }),
  })
  if (!response.ok) throw new Error(await response.text())
  return response.json() as Promise<Record<string, any>>
}
