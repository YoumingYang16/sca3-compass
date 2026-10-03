export async function researchApi<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => null)
    const detail = payload?.detail
    const message = Array.isArray(detail) ? detail.map((item: { msg: string }) => item.msg).join('; ') : detail
    throw new Error(message || `Request failed (${response.status}). Please retry.`)
  }
  return response.json() as Promise<T>
}

export type Probability = { estimate: number; low: number; high: number; mc_se: number }
export type DesignCell = {
  id: string; participants_per_arm: number; duration_months: number; maximum_participants: number
  planned_participant_months: number; slope_difference_se: number; plug_in_power: number
  fixed_assurance: number; fixed_assurance_mc_se: number; sequential_assurance: Probability
  expected_information_fraction: number; early_efficacy_stop_probability: number
  minimum_scenario_fixed_assurance: number; pareto_efficient: boolean
  sensitivity: Array<{ tau_prior_scale: number; slope_sd_multiplier: number; fixed_assurance: number; mc_se: number }>
}
export type DesignRun = {
  run_id: string; created_at: string; sha256: string; integrity_verified: boolean
  reproducibility: { python: string; numpy: string; scipy: string; source_sha256: Record<string, string>; input_sha256: string }
  result: {
    method_version: string; request: { sample_sizes: number[]; durations: number[]; seed: number; simulations: number }
    calibration: { method: string; null_type1: Probability; naive_repeated_testing_type1: Probability; information_fractions: number[]; z_boundaries: number[]; calibration_draws: number; independent_validation_draws: number; quadrature_nodes?: number; doubled_resolution_alpha_discrepancy?: number; integrated_null_rejection?: number }
    cells: DesignCell[]; candidate: { design_id: string; rule: string } | null; candidate_message: string
    progression_prior_sensitivity: Array<{ tau_prior_scale: number; mean: number; low: number; high: number; probability_nonpositive: number }>
    input_contract: Record<string, string>; assumptions: string[]; interpretation_boundary: string
    sources: Array<{ study_id: string; cohort: string; effect: number; standard_error: number; source_url: string }>
  }
}
export type LearningSession = {
  session_id: string; completed_items: number; mastery: Record<string, number>; completion: string | null
  concepts: Array<{ id: string; title: string; prerequisites: string[]; mastery_threshold: number }>
  next_item: { id: string; concept_id: string; question_token: string; prompt: string; options: string[]; selection: { policy: string; review?: boolean; expected_information_gain_bits: number } } | null
  feedback: { correct: boolean; rationale: string; misconception: string | null; selected_option: string; correct_option: string; prompt: string; evidence_claim_id: string; review_assumption: string | null; update: { prior: number; posterior_after_observation: number; posterior_after_learning: number } } | null
  privacy: string; interpretation_boundary: string; model_sha256: string
}

export type PolicyEvaluation = {
  status: string; learners: number; steps: number; seed: number
  adaptive: { expected_posttest_accuracy: number }
  fixed_sequence: { expected_posttest_accuracy: number }
  paired_difference_intervals: { expected_posttest_accuracy: { mean: number; low: number; high: number } }
  comparison_design: string; comparator_policy: string; interpretation_boundary: string
}
