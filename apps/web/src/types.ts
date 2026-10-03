export type Overview = {
  trial_candidates: number
  exact_sca3_matches: number
  records_with_results: number
  registered_public_sources: number
  verified_public_sources: number
  regions: number
  governance: Record<string, string>
  research_methods?: Record<string, string>
}

export type TrialSummary = {
  total: number
  statuses: Record<string, number>
  phases: Record<string, number>
  intervention_types: Record<string, number>
  provenance_tier: string
}

export type Trial = {
  nct_id: string
  brief_title: string
  overall_status: string
  phases: string
  enrollment: number | null
  intervention_types: string
  primary_outcomes: string
  countries: string
  has_results: boolean
  record_url: string
}

export type MetaStudy = {
  study_id: string
  cohort: string
  effect: number
  standard_error: number
  ci_low: number
  ci_high: number
  weight_percent: number
  sample_size: number | null
  source_url: string
}

export type MetaAnalysis = {
  method: string
  k: number
  pooled_effect: number
  standard_error: number
  ci_low: number
  ci_high: number
  prediction_low: number
  prediction_high: number
  tau_squared: number
  i_squared_percent: number
  outcome: string
  forest: MetaStudy[]
  leave_one_out: Array<{
    omitted_study_id: string
    pooled_effect: number
    ci_low: number
    ci_high: number
  }>
  interpretation_boundary: string
}

export type BayesianMetaAnalysis = {
  method: string
  k: number
  posterior_draws: number
  pooled_effect: { mean: number; median: number; low: number; high: number }
  heterogeneity_tau: { mean: number; median: number; low: number; high: number }
  new_setting_prediction: { mean: number; median: number; low: number; high: number }
  probability_progression_positive: number
  probability_progression_above_one: number
  priors: Record<string, string>
  diagnostics: { integration_grid_points: number; posterior_boundary_mass: number; algorithm: string }
  prior_sensitivity: Array<{ tau_prior_scale: number; pooled_median: number; pooled_low: number; pooled_high: number; tau_median: number }>
  interpretation_boundary: string
}

export type EvidenceClaim = {
  id: string
  topic: string
  title: string
  plain_language: string
  boundary: string
  source_title: string
  source_url: string
  provenance_tier: string
  evidence_type: string
}

export type LearningModule = {
  id: string
  number: string
  title: string
  description: string
  objectives: string[]
  estimated_minutes: number
  mastery_threshold: number
}

export type SimulationResult = {
  design: string
  seed: number
  completed_simulations: number
  power: number
  target_mean_difference: number
  mean_estimated_difference: number
  bias: number
  mean_standard_error: number
  mean_analyzed_participants: number
  assumptions: Record<string, number>
  provenance_tier: string
  interpretation_boundary: string
}

export type TranscriptomicsAnalysis = {
  analysis_status: string
  provenance_tier: string
  metrics: {
    sample_files?: number
    retained_transcripts?: number
    sca3_donors?: number
    control_donors?: number
    fdr_005?: number
    pca_variance_percent?: number[]
  }
  pca: Array<{ donor_id: string; genotype: string; pc1: number; pc2: number }>
  top_signals: Array<{
    gene_name: string
    transcript_id: string
    mean_sca3_tpm: number
    mean_control_tpm: number
    log2_fold_change: number
    p_value: number
    q_value: number
  }>
  interpretation_boundary?: string
  limitations?: string[]
  source?: { sha256?: string }
}

export type GeneExpressionAnalysis = {
  analysis_status: string
  provenance_tier: string
  research_question?: string
  method?: Record<string, string | string[]>
  metrics: {
    quantified_transcripts?: number
    mapped_genes?: number
    transcript_mapping_rate?: number
    abundance_mapping_rate?: number
    sample_files?: number
    donors?: number
    sca3_donors?: number
    control_donors?: number
    genes_before_filter?: number
    genes_after_filter?: number
    fdr_005?: number
    age_adjusted_fdr_005?: number
    pca_variance_percent?: number[]
    primary_model?: {
      design?: string[]
      residual_df?: number
      condition_number?: number
      empirical_bayes_prior_df?: number
    }
    sensitivity_model?: { residual_df?: number; condition_number?: number }
    assignment_calibration?: {
      enumerated_assignments?: number
      median_fdr_discoveries?: number
      max_statistic_tail_probability?: number
      null_max_statistic_p95?: number
    }
  }
  pca: Array<{
    donor_id: string
    genotype: string
    sex: string
    time: string
    pc1: number
    pc2: number
  }>
  top_signals: Array<{
    gene_id: string
    gene_name: string
    gene_type: string
    mean_sca3_cpm: number
    mean_control_cpm: number
    log2_fold_change: number
    moderated_t: number
    p_value: number
    q_value: number
    age_adjusted_log2_fold_change: number
    age_adjusted_q_value: number
    direction_concordant: boolean
    leave_one_out_direction_agreement?: number
  }>
  interpretation_boundary?: string
  limitations?: string[]
  sources?: {
    geo_archive?: { sha256?: string; bytes?: number }
    gencode_annotation?: { sha256?: string; bytes?: number }
  }
}
