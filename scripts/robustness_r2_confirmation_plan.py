"""One pre-C2 plan using completed DEV only; refuses any existing C2 batch."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sca3_compass.robustness_io import content_digest
from robustness_r2_development import read, sha


def build():
    base = ROOT/'artifacts/robustness'
    registry = read(base/'EXPERIMENT_REGISTRY.json')
    if any(r.get('settings', {}).get('phase') == 'C2' for r in registry['experiments']):
        raise ValueError('C2 already consumed; cannot redesign after confirmation')
    old = read(base/'R0073-screen.protocol.json')
    dev = read(base/'R0074-analysis/summary.json')
    stress = read(base/'R0075-analysis/summary.json')
    if not dev['complete'] or not stress['complete']:
        raise ValueError('Completed audited development required')
    p1 = old['analysis_plan']
    cases = old['cases'] + [r['case'] for r in stress['rows']]
    rows = dev['rows'] + stress['rows']
    candidate = 'R2K_pilotc0.5_projection_gate_eBH'
    reference = p1['candidate']
    names = [m for m in dev['method_labels'] if not m.startswith('R2P_')]
    if len(names) != 124 or len(set(names)) != 124:
        raise ValueError('Fixed 88 reference +36 K labels required')
    families = {'H': p1['families']['H']}
    for f in ['F_safe', 'F_empirical']:
        families[f+'_same_K'] = [m.replace('R1B_guard_', 'R2K_') for m in p1['families'][f]]
    simple = p1['families']['F_safe'] + p1['families']['F_empirical']
    families['F_all_same_information'] = (simple + [m.replace('R1B_guard_', 'R1B_plugin_') for m in simple]
                                         + [m.replace('R1B_guard_', 'R2K_') for m in simple])
    selected = {f: {str(i): max(members, key=lambda m: rows[i]['methods'][m]['power'])
                   for i in range(54)} for f, members in families.items()}
    power_cases = [i for i, r in enumerate(rows) if r['methods'][candidate]['power'] is not None]
    counts = [800 if i in power_cases else 2000 for i in range(len(cases))]
    assert len(cases) == 84 and sum(counts) == 82800 and len(power_cases) == 71
    allocations = {'envelopes': .006, 'I_strata': .004, 'I_local': .003,
                   'FDR_candidate_and_reference': .008, 'FDR_other': .004}
    precision = {}
    for label, indices in p1['strata'].items():
        d = dev['comparisons'][label+'/K']['I_vs_R1']
        means = np.array([rows[i]['methods'][candidate]['power']-rows[i]['methods'][reference]['power'] for i in indices])
        within = d['paired_mcse']**2 * len(indices) * 100
        pooled = within + float(means.var())
        n = 800 * len(indices)
        log = np.log(2/(allocations['I_strata']/6))
        precision[label] = {'DEV_I': d['mean_difference'],
            'projected_paired_mcse': d['paired_mcse']/np.sqrt(8),
            'projected_EB_radius': float(np.sqrt(2*pooled*log/n)+14*log/(3*(n-1))),
            'envelope_lower_radius': float(np.sqrt(2*np.log(24/allocations['envelopes'])/n))}
    return {'phase': 'C2', 'method_version': 'R2K-rho-only-rc1', 'run_id': 'R0076',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'candidate': candidate, 'reference': reference,
        'candidate_source_sha256': sha(ROOT/'src/sca3_compass/robustness_calibration_efficiency.py'),
        'reference_source_sha256': p1['candidate_source_sha256'],
        'candidate_change_since_DEV': 'None. K only; no hybrid, gate or parameter change.',
        'cases': cases, 'case_definition_digest': content_digest(cases),
        'repetition_counts': counts, 'power_defined_indices': power_cases,
        'original_validity_scope_indices': p1['validity_scope_indices'],
        'expanded_matched_indices': list(range(76,84)),
        'scope_note': 'Original68 working cases and eight newly defined smaller-calibration matched cases separate. Original eight assumption violations retained, not reclassified.',
        'seed': 7605229, 'bootstrap_draws': 16, 'workers': 8, 'batch_size': 4,
        'algorithm_nominal_fdr': .05, 'reporting_error_budget': .025,
        'error_allocation': allocations, 'reported_methods': names, 'families': families,
        'strata': p1['strata'], 'development_selected_baselines': selected,
        'development_evidence_sha256': {str(p.relative_to(ROOT)).replace('\\','/'): sha(p) for p in
            [base/'R0074-analysis/summary.json',base/'R0075-analysis/summary.json',base/'R0073-confirmation-analysis.json']},
        'C1_notice': 'C1 informed R2 selection and is R2 DEV, never reused as R2 confirmation. Frozen R1 C1 inference is preserved.',
        'P_notice': 'P == stored R1B_plugin output, exact parity already tested on11600 DEV families; retain P and its strong simple baselines. DEV8% FDR at t5/n4 composite null blocks promotion, not a universal invalidity theorem.',
        'numerical_reuse': 'Same-family freshly fitted R1 successful nuisance/calibration bank may memoize K; no external DEV or C1 fits. Any incomplete R1 bank forces ordinary fresh K evaluation. Verify exact fresh path replay.',
        'precision_from_DEV_not_guarantee': precision,
        'fixed_size_reason': '800 per defined-Power scene targets sub0.3pp I radius and about2pp core/2.8pp heavy envelope radius from DEV variance. 2000 per undefined-Power/composite-null scene resolves small false-discovery probabilities. Local I intervals may remain wide. Counts fixed, no significance-driven additions.',
        'cost_plan': '82800 families; DEV cost roughly1.34CPU s/family before same-family memoization.8 single-BLAS workers; finite21600s generation cap and3600s analysis cap, resumable identities. No GPU kernels, new dependency or paid resources.',
        'analysis': {
            'envelope': 'K only vs4 fixed families x3 strata x2 endpoints, Jensen-McDiarmid lower / DEV-selected weighted-Hoeffding upper; unchanged H estimand.',
            'I': 'Frozen K minus frozen R1 on SAME C2 families. Maurer-Pontil2009 Thm11 range2; equal counts pooling independent non-iid whole families.3 strata x2 endpoints and71 local x2 separately allocated.',
            'FDR': 'E[FP/max(1,R)] per scene. Inverse binary-KL Chernoff independent bounded FDP; candidate+reference2x84x2 endpoints share.008, other122x84x2 share.004. No truth-dependent threshold tuning.',
            'undefined_power': 'Report null Power as null/undefined, never zero, never include in macro.',
            'descriptive': 'All meanTP, quantiles, MCSE, fallback counts, time and local point differences descriptive unless an explicitly allocated interval.'},
        'retention_rule': 'Retain K as incremental scoped engineering/empirical variant only if core I lower>0, original and expanded matched candidate FDR upper<=.05, complete provenance/replay, and local/stratum losses/cost fully reviewed. Each superiority statement requires its own allocated lower>0. No invented minimum-gain/local tolerance. If incremental evidence fails, retain scoped frozen R1; do not call no-significance proof of no value. Broad success requires both core and t5 positive H and all-same-information F and applicable validity; scoped K gain cannot substitute.',
        'stop_rule': 'One fixed C2 only. No interim scientific analysis or extra draws, no method/analysis change after outputs. Incomplete batch descriptive only. C2 exhausts remaining .025; no third confirmation or candidate expansion.',
        'provenance': 'SIMULATION_NOT_PATIENT_DATA'}


if __name__ == '__main__':
    plan = build()
    target = ROOT/'configs/robustness_C2_analysis.json'
    with target.open('x', encoding='utf-8') as stream:
        json.dump(plan,stream,indent=2,ensure_ascii=False)
    print(target)
    print(json.dumps(plan['precision_from_DEV_not_guarantee'],indent=2))
