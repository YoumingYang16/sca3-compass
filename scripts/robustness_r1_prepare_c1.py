"""Freeze C1 specification AFTER DEV completion and BEFORE any C1 draws.

Refuses incomplete development. Selection is only the baseline used for the
upper confidence bound; the candidate itself stays the predeclared c=.5 gate.
The lower bound compares the full per-case baseline envelope, not this mapping.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from sca3_compass.robustness_io import content_digest
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--development-run',required=True);a.add_argument('--core-repetitions',type=int,required=True);a.add_argument('--stress-repetitions',type=int,required=True);a.add_argument('--null-repetitions',type=int,required=True);a.add_argument('--workers',type=int,required=True);args=a.parse_args()
    folder=ROOT/'artifacts/robustness';run=args.development_run;protocol=read(folder/f'{run}-screen.protocol.json');summary=read(folder/f'{run}-summary.json')
    if not summary['complete'] or protocol['phase']!='development' or protocol['backend']!='bootstrap_guard':raise ValueError('Complete guard development required')
    if summary['records_verified']!=sum(protocol.get('repetition_counts',[protocol['repetitions']]*len(protocol['cases']))):raise ValueError('Development records incomplete')
    if not summary['full_arrays_verified']:raise ValueError('Verify full development arrays first')
    if not 2<=args.core_repetitions<=1000 or not 2<=args.stress_repetitions<=2000 or not 2<=args.null_repetitions<=2000:raise ValueError('Predeclared ceilings exceeded')
    cases=protocol['cases'];main=protocol['R1_main'];labels=list(summary['rows'][0]['methods'])
    families={'H':protocol['H_baselines'],
        'F_safe':[f'R1B_guard_pilotc{c}_{m}_eBH' for c in protocol['F_multipliers'] for m in protocol['F_safe_kernels']],
        'F_empirical':[f'R1B_guard_pilotc{c}_{m}_eBH' for c in protocol['F_multipliers'] for m in protocol['F_empirical_kernels']]}
    core=[i for i,c in enumerate(cases) if c['stratum']=='core']
    if len(core)!=54:raise ValueError('Historical54core required')
    strata={'core54':core,'normal27':[i for i in core if cases[i]['distribution']=='normal'],'t5_27':[i for i in core if cases[i]['distribution']=='t5']}
    mapping={family:{str(i):max(base,key=lambda m:summary['rows'][i]['methods'][m]['power']) for i in core} for family,base in families.items()}
    counts=[args.core_repetitions if i in core else (args.null_repetitions if summary['rows'][i]['methods'][main]['power'] is None else args.stress_repetitions) for i in range(len(cases))]
    # Scope chosen from model information, NEVER retrospectively from goodFDR.
    allowed={'matched_working_model','in_scope'}
    scope=[i for i,c in enumerate(cases) if c['assumption_class'] in allowed]
    pairs={
        'guard_cost_complex':('R1B_guard_pilotc0.5_projection_eBH','R1B_plugin_pilotc0.5_projection_eBH'),
        'guard_cost_simple':('R1B_guard_pilotc0.5_ordinary_bonf_eBH','R1B_plugin_pilotc0.5_ordinary_bonf_eBH'),
        'projection_extra_guard':('R1B_guard_pilotc0.5_projection_eBH','R1B_guard_pilotc0.5_ordinary_bonf_eBH'),
        'projection_extra_plugin':('R1B_plugin_pilotc0.5_projection_eBH','R1B_plugin_pilotc0.5_ordinary_bonf_eBH')}
    plan={'phase':'C1','created_utc':datetime.now(timezone.utc).isoformat(),'candidate':main,'method_version':'R1-bootstrap-guard-rc1',
        'candidate_source_sha256':sha(ROOT/'src/sca3_compass/robustness_bootstrap_guard.py'),
        'candidate_delta_from_development':'only failed-draw sampledindices added; completed computations unchanged',
        'case_definition_digest':content_digest(cases),'bootstrap_draws':16,'reporting_error_budget':.025,
        'error_allocation':{'envelopes':.012,'FDR_candidate':.006,'FDR_other':.004,'contributions':.003},
        'algorithm_nominal_fdr':.05,'families':families,'strata':strata,'reported_methods':labels,
        'development_selected_baselines':mapping,'contribution_pairs':pairs,'validity_scope_indices':scope,
        'repetition_counts':counts,'workers':args.workers,'development_run':run,
        'development_summary_sha256':sha(folder/f'{run}-summary.json'),
        'precision_estimate_from_development':{'macro_mcse_at_dev_H_winner':float(np.sqrt(sum(summary['rows'][i]['comparisons']['H']['paired_mcse_at_sample_winner']**2*summary['rows'][i]['n']/counts[i] for i in core))/54),
            'cpu_seconds_extrapolated':float(sum(summary['rows'][i]['cpu_seconds_total']/summary['rows'][i]['n']*counts[i] for i in range(len(cases)))),
            'warning':'MCSE at selected development comparator is planning precision, NOT final selection-aware interval'},
        'stop_rule':'All fixed planned families complete, or hard resource/time cutoff; incomplete result descriptive only. No outcome monitoring or added draws.',
        'promotion_rule':'Limited empirical closure only if engineering/replay complete, candidate scoped FDR simultaneous upper endpoints<=.05, and fair H/F results fully reported; positive advantage claimed only for the corresponding lower endpoint>0. No inherited2.4 guarantee or minimumgain floor.',
        'C2_reservation':{'budget':.025,'use':'R2 after scopedR1closure OR substantively repairedR1.1 after C1failure, exclusively; no third attempt'},
        'provenance':'SIMULATION_NOT_PATIENT_DATA'}
    out=ROOT/'configs/robustness_C1_analysis.json'
    with out.open('x',encoding='utf-8') as s:json.dump(plan,s,indent=2)
    print('C1plan frozen; NO C1samples started.',out,'families',sum(counts),'scope',len(scope));print(plan['precision_estimate_from_development'])
if __name__=='__main__':main()
