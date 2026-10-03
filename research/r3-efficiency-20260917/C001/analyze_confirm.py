"""Frozen R3 confirmatory analysis; no interim method selection."""
import argparse
from pathlib import Path
import numpy as np
from r3_experiment import audit_rows
from r3_common import read,write,sha
from bounded_intervals import interval

def analyze(out):
    rows=audit_rows(out);p=read(out/'protocol.json');count=0;results=[]
    def ci(v,low=0.,high=1.):
        nonlocal count
        count+=1
        return interval(v,low,high,p['interval_cap'],p['alpha'])
    for c,case in enumerate(p['cases']):
        rr=[r for r in rows if r['case']==c]
        if len(rr)!=p['repetitions']:raise ValueError('repetition count')
        methods=p['methods']+(['R3_Delta5','R2_Delta5'] if case.get('outside_M0') else [])
        summary={'case':c,'design':case,'methods':{},'paired_power':{},'paired_boundary_fdr':{}}
        for method in methods:
            summary['methods'][method]={}
            for metric in ['power','fdp']:
                val=[r['metrics'][method][metric] for r in rr]
                summary['methods'][method][metric]=None if val[0] is None else ci(val)
            summary['methods'][method]['mean_tp']=float(np.mean([r['metrics'][method]['tp'] for r in rr]))
            summary['methods'][method]['mean_fp']=float(np.mean([r['metrics'][method]['fp'] for r in rr]))
            counts=[r['metrics'][method]['discoveries'] for r in rr]
            thresholds=[2*p['family_G']/(.05*v) for v in counts if v>0]
            summary['methods'][method]['positive_discovery_family_fraction']=float(np.mean(np.array(counts)>0))
            summary['methods'][method]['eBH_threshold_quantiles_when_positive']=np.quantile(thresholds,[.1,.5,.9]).tolist() if thresholds else None
        if rr[0]['metrics']['R3_main']['power'] is not None:
            for method in ['PB_grid','R3_scale','R3_median','R3_ordinary','B_strong']:
                summary['paired_power'][method]=ci([r['metrics']['R3_main']['power']-r['metrics'][method]['power'] for r in rr],-1,1)
        if case.get('outside_M0'):
            summary['paired_boundary_fdr']['unprotected']=ci([r['metrics']['R3_main']['fdp']-r['metrics']['PB_grid']['fdp'] for r in rr],-1,1)
            summary['paired_boundary_fdr']['external_Delta5']=ci([r['metrics']['R3_Delta5']['fdp']-r['metrics']['R2_Delta5']['fdp'] for r in rr],-1,1)
        summary['numerical_failures']=sum(r['status']!='completed' for r in rr)
        summary['r2_numerical_failures']=sum(r['old_times']['r2_status']!='completed' for r in rr)
        summary['v1_fallbacks']=sum(r['old_times']['v1_status']!='COMPUTED' for r in rr)
        summary['sensitivity_failures']={key:sum(r['sensitivity'][key]['status']!='completed' for r in rr) for key in rr[0]['sensitivity']}
        summary['runtime']={key:{'mean':float(np.mean(v)),'p90':float(np.quantile(v,.9))} for key,v in {
            'r3_with_ablations':[r['r3_seconds'] for r in rr if r['r3_seconds'] is not None],
            'r2_with_controls':[r['old_times']['r2'] for r in rr if r['old_times']['r2'] is not None],
            'v1_all_outputs':[r['old_times']['v1'] for r in rr],
            'whole_task':[r['wall_seconds'] for r in rr]}.items() if len(v)}
        ratios={name:[r[field]/r['kappa_true'] for r in rr if r[field] is not None] for name,field in [('geometric','kappa_geometric'),('median','kappa_median')]}
        summary['scale_ratio_quantiles']={name:np.quantile(v,[.1,.5,.9]).tolist() for name,v in ratios.items() if v}
        gamma=[g==0 for r in rr for f in r['folds'] for g in f['gamma']]
        summary['gamma_zero_fraction']=float(np.mean(gamma)) if gamma else None
        results.append(summary)
    by={(r['case'],r['rep']):r for r in rows}
    aggregate=ci([np.mean([by[(c,j)]['metrics']['R3_main']['power']-by[(c,j)]['metrics']['PB_grid']['power'] for c in p['core_cases']]) for j in range(p['repetitions'])],-1,1)
    if count!=p['planned_two_sided_intervals'] or count>p['interval_cap']:raise ValueError('declared interval family differs:'+str(count))
    model=[r for r in results if not r['design'].get('outside_M0')]
    boundary={str(r['case']):('WORSENING_DETECTED' if r['paired_boundary_fdr']['unprotected']['simultaneous_interval'][0]>0 else
                             'NONINCREASE_SUPPORTED_THIS_SCENE' if r['paired_boundary_fdr']['unprotected']['simultaneous_interval'][1]<=0 else
                             'UNRESOLVED') for r in results if r['paired_boundary_fdr']}
    gates={
      'model_fdr_uppers_at_most_05':all(r['methods']['R3_main']['fdp']['simultaneous_interval'][1]<=.05 for r in model),
      'core_improvement_lower_positive':aggregate['simultaneous_interval'][0]>0,
      'all_M0_nonnull_no_important_regression':all(r['paired_power']['PB_grid']['simultaneous_interval'][0]>=-p['noninferiority_margin'] for r in model if r['paired_power']),
      'additional_drift_worsening_detected':any(r['paired_boundary_fdr']['unprotected']['simultaneous_interval'][0]>0 for r in results if r['paired_boundary_fdr']),
      'full_user_success_excluded_by_drift':any(v=='WORSENING_DETECTED' for v in boundary.values()),
      'drift_nonincrease_demonstrated_in_both_scenarios':all(v=='NONINCREASE_SUPPORTED_THIS_SCENE' for v in boundary.values()),
      'full_user_success_not_automatically_decidable':'novelty, utility magnitude, complexity and boundary qualifications require main/reviewer evidence judgment'}
    write(out/'summary.json',{'rows':results,'core_paired_improvement':aggregate,'empirical_gates':gates,'boundary_status':boundary,
      'two_sided_intervals':count,'interval_cap':p['interval_cap'],'joint_confidence':1-p['alpha'],
      'families':len(rows),'freeze_sha256':sha(out/'freeze.json'),'index_sha256':sha(out/'index.json'),
      'protocol_sha256':sha(out/'protocol.json')})
    for r in results:print(r['design']['name'],{k:{m:None if v is None else v['mean'] for m,v in x.items() if m in ['power','fdp']} for k,x in r['methods'].items()},flush=True)
    print({'gates':gates,'core':aggregate},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();analyze(Path(a.out).resolve())
