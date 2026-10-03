"""Bounded read-only reanalysis of frozen repetitions; NO simulation or fitting."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import norm, t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import robustness_replay as replay
from sca3_compass.robustness_analysis import analyze_artifact, empirical_bernstein_interval
from sca3_compass.robustness_betting import betting_interval

MAIN = 'pilotc0.5_pattern_projection_support_gate_eBH'
BASES = [f'pilot{c}_{method}_eBH' for c in ['c0.25','c0.5','c0.65','c0.8','cv']
         for method in ['pattern_support_simes','ordinary_simes','ordinary_bonf']]
PACKAGES = ['joint_pattern','free_central_capped_joint_pattern','continuous_pattern','continuous_floor_pattern']


def bounded(x, support=(0,1)):
    x = np.asarray(x, float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or np.any((x<support[0])|(x>support[1])):
        raise ValueError('Invalid bounded repetition vector')
    return x


def contrast(x, support=(-1,1), alpha=.05):
    x = bounded(x, support)
    se = float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else None
    critical = t.ppf(1-alpha/2,len(x)-1) if len(x)>1 else None
    return {'n':len(x),'mean':float(x.mean()), 'mc_se':se,
            'approximate_t_interval': None if se is None else [float(x.mean()-critical*se),float(x.mean()+critical*se)],
            'approximate_notice':'pointwise Monte Carlo approximation; no development-selection correction',
            'bounded_betting':betting_interval(x, alpha=alpha, support=support)}


def stratified(x, alpha=.05/12):
    x = np.asarray(x,float)
    if x.ndim!=2 or min(x.shape)<2:
        raise ValueError('At least two equally replicated scenes required')
    bounded(x.ravel(),(-1,1))
    c,n=x.shape
    se=float(np.sqrt(np.sum(x.var(axis=1,ddof=1)/n))/c)
    return {'cases':c,'repetitions_per_case':n,'mean':float(x.mean()),'mc_se':se,
            'approximate_95_interval':[float(x.mean()-norm.ppf(.975)*se),float(x.mean()+norm.ppf(.975)*se)],
            'scenario_mean_sd':float(x.mean(axis=1).std(ddof=1)),
            'scenario_mean_min':float(x.mean(axis=1).min()),'scenario_mean_max':float(x.mean(axis=1).max()),
            'supplemental_bernstein':empirical_bernstein_interval(x.ravel(),lower=-1,upper=1,alpha=alpha),
            'assumptions':'Independent whole-family repetitions/scenes, equal scene counts; genes are not observations. Theorem11 allows different scene means. Development selection not covered.'}


def factorial(a,b,c,d):
    arrays=[bounded(x) for x in [a,b,c,d]]
    if len({len(x) for x in arrays})!=1:
        raise ValueError('No truncation or unpaired factorial arrays')
    a,b,c,d=arrays
    return {name:contrast(x,support) for name,x,support in [
        ('package_simple_B-A',b-a,(-1,1)),('projection_original_C-A',c-a,(-1,1)),
        ('projection_candidate_D-B',d-b,(-1,1)),('package_complex_D-C',d-c,(-1,1)),
        ('interaction_D-C-B+A',d-c-b+a,(-2,2))]}


def scale_records(diagnostics):
    records=[]
    for rep,di in enumerate(diagnostics):
        for package in ['original',*PACKAGES]:
            if package!='original' and package+'_combined' not in di:
                continue
            view=di if package=='original' else di[package+'_combined']
            for fold,f in enumerate(view['energy_prior']['folds']):
                scale=f['domain_scale']
                row={'repeat':rep,'fold':fold,'package':package,
                     'selected_multiplier':1. if scale is None else scale['selected_variance']/scale['base_variance'],
                     'scale_fit_converged':True if scale is None else scale['converged'],
                     'scale_fallback':False if scale is None else scale['fallback_to_transport_fcentral'],
                     'prior_df':f['target_only']['df'], 'prior_scatter':f['target_only']['scatter'],
                     'projection_variance':f['projection_variance']}
                if scale and 'continuous_fit' in scale:
                    fit=scale['continuous_fit'];info=fit['diagnostics']
                    row.update(null_weight=fit['weights'][0],floor_active=info.get('null_floor_active',False),
                        final_numerical_converged=fit['converged'],quadrature_passed=info['quadrature_converged'])
                pat=view.get('pattern_testing')
                if pat:
                    pf=pat['folds'][fold]
                    row.update(gamma=pf['mixing_fraction_by_sign'],profiles=pf['dominant_positive_profiles'],
                        pattern_converged=pf['fit']['diagnostics']['converged'],
                        pilot_projection=pf['pilot_calibration']['pilot80_pattern_projection_eBH'],
                        pilot_simes=pf['pilot_calibration']['pilot80_pattern_support_simes_eBH'])
                records.append(row)
    return records


def load_run(folder,run,output,mechanisms=False):
    protocol=replay.read_json(folder/f'{run}-screen.protocol.json')
    registry=replay.read_json(folder/'EXPERIMENT_REGISTRY.json')
    entry=next(e for e in registry['experiments'] if e['id']==run)
    if entry['status']!='completed' or entry['settings']!=protocol:
        raise ValueError('Require complete matching registry entry')
    audit,_=replay.audit_source(folder/f'{run}-source',protocol)
    scenes=[];receipts=[];records=[]
    for index,case in enumerate(protocol['cases']):
        path=folder/f'{run}-cases/case-{index:04}.json.gz'
        if not path.exists():path=path.with_suffix('')
        raw=replay.read_json(path);result=raw['result']
        if (raw['case_index']!=index or raw['settings_digest']!=replay.content_digest(protocol)
            or replay.content_digest(result)!=raw['result_digest'] or result['case']!=case):
            raise ValueError('Raw checkpoint content/protocol mismatch')
        for row in result['rows']:
            for metric,key in [('power','power_by_repetition'),('fdr','fdp_by_repetition')]:
                values=bounded(row[key])
                if len(values)!=protocol['repetitions'] or not np.isclose(values.mean(),row[metric]['mean'],atol=2e-14,rtol=0):
                    raise ValueError('Raw per-repetition mean/count mismatch')
        scenes.append({k:v for k,v in result.items() if k!='diagnostics'})
        if mechanisms:
            records.append({'case_index':index,'case':case,'fold_records':scale_records(result['diagnostics'])})
        receipts.append({'path':str(path.resolve()),'sha256':replay.sha(path),'result_digest':raw['result_digest']})
        print(run,index+1,'/',len(protocol['cases']),flush=True)
    replay.write_new(output/f'{run}-source-raw-audit.json',{'source':audit,'raw':receipts,'protocol':protocol})
    if records:replay.write_new(output/f'{run}-scale-mechanism-records.json',records)
    return {'settings':protocol,'scenarios':scenes,'elapsed_seconds':entry['elapsed_seconds']}


def rows(scene):return {r['method']:r for r in scene['rows']}


def main_audit(artifact,reference_mapping=None):
    cases=artifact['scenarios'];out={'per_case':[], 'macro':{}, 'fixed_baseline_contrasts':{}}
    mapping=[]
    for index,case in enumerate(cases):
        r=rows(case);best=max(BASES,key=lambda b:r[b]['power']['mean']);mapping.append(best)
        diff=bounded(r[MAIN]['power_by_repetition'])-bounded(r[best]['power_by_repetition'])
        vals=bounded(r[MAIN]['fdp_by_repetition'])
        out['per_case'].append({'index':index,'case':case['case'],'best_observed_base':best,
            'main_power':r[MAIN]['power']['mean'],'base_power':r[best]['power']['mean'],
            'paired_observed_best':contrast(diff),
            'mean_fdp':float(vals.mean()),'fdp_q95':float(np.quantile(vals,.95)),'fdp_q99':float(np.quantile(vals,.99)),
            'fdp_simultaneous':betting_interval(vals,alpha=.05/54),
            'baseline_fdps':{b:r[b]['fdr']['mean'] for b in BASES}})
    for group in ['all','normal','t5']:
        ids=[i for i,c in enumerate(cases) if group=='all' or c['case']['distribution']==group]
        out['macro'][group]={'power':float(np.mean([out['per_case'][i]['main_power'] for i in ids])),
            'gap_to_observed_case_max':float(np.mean([out['per_case'][i]['paired_observed_best']['mean'] for i in ids])),
            'mean_fdp':float(np.mean([out['per_case'][i]['mean_fdp'] for i in ids])),
            'max_case_fdp':max(out['per_case'][i]['mean_fdp'] for i in ids),
            'worst_gap':min(out['per_case'][i]['paired_observed_best']['mean'] for i in ids)}
        for baseline in ['pilotc0.5_pattern_support_simes_eBH','pilotc0.65_pattern_support_simes_eBH']:
            x=np.array([np.array(rows(cases[i])[MAIN]['power_by_repetition'])-
                        np.array(rows(cases[i])[baseline]['power_by_repetition']) for i in ids])
            out['fixed_baseline_contrasts'][group+'/'+baseline]=stratified(x)
        if reference_mapping:
            x=np.array([np.array(rows(cases[i])[MAIN]['power_by_repetition'])-
                        np.array(rows(cases[i])[reference_mapping[i]]['power_by_repetition']) for i in ids])
            out.setdefault('R62_mapping_on_R69',{})[group]=stratified(x,alpha=.05)
    out['observed_best_mapping']=mapping
    return out


def decomposition(artifact):
    result=[]
    for case in artifact['scenarios']:
        r=rows(case)
        for package in PACKAGES:
            if package+'_pilot80_pattern_projection_eBH' not in r:continue
            for mode in ['raw','gated']:
                comp='pattern_projection_eBH' if mode=='raw' else 'pattern_projection_gatedmix_eBH'
                names={'A':'pilot80_pattern_support_simes_eBH','B':package+'_pilot80_pattern_support_simes_eBH',
                       'C':'pilot80_'+comp,'D':package+'_pilot80_'+comp}
                cells={k:{'method':v,'power':r[v]['power']['mean'],'fdr':r[v]['fdr']['mean'],
                    'power_array':r[v]['power_by_repetition'],'fdp_array':r[v]['fdp_by_repetition']} for k,v in names.items()}
                effects=factorial(*(cells[k]['power_array'] for k in 'ABCD')) if case['power_defined'] else None
                result.append({'case':case['case'],'package':package,'projection_mode':mode,
                    'repetitions':case['repetitions'],'power_defined':case['power_defined'],'cells':cells,'effects':effects})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    folder=ROOT/'artifacts/robustness'
    replay.write_new(args.output/'audit-start.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':replay.sha(__file__),'protocol_sha256':replay.sha(ROOT/'docs/ROBUSTNESS_CONVERGENCE_PROTOCOL_2026-09-15.md'),
        'new_simulation_draws':0,'analysis_scope':'existing fixed runs only, no methods changed'})
    core={run:load_run(folder,run,args.output) for run in ['R0062','R0069']}
    old=replay.read_json(folder/'R0062-support-gate-checkpoint-analysis.json')
    rerun=analyze_artifact(core['R0062'],candidates=old['candidates'],baselines=old['baselines'],interval='betting')
    if rerun['macro']!=old['macro'] or rerun['uncertainty']!=old['uncertainty']:
        raise ValueError('Original interval result not exactly reproduced')
    replay.write_new(args.output/'original-interval-exact-reproduction.json',{'exact_macro_equal':True,
        'exact_uncertainty_equal':True,'uncertainty':rerun['uncertainty'],'main_macro':rerun['macro'][MAIN]})
    analysis62=main_audit(core['R0062']);analysis69=main_audit(core['R0069'],analysis62['observed_best_mapping'])
    replay.write_new(args.output/'mainline-comparison.json',{'R0062':analysis62,'R0069':analysis69})
    # A/B/C/D without new simulation: all saved cases retained, including nulls.
    for run in ['R0064','R0067']:
        artifact=load_run(folder,run,args.output,mechanisms=True)
        replay.write_new(args.output/f'{run}-contribution-decomposition.json',decomposition(artifact))
    replay.write_new(args.output/'completed.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'new_simulation_draws':0,'independent_confirmation':False,'status':'BOUNDED_REANALYSIS_COMPLETE'})
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
