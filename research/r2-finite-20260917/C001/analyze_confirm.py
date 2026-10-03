"""Pre-frozen bounded-mean intervals and paired summaries; whole families."""
from pathlib import Path
import argparse
import gzip
import json
import math
import numpy as np
from scipy.special import xlogy
from scipy.optimize import brentq
from experiment import read,write,sha
from provenance import verify_freeze,check_record


def kl_interval(values, low=0., high=1., comparisons=512, alpha=.05):
    x=np.asarray(values,float)
    if not np.isfinite(x).all() or np.any(x<low-1e-12) or np.any(x>high+1e-12): raise ValueError('metric outside declared bound')
    n=len(x); mean=float(x.mean()); a=float(np.clip((mean-low)/(high-low),0,1)); target=math.log(2*comparisons/alpha)/n
    def equation(b):
        return float(xlogy(a,a/b)+xlogy(1-a,(1-a)/(1-b))-target)
    lower=0. if a==0 else (math.exp(-target) if a==1 else brentq(equation,np.nextafter(0.,1.),a,xtol=1e-14))
    upper=1. if a==1 else (1-math.exp(-target) if a==0 else brentq(equation,a,np.nextafter(1.,0.),xtol=1e-14))
    return {'mean':mean,'simultaneous_95_kl':[low+(high-low)*lower,low+(high-low)*upper],
            'mcse':float(np.std(x,ddof=1)/np.sqrt(n)), 'n':n,
            'p90':float(np.quantile(x,.9)), 'p99':float(np.quantile(x,.99))}


def analyze(out):
    verify_freeze(out)
    protocol=read(out/'protocol.json'); index=read(out/'index.json')
    expected=len(protocol['cases'])*protocol['repetitions']
    if not index['complete'] or len(index['rows'])!=expected: raise ValueError('not complete')
    plan={(c,r) for c in range(len(protocol['cases'])) for r in range(protocol['repetitions'])}
    if {(r['case'],r['rep']) for r in index['rows']}!=plan: raise ValueError('planned Cartesian identity differs')
    cap=protocol['comparison_interval_family_cap']; alpha=protocol['alpha']
    interval=lambda values,low=0.,high=1.:kl_interval(values,low,high,cap,alpha)
    rows=[]; endpoints=0; failures=[]
    for entry in index['rows']:
        path=out/entry['path']
        if entry['path']!=f"raw/case-{entry['case']:02}/rep-{entry['rep']:05}.json.gz": raise ValueError('noncanonical record path')
        if sha(path)!=entry['sha256']: raise ValueError('record hash changed')
        with gzip.open(path,'rt',encoding='utf-8') as stream: row=json.load(stream)
        check_record(row,out,protocol,entry['case'],entry['rep'],metrics=True)
        if row['status']!=entry['status']: raise ValueError('record/index status differs')
        for part in ['input','evidence']:
            if part+'_path' in row and sha(out/row[part+'_path'])!=row[part+'_sha256']: raise ValueError('artifact hash changed')
        if row['status']=='hard_failure': failures.append(row)
        rows.append(row)
    if failures:
        write(out/'analysis-blocked.json',{'reason':'hard failures retained, no success-only summaries','failures':failures})
        raise ValueError('hard failure invalidates completed analysis')
    summaries=[]
    for i,case in enumerate(protocol['cases']):
        sub=[r for r in rows if r['case']==i]
        if len(sub)!=protocol['repetitions'] or len(set(r['rep'] for r in sub))!=len(sub): raise ValueError('incorrect repetitions')
        methods=protocol['methods']+(['PB_Delta2','PB_Delta5'] if case.get('outside_M0') else [])
        stats={}
        for method in methods:
            stats[method]={}
            for metric in ['power','fdp']:
                val=[r['metrics'][method][metric] for r in sub]
                stats[method][metric]=None if val[0] is None else interval(val)
                endpoints+=int(val[0] is not None)
        paired={}
        if sub[0]['metrics']['PB_grid']['power'] is not None:
            for comparator in ['PB_main','PB_grid_ordinary','K_NR','B_strong','B_fair_conditional_eBH','R2_main']:
                dif=[r['metrics']['PB_grid']['power']-r['metrics'][comparator]['power'] for r in sub]
                paired[comparator]=interval(dif,0. if comparator=='PB_main' else -1.,1.)
                endpoints+=1
        ratios=[r['kappa_upper']/r['kappa_true_cal'] for r in sub]
        coverage=interval([float(r['kappa_upper']>=r['kappa_true_cal']) for r in sub]); endpoints+=1
        summaries.append({'case_index':i,'case':case,'repetitions':len(sub),'methods':stats,'paired_power_difference':paired,
                          'coverage_of_calibration_domain_only':coverage,'kappa_upper_ratio_quantiles':np.quantile(ratios,[.1,.5,.9]).tolist(),
                          'numerical_failures':sum(r['status']!='completed' for r in sub),
                          'sensitivity_failures':{key:sum(r['sensitivity'][key]['status']!='completed' for r in sub) for key in sub[0]['sensitivity']},
                          'v1_declared_fallback_families':sum(r['v1_status']!='COMPUTED' for r in sub),
                          'runtime':{key:{'mean':float(np.mean([r[key] for r in sub])), 'p90':float(np.quantile([r[key] for r in sub],.9))} for key in ['pb_seconds','v1_seconds','fc_seconds','wall_seconds']},
                          'mean_pilot_gamma':np.mean([f['gamma_from_pilot'] for r in sub for f in r['folds']],axis=0).tolist()})
    if endpoints>cap or endpoints!=protocol['planned_two_sided_intervals']: raise ValueError('predeclared interval roster differs')
    write(out/'summary.json',{'rows':summaries,'two_sided_intervals':endpoints,'interval_family_cap':cap,
                             'alpha':alpha,'freeze_sha256':sha(out/'freeze.json'),'protocol_sha256':sha(out/'protocol.json'),
                             'index_sha256':sha(out/'index.json'),'total_families':expected,
                             'status':'completed_fixed_protocol_no_success_gate','wall_seconds':index['seconds']})
    for row in summaries:
        print(row['case']['name'],json.dumps({k:{m:None if v is None else v['mean'] for m,v in metrics.items()} for k,metrics in row['methods'].items()}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args(); analyze(Path(a.out).resolve())
