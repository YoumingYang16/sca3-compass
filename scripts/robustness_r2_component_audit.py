"""Complete-only descriptive fallback attribution, never a new experiment.

Whole-family paired I is partitioned, not selected or recalibrated. Conditional
associations are not causal treatment effects. No extra confidence intervals.
"""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
import sys
import shutil
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sca3_compass.robustness_io import write_json
from robustness_r2_development import read,sha


def partition(differences,groups):
    """Each scene is equally weighted, each replicate equally within scene."""
    if not differences or len(differences)!=len(groups):raise ValueError('Matching complete scenes required')
    result={}
    for label in ['both_success','R1_failure_only','K_failure_only','both_failure']:
        mass=total=0.;count=0
        for values,codes in zip(differences,groups):
            x=np.asarray(values,float);c=np.asarray(codes)
            if x.ndim!=1 or not len(x) or x.shape!=c.shape or not np.isfinite(x).all():
                raise ValueError('Finite complete whole-family vectors required')
            if not set(c)<=set(['both_success','R1_failure_only','K_failure_only','both_failure']):
                raise ValueError('Unknown failure group')
            selected=c==label
            weight=1/(len(differences)*len(x))
            count+=int(selected.sum());mass+=selected.sum()*weight;total+=float(x[selected].sum())*weight
        result[label]={'families':count,'equal_scene_weighted_fraction':float(mass),
                       'contribution_to_total_I':float(total),
                       'conditional_mean_I':float(total/mass) if mass else None}
    expected=float(np.mean([np.mean(x) for x in differences]))
    if not np.isclose(sum(v['contribution_to_total_I'] for v in result.values()),expected,atol=1e-13,rtol=0):
        raise ValueError('Partition must conserve original paired mean')
    if not np.isclose(sum(v['equal_scene_weighted_fraction'] for v in result.values()),1,atol=1e-13,rtol=0):
        raise ValueError('Partition must retain every family')
    return {'total_I':expected,'groups':result,'contribution_sum_exact_within_1e_minus13':True,
            'kind':'DESCRIPTIVE_DECOMPOSITION_NOT_CAUSAL_OR_NEW_CONFIDENCE_INTERVAL'}


def main():
    base=ROOT/'artifacts/robustness';path=base/'R0076-confirmation-analysis.json'
    result=read(path);replay=read(base/'R0076-replay.json')
    if not result['complete'] or not replay['complete'] or replay['analysis_sha256']!=sha(path):
        raise ValueError('Completed matching C2 and fresh replay required')
    out=base/'R0076-component-audit';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'generate_source.py')
    p=result['analysis_plan'];names=p['reported_methods'];a=names.index(p['candidate']);b=names.index(p['reference'])
    failed_r1={(r['case'],r['rep']) for r in result['R1_failed_folds']}
    failed_k={(r['case'],r['rep']) for r in result['K_failed_folds']}
    differences={};groups={};checked=[]
    for row in result['rows']:
        path=Path(row['analysis_array_path'])
        if sha(path)!=row['analysis_array_sha256']:raise ValueError('Audited array changed')
        with np.load(path,allow_pickle=False) as z:
            if list(z['methods'])!=names or len(z['power'])!=row['n']:raise ValueError('Unexpected method membership/count')
            if row['I'] is None:continue # Power undefined, not dropped from any defined I
            i=row['case_index'];differences[i]=z['power'][:,a]-z['power'][:,b]
        group=[]
        for rep in range(row['n']):
            u,v=(i,rep) in failed_r1,(i,rep) in failed_k
            group.append('both_failure' if u and v else 'R1_failure_only' if u else 'K_failure_only' if v else 'both_success')
        groups[i]=group;checked.append({'case':i,'array_sha256':row['analysis_array_sha256']})
    output={}
    for label,indices in p['strata'].items():
        output[label]=partition([differences[i] for i in indices],[groups[i] for i in indices])
        if not np.isclose(output[label]['total_I'],result['I'][label]['mean_difference'],atol=1e-13,rtol=0):
            raise ValueError('Frozen primary estimate mismatch')
    per_case={str(i):partition([differences[i]],[groups[i]]) for i in sorted(differences)}
    write_json(out/'summary.json',{'run_id':'R0076','complete':True,'kind':'SECONDARY_DESCRIPTIVE_ONLY',
        'analysis_sha256':sha(base/'R0076-confirmation-analysis.json'),'replay_sha256':sha(base/'R0076-replay.json'),
        'script_sha256':sha(__file__),'strata':output,'per_case':per_case,'array_receipts':checked,
        'R1_failed_families':len(failed_r1),'K_failed_families':len(failed_k),
        'finished_utc':datetime.now(timezone.utc).isoformat(),'provenance':'SIMULATION_NOT_PATIENT_DATA',
        'limits':['No new samples/test/CI','All families retained in original estimates',
                  'Partition is observational association, not causal effect of removing fallback',
                  'Original guarded p=1 fallback rules unchanged','No inference from genes or folds as independent replicates']})
    print(out,flush=True)


if __name__=='__main__':main()
