"""Reviewer-requested existing-comparator scope check, no methods or draws."""
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import robustness_replay as rp
from robustness_convergence_audit import MAIN, stratified

ROOT=Path(__file__).resolve().parents[1]


def main():
    folder=ROOT/'artifacts/robustness';out=folder/'convergence-20260915-hybrid-scope.json'
    registry=rp.read_json(folder/'EXPERIMENT_REGISTRY.json')
    result={}
    for run in ['R0062','R0069']:
        path=folder/f'{run}-screen.json';entry=next(e for e in registry['experiments'] if e['id']==run)
        # Require the exact registered artifact, not an unverified summary edit.
        expected=entry.get('sha256') or entry.get('result_sha256')
        if expected is None:
            raise ValueError('Inspect registry hash schema before proceeding: '+str(entry.keys()))
        if rp.sha(path)!=expected:raise ValueError('Registered compact artifact hash mismatch')
        data=rp.read_json(path);protocol=rp.read_json(folder/f'{run}-screen.protocol.json')
        if data['settings']!=protocol:raise ValueError('Settings mismatch')
        names=[f'pilot{c}_pattern_support_simes_support_gate_eBH' for c in ['c0.25','c0.5','c0.65','c0.8','cv']]
        scenes=data['scenarios'];case_rows=[]
        for i,s in enumerate(scenes):
            if s['case']!=protocol['cases'][i]:raise ValueError('Case mismatch')
            rows={r['method']:r for r in s['rows']}
            for name in [MAIN,*names]:
                r=rows[name]
                for field,metric in [('power_by_repetition','power'),('fdp_by_repetition','fdr')]:
                    vals=np.asarray(r[field])
                    if len(vals)!=protocol['repetitions'] or not np.isfinite(vals).all() or np.any((vals<0)|(vals>1)) or not np.isclose(vals.mean(),r[metric]['mean'],atol=2e-14,rtol=0):
                        raise ValueError('Repetition/mean mismatch')
            case_rows.append({'case':s['case'],'methods':{n:{'power':rows[n]['power']['mean'],'fdr':rows[n]['fdr']['mean']} for n in [MAIN,*names]}})
        r={}
        for name in names:
            x=np.array([np.array(next(v for v in s['rows'] if v['method']==MAIN)['power_by_repetition'])-
                np.array(next(v for v in s['rows'] if v['method']==name)['power_by_repetition']) for s in scenes])
            r[name]={'comparison':stratified(x,alpha=.05),'power':float(np.mean([c['methods'][name]['power'] for c in case_rows])),
                'mean_fdp':float(np.mean([c['methods'][name]['fdr'] for c in case_rows])),
                'max_case_fdp':max(c['methods'][name]['fdr'] for c in case_rows)}
        result[run]={'registered_artifact_sha256':expected,'cases':case_rows,'hybrids':r}
    rp.write_new(out,{'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':rp.sha(__file__),
        'protocol_sha256':rp.sha(ROOT/'docs/ROBUSTNESS_CONVERGENCE_PROTOCOL_2026-09-15.md'),
        'new_simulations':0,'notice':'Separate exploratory comparator-scope sensitivity, not original15-baseline analysis or confirmation.',
        'runs':result})
    print('COMPLETE',out)


if __name__=='__main__':main()
