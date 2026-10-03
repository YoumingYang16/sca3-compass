"""R0021: known-parameter Gaussian fission mechanism, never deployable evidence."""
import json,time
from pathlib import Path
from datetime import datetime,UTC
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from sca3_compass.molecular_data import PROJECT_ROOT,write_json,digest
from sca3_compass.molecular_methods import partial_conjunction,fdr_adjust,ebh
from sca3_compass.robustness_cone import cone_partial_conjunction
from sca3_compass.robustness_methods import contrasts
from sca3_compass.robustness_calibrators import by_calibrator,focused_calibrator
from sca3_compass.robustness_registry import update_registry
from robustness_screen import data


def main():
    run='R0021'
    out=PROJECT_ROOT/f'artifacts/robustness/{run}-fission-diagnostic.json'
    reg=PROJECT_ROOT/'artifacts/robustness/EXPERIMENT_REGISTRY.json'
    if out.exists() or any(r['id']==run for r in json.loads(reg.read_text())['experiments']):
        raise SystemExit('Preserve used run ID')
    started=time.perf_counter()
    entry={'id':run,'status':'running','started_at':datetime.now(UTC).isoformat(),
           'phase':'DEVELOPMENT_ORACLE_MECHANISM_ONLY','seed':2101223,'repetitions':200,
           'source_sha256':digest(Path(__file__)),'independent_confirmation':False}
    update_registry(reg,entry,create=True)
    # Copy every source used, not just the driver, before generating results.
    import shutil
    src=[Path(__file__),PROJECT_ROOT/'scripts/robustness_screen.py']
    src+=list((PROJECT_ROOT/'src/sca3_compass').glob('robustness_*.py'))
    entry['source_sha256']={str(p.relative_to(PROJECT_ROOT)):digest(p) for p in src}
    for p in src:
        dest=out.parent/f'{run}-source'/p.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,dest)
    write_json(out.with_suffix('.protocol.json'),entry)
    results=[]
    try:
        for rho in [.1,.8,.95]:
            case={'name':f'normal_rho{rho}','distribution':'normal','rho':rho,'n':64,'effect':3.5}
            rng=np.random.default_rng(np.random.SeedSequence([2101223,int(rho*100)]))
            records={}
            for rep in range(200):
                z,_,truth=data(rng,case)
                g,s,k=z.shape
                v=(1+5*rho)/6
                shape=.35*np.eye(4)+.65
                means=np.stack((z.mean(-1),-z.mean(-1)),1)
                ordinary=cone_partial_conjunction(means,np.full(g,v),[shape,shape],np.inf)
                decisions={'oracle_cone_BY':fdr_adjust(ordinary)<=.05,
                    'oracle_cone_focused':ebh(focused_calibrator(ordinary,2*g,.001,.8),.05)}
                for tau in [1.,2.,4.]:
                    filter_e=[]
                    for col in range(5):
                        l=(z@contrasts(k)[...,col])*np.sqrt(v/(1-rho))
                        signl=np.stack((l,-l),1)
                        a=means+tau*signl
                        b=means-signl/tau
                        pa=cone_partial_conjunction(a,np.full(g,v*(1+tau*tau)),[shape,shape],np.inf)
                        pb=cone_partial_conjunction(b,np.full(g,v*(1+1/(tau*tau))),[shape,shape],np.inf)
                        e=focused_calibrator(pb,2*g,.001,.8)
                        for threshold in [.1,.3,.5]:
                            key=f'tau{tau}_screen{threshold}'
                            selected=pa<=threshold
                            # Whole-array A/B independence, known Gaussian
                            # shapes and common pipeline means are essential.
                            weighted=selected/threshold*e
                            if col==0:
                                decisions[key+'_single_eBH']=ebh(weighted,.05)
                                chosen=np.zeros_like(selected)
                                if selected.any():
                                    chosen[selected]=fdr_adjust(pb[selected])<=.05
                                decisions[key+'_selected_BY']=chosen
                            filter_e.append((key,weighted))
                    for key in {k for k,_ in filter_e}:
                        merged=np.mean([e for k,e in filter_e if k==key],axis=0)
                        decisions[key+'_five_eBH']=ebh(merged,.05)
                for name,reject in decisions.items():
                    d=records.setdefault(name,{'fdp':[],'power':[]})
                    d['fdp'].append(float((reject&~truth).sum()/max(1,reject.sum())))
                    d['power'].append(float((reject&truth).sum()/truth.sum()))
            results.append({'case':case,'rows':[{'method':k,'fdp':d['fdp'],'power':d['power'],
                'mean_fdp':float(np.mean(d['fdp'])),'mean_power':float(np.mean(d['power']))} for k,d in records.items()]})
            print(case['name'],[(r['method'],round(r['mean_power'],4)) for r in sorted(results[-1]['rows'],key=lambda r:-r['mean_power'])[:8]],flush=True)
        entry.update(status='completed',elapsed_seconds=time.perf_counter()-started)
        write_json(out,{'settings':entry,'scenarios':results})
        entry['result_sha256']=digest(out)
    except BaseException as exc:
        entry.update(status='failed',failure=str(exc),elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        update_registry(reg,entry)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
