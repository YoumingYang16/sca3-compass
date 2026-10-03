"""Bounded, frozen paired R4 experiments, with per-family checkpoints."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,json,os,shutil,sys,time,traceback,platform
sys.dont_write_bytecode=True
import numpy as np
import scipy
from threadpoolctl import threadpool_limits
from r4_common import PHASE,CODE,ROOT,exp,fc,ci,read,write,sha,js,verify_runtime,ANCHORS
from r4_kernel import evaluate_kernel
from r4_simulation import generate

FILES=['r4_common.py','r4_model.py','r4_kernel.py','target_calibration.py','two_bank_reference.py',
       'r4_simulation.py','r4_experiment.py','test_r4.py','information_bounds.py','r4_cli.py']

def diagnostic_json(value):
    """Retain legitimate infinite-df diagnostics explicitly, never alter arrays/metrics."""
    if isinstance(value,np.ndarray): return diagnostic_json(value.tolist())
    if isinstance(value,np.generic): return diagnostic_json(value.item())
    if isinstance(value,float) and not np.isfinite(value):
        return {'nonfinite_float': 'nan' if np.isnan(value) else ('positive_infinity' if value>0 else 'negative_infinity')}
    if isinstance(value,dict): return {k:diagnostic_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [diagnostic_json(v) for v in value]
    return value

def freeze(out,protocol):
    if out.exists(): raise FileExistsError('never overwrite a frozen batch')
    out.mkdir();shutil.copy2(protocol,out/'protocol.json')
    for f in FILES: shutil.copy2(PHASE/f,out/f)
    shutil.copy2(PHASE/'R4_THEORY.md',out/'R4_THEORY.md')
    write(out/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'files':{f:sha(out/f) for f in FILES+['R4_THEORY.md','protocol.json']},
        'anchors':ANCHORS,'dependencies':verify_runtime(),
        'environment':{'python':sys.version,'executable':sys.executable,'numpy':np.__version__,
                       'scipy':scipy.__version__,'platform':platform.platform()}})

def validate(out):
    out=Path(out)
    for f,d in read(out/'freeze.json')['files'].items():
        if sha(out/f)!=d: raise ValueError('frozen file changed: '+f)
    for name in ['r4_common','r4_kernel','r4_simulation','target_calibration','two_bank_reference']:
        if Path(sys.modules[name].__file__).resolve()!=out/(name+'.py'):
            raise ValueError('wrong R4 imported source: '+name)
    if Path(__file__).resolve()!=out/'r4_experiment.py': raise ValueError('execute frozen script')
    if verify_runtime()!=read(out/'freeze.json')['dependencies']: raise ValueError('dependency mismatch')

def one(task):
    folder,case,rep=task;out=Path(folder);p=read(out/'protocol.json');c=p['cases'][case]
    dest=out/f'raw/case-{case:02}/rep-{rep:05}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError('duplicate experiment')
    row={'case':case,'rep':rep,'stage':p['stage'],'freeze_sha256':sha(out/'freeze.json'),
         'independent_unit':'entire_family','provenance':'SIMULATION_NOT_PATIENT_DATA'}
    start=time.perf_counter();cpu=time.process_time()
    try:
        with threadpool_limits(1):
            observed,diag,seed=generate(p,case,rep)
            z=observed['z'];cs=observed['calibration_source'];ct=observed['calibration_target']
            for tag,data in [('observed',observed),('diagnostic',diag)]:
                path=dest.with_name(dest.name.replace('.json.gz','-'+tag+'.npz'))
                with path.open('xb') as f: np.savez_compressed(f,**data)
                row[tag+'_path']=path.relative_to(out).as_posix();row[tag+'_sha256']=sha(path)
            decisions={};arrays={};receipts={};times={};statuses={}
            specs=[('source',0.,1.),('source_bound',0.,c['D']),('target',1.,1.),
                   ('bridge',len(ct)/(len(cs)+len(ct)),c['D']),
                   ('naive_pool',len(ct)/(len(cs)+len(ct)),1.)]
            for tag,w,D in specs:
                result=evaluate_kernel(z,cs,ct,seed=seed,reference_draws=p['reference_draws'],weight=w,bound=D)
                decisions[tag]=result['decision'];statuses[tag]=result['status'];times[tag]=result['seconds']
                for field in ['p','e','reference']: arrays[field+'_'+tag]=result[field]
                receipts[tag]={'calibration':result['calibration'],'folds':result['folds'],
                    'reference_seconds':result['reference_seconds'],'error':result.get('error')}
                if tag in ['target','bridge']:
                    decisions[tag+'_ordinary']=result['ordinary_decision']
                    arrays['e_'+tag+'_ordinary']=result['ordinary_e']
            # Strong comparator gets observed target bank; pooled counterpart
            # also receives Cs/Ct and the SAME external D, never true delta.
            for tag,bank in [('strong_target',ct),('strong_pool_bound',np.concatenate([
                    cs+(np.sqrt(c['D'])-1)*cs.mean(-1,keepdims=True),ct]))]:
                begin=time.perf_counter()
                v=exp.evaluate_v1(z,bank.transpose(1,0,2),seed=seed,acknowledge_scope=True)
                decisions[tag]=v['discoveries']['B_strong'];times[tag]=time.perf_counter()-begin
                arrays['e_'+tag]=v['e_values']['B_strong']
                statuses[tag]=v['status'];receipts[tag]={'diagnostics':diagnostic_json(v['diagnostics'])}
            if set(decisions)!=set(p['methods']): raise ValueError('method roster differs')
            arrays.update({'decision_'+k:v for k,v in decisions.items()})
            ep=dest.with_name(dest.name.replace('.json.gz','-evidence.npz'))
            with ep.open('xb') as f: np.savez_compressed(f,**arrays)
            row.update(status='completed',algorithm_seed=seed,method_status=statuses,times=times,
                receipts=receipts,metrics={k:exp.score(v,diag['truth']) for k,v in decisions.items()},
                evidence_path=ep.relative_to(out).as_posix(),evidence_sha256=sha(ep),
                true_delta_DIAGNOSTIC_ONLY=diag['kappa_target_DIAGNOSTIC_ONLY']/diag['kappa_source_DIAGNOSTIC_ONLY'])
    except Exception as err:
        row.update(status='hard_failure',error=repr(err),traceback=traceback.format_exc())
    row.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
               utc=datetime.now(timezone.utc).isoformat())
    payload=json.dumps(row,default=js,allow_nan=False)
    with gzip.open(dest,'xt',encoding='utf8') as f: f.write(payload)
    return {'case':case,'rep':rep,'path':dest.relative_to(out).as_posix(),'sha256':sha(dest),'status':row['status']}

def checkpoint(out,c,r):
    rel=f'raw/case-{c:02}/rep-{r:05}.json.gz';dest=out/rel
    if not dest.exists():
        if list(dest.parent.glob(f'rep-{r:05}-*.npz')): raise ValueError('orphan requires audit, not regeneration')
        return None
    row=json.load(gzip.open(dest,'rt',encoding='utf8'))
    if row['case']!=c or row['rep']!=r or row['freeze_sha256']!=sha(out/'freeze.json'):
        raise ValueError('checkpoint mismatch')
    for tag in ['observed','diagnostic','evidence']:
        if tag+'_path' in row and sha(out/row[tag+'_path'])!=row[tag+'_sha256']:
            raise ValueError('checkpoint artifact mismatch')
    return {'case':c,'rep':r,'path':rel,'sha256':sha(dest),'status':row['status']}

def run(out,resume=False):
    validate(out);p=read(out/'protocol.json')
    if (out/'index.json').exists(): raise FileExistsError('batch complete, no rerun')
    if (out/'started.json').exists() and not resume: raise ValueError('inspect owner and explicitly resume')
    if resume:
        import subprocess
        receipts=[out/'started.json',*out.glob('resumed-*.json')]
        for receipt in receipts:
            if not receipt.exists(): continue
            pid=int(read(receipt)['pid'])
            if os.name=='nt':
                check=subprocess.run(['powershell','-NoProfile','-Command',
                    f"try {{ if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{exit 0}} else {{exit 10}} }} catch {{exit 2}}"],capture_output=True)
                if check.returncode!=10: raise RuntimeError('prior PID alive or status unknown; inspect before resuming')
            else:
                try: os.kill(pid,0)
                except ProcessLookupError: pass
                else: raise RuntimeError('prior PID exists')
    rows=[];tasks=[]
    for c in range(len(p['cases'])):
        for r in range(p['repetitions']):
            old=checkpoint(out,c,r)
            if old is not None:
                if not resume: raise FileExistsError('old row on new run')
                rows.append(old)
            else: tasks.append((str(out),c,r))
    write(out/(f'resumed-{time.time_ns()}.json' if resume else 'started.json'),
        {'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),'retained':len(rows),'remaining':len(tasks)})
    start=time.perf_counter();total=len(rows)+len(tasks)
    with ProcessPoolExecutor(max_workers=p['workers'],initializer=validate,initargs=(out,)) as pool:
        for row in pool.map(one,tasks,timeout=p['safety_timeout_seconds']):
            rows.append(row)
            if len(rows)%16==0:
                progress={'completed':len(rows),'total':total,'failed':sum(x['status']!='completed' for x in rows),
                    'seconds':time.perf_counter()-start,'utc':datetime.now(timezone.utc).isoformat()}
                print(json.dumps(progress),flush=True);write(out/f'progress-{len(rows):06}.json',progress)
    rows.sort(key=lambda x:(x['case'],x['rep']))
    write(out/'index.json',{'complete':len(rows)==total,'rows':rows,'seconds':time.perf_counter()-start})

def collect(out):
    validate(out);idx=read(out/'index.json');p=read(out/'protocol.json');rows=[]
    expected={(c,r) for c in range(len(p['cases'])) for r in range(p['repetitions'])}
    if not idx['complete'] or {(x['case'],x['rep']) for x in idx['rows']}!=expected or len(idx['rows'])!=len(expected):
        raise ValueError('incomplete task identity')
    for item in idx['rows']:
        if sha(out/item['path'])!=item['sha256']: raise ValueError('record changed')
        row=json.load(gzip.open(out/item['path'],'rt',encoding='utf8'));rows.append(row)
        if row['status']!='completed': continue
        for tag in ['observed','diagnostic','evidence']:
            if sha(out/row[tag+'_path'])!=row[tag+'_sha256']: raise ValueError('artifact changed')
        with np.load(out/row['diagnostic_path'],allow_pickle=False) as d: truth=d['truth']
        with np.load(out/row['evidence_path'],allow_pickle=False) as a:
            for k in p['methods']:
                if exp.score(a['decision_'+k],truth)!=row['metrics'][k]: raise ValueError('score mismatch')
                if 'e_'+k in a and not np.array_equal(fc.ebh(a['e_'+k],.05),a['decision_'+k]):
                    raise ValueError('eBH mismatch')
    return p,rows

def analyze(out):
    p,rows=collect(out);fail=[r for r in rows if r['status']!='completed']
    if fail:
        write(out/'summary.json',{'status':'INCOMPLETE_NO_SUCCESS_ONLY_MEANS','failures':fail});return
    result=[]
    for c,scene in enumerate(p['cases']):
        rr=[r for r in rows if r['case']==c];methods={};contrasts={}
        for k in p['methods']:
            methods[k]={metric:None if rr[0]['metrics'][k][metric] is None else
                float(np.mean([r['metrics'][k][metric] for r in rr])) for metric in ['power','fdp','tp','fp','discoveries']}
            methods[k]['seconds']=float(np.mean([r['times'].get(k,r['times'].get(k.removesuffix('_ordinary'),0)) for r in rr]))
            methods[k]['FDR_interval']=ci.interval([r['metrics'][k]['fdp'] for r in rr],cap=p['interval_cap'],alpha=p['alpha'])
        if rr[0]['metrics']['target']['power'] is not None:
            for a,b in p['comparisons']:
                vals=[r['metrics'][a]['power']-r['metrics'][b]['power'] for r in rr]
                contrasts[a+'_minus_'+b]=ci.interval(vals,low=-1,high=1,cap=p['interval_cap'],alpha=p['alpha'])
        result.append({'case':c,'scene':scene,'n':len(rr),'methods':methods,'paired_power':contrasts,
                       'statuses':{k:sorted(set(r['method_status'].get(k.removesuffix('_ordinary')) for r in rr)) for k in p['methods']}})
    core={};lookup={(r['case'],r['rep']):r for r in rows}
    for a,b in p['comparisons']:
        vals=[np.mean([lookup[(c,rep)]['metrics'][a]['power']-
                       lookup[(c,rep)]['metrics'][b]['power']
                       for c in p['core_cases']]) for rep in range(p['repetitions'])]
        core[a+'_minus_'+b]=ci.interval(vals,low=-1,high=1,cap=p['interval_cap'],alpha=p['alpha'])
    allocation={}
    for ca,cb in p.get('allocation_comparisons',[]):
        for method in ['target','bridge','strong_target']:
            vals=[lookup[(ca,r)]['metrics'][method]['power']-lookup[(cb,r)]['metrics'][method]['power']
                  for r in range(p['repetitions'])]
            allocation[f'case{ca}_minus_case{cb}_{method}']=ci.interval(vals,low=-1,high=1,cap=p['interval_cap'],alpha=p['alpha'])
    intervals_count=sum(len(r['methods'])+len(r['paired_power']) for r in result)+len(core)+len(allocation)
    if intervals_count>p['interval_cap']: raise ValueError('unregistered excess comparison family')
    write(out/'summary.json',{'status':'COMPLETE','stage':p['stage'],'index_sha256':sha(out/'index.json'),
        'freeze_sha256':sha(out/'freeze.json'),'rows':result,'core_paired_power':core,'allocation_comparisons':allocation,
        'intervals_count':intervals_count,'interval_cap':p['interval_cap'],'alpha':p['alpha'],
        'interpretation':'finite fixed-design simultaneous intervals; whole families, not genes; DEV is not confirmation'})
    for r in result:
        print(r['case'],{k:round(v['power'] or 0,4) for k,v in r['methods'].items()},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','resume','analyze'])
    p.add_argument('--out',required=True);p.add_argument('--protocol');a=p.parse_args();out=Path(a.out).resolve()
    if a.action=='freeze': freeze(out,Path(a.protocol))
    elif a.action=='resume': run(out,True)
    else: globals()[a.action](out)
