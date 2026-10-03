"""Frozen fixed-size R2 confirmation; raw data + evidence + hashes retained."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import gzip
import json
import os
import shutil
import sys
import time
import traceback
sys.dont_write_bytecode=True
PHASE=Path(__file__).resolve().parent
from experiment import sha,read,write,js,generate,score,ROOT,evaluate_v1
from finite_calibration import evaluate as envelope
from predictive_bridge import evaluate
from provenance import bind_v1,verify_freeze,algorithm_seed,check_record
import numpy as np
from threadpoolctl import threadpool_limits
FILES=['confirm.py','analyze_confirm.py','provenance.py','experiment.py','predictive_bridge.py','grid_calibration.py','finite_calibration.py',
       'THEORY.md','PROBLEM_AND_ASSUMPTIONS.md','NUMERICAL_SCOPE.md',
       'test_predictive_bridge.py','test_grid_calibration.py','test_finite_calibration.py']


def freeze(out, protocol):
    if out.exists(): raise FileExistsError('preserve frozen directory')
    out.mkdir()
    shutil.copy2(protocol,out/'protocol.json')
    for f in FILES: shutil.copy2(PHASE/f,out/f)
    write(out/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),
          'files':{f:sha(out/f) for f in FILES+['protocol.json']},
          'v1_zip_sha256':sha(ROOT/'releases/K-NR-1.0.0.zip'),'v1_dependencies':bind_v1(),
          'stage':'one independent R2 confirmation, unchanged source required'})


def one(task):
    folder,case,rep=task; out=Path(folder); protocol=read(out/'protocol.json')
    record=out/f'raw/case-{case:02}/rep-{rep:05}.json.gz'
    if record.exists():
        with gzip.open(record,'rt',encoding='utf-8') as stream: row=json.load(stream)
        check_record(row,out,protocol,case,rep,metrics=True)
        return {'case':case,'rep':rep,'path':record.relative_to(out).as_posix(),'sha256':sha(record),'status':row['status']}
    record.parent.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); cpu=time.process_time()
    seed=algorithm_seed(protocol,case,rep)
    row={'case':case,'rep':rep,'algorithm_seed':seed,'freeze_sha256':sha(out/'freeze.json'),'provenance':'SIMULATED_WHOLE_FAMILY_NOT_PATIENT_DATA'}
    try:
        with threadpool_limits(1):
            z,cal,truth,r,kappa=generate(protocol,case,rep)
            ip=record.with_name(record.name.replace('.json.gz','-input.npz'))
            with ip.open('xb') as stream: np.savez_compressed(stream,z=z,calibration=cal,truth=truth,shape=r,kappa=kappa)
            row.update(input_path=ip.relative_to(out).as_posix(),input_sha256=sha(ip))
            begin=time.perf_counter()
            result=evaluate(z,cal,seed=seed,reference_draws=protocol['reference_draws'],iterations=protocol['shape_iterations'])
            pb_seconds=time.perf_counter()-begin
            begin=time.perf_counter()
            legacy=evaluate_v1(z,cal.transpose(1,0,2),seed=seed,acknowledge_scope=True)
            v1_seconds=time.perf_counter()-begin
            begin=time.perf_counter()
            fc=envelope(z,cal,seed=seed,reference_draws=199,delta_kappa=.005)
            fc_seconds=time.perf_counter()-begin
            decisions={**result['decisions'],**{k:legacy['discoveries'][k] for k in ['K_NR','B_strong','B_fair_conditional_eBH']},
                       **{k:fc['decisions'][k] for k in ['BB_eBH','BB_BY','R2_main']}}
            sensitivity={}; sensitivity_arrays={}
            if protocol['cases'][case].get('outside_M0'):
                for delta in [2.,5.]:
                    v=evaluate(z,cal,seed=seed,reference_draws=protocol['reference_draws'],iterations=protocol['shape_iterations'],mismatch_bound=delta)
                    key=f'PB_Delta{int(delta)}'; decisions[key]=v['decisions']['PB_grid']
                    sensitivity[key]={'status':v['status'],'seconds':v['seconds'],'error':v.get('error')}
                    sensitivity_arrays['p_'+key]=v['p']; sensitivity_arrays['e_'+key]=v['evidence']['PB_grid']
            ep=record.with_name(record.name.replace('.json.gz','-evidence.npz'))
            arrays={'decision_'+k:v for k,v in decisions.items()}
            arrays.update({'e_'+k:v for k,v in result['evidence'].items()})
            arrays.update(p=result['p'],reference=result['reference'])
            arrays.update(sensitivity_arrays)
            with ep.open('xb') as stream: np.savez_compressed(stream,**arrays)
            row.update(status=result['status'],algorithm_seed=seed,metrics={k:score(v,truth) for k,v in decisions.items()},
                       evidence_path=ep.relative_to(out).as_posix(),evidence_sha256=sha(ep),
                       folds=result['folds'],kappa_hat=result['kappa_estimate'],kappa_true_target=kappa,
                       kappa_upper=fc['kappa_upper'],kappa_true_cal=(1+5*protocol['cases'][case].get('calibration_rho',protocol['cases'][case]['rho']))/(6*(1-protocol['cases'][case].get('calibration_rho',protocol['cases'][case]['rho']))),
                       pb_seconds=pb_seconds,v1_seconds=v1_seconds,fc_seconds=fc_seconds,sensitivity=sensitivity,
                       numerical_error=result.get('error'),v1_status=legacy['status'])
    except Exception as err:
        row.update(status='hard_failure',error=repr(err),traceback=traceback.format_exc())
    row.update(wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,finished_utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(record,'xt',encoding='utf-8') as stream: json.dump(row,stream,default=js,allow_nan=False)
    return {'case':case,'rep':rep,'path':record.relative_to(out).as_posix(),'sha256':sha(record),'status':row['status']}


def run(out,resume=False):
    p=read(out/'protocol.json'); fr=verify_freeze(out)
    for f,digest in fr['files'].items():
        if sha(out/f)!=digest: raise ValueError('frozen file changed')
        if f!='protocol.json' and sha(PHASE/f)!=digest: raise ValueError('working executor differs')
    if sha(ROOT/'releases/K-NR-1.0.0.zip')!=fr['v1_zip_sha256']: raise ValueError('V1 changed')
    if (out/'index.json').exists(): raise FileExistsError('already finished')
    if (out/'started.json').exists() and not resume: raise FileExistsError('inspect PID, then explicit --resume only if stopped')
    if not resume: write(out/'started.json',{'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat()})
    tasks=[(str(out),c,r) for c in range(len(p['cases'])) for r in range(p['repetitions'])]
    rows=[]; start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=p['workers'],initializer=verify_freeze,initargs=(out,)) as pool:
        for row in pool.map(one,tasks,timeout=7200):
            rows.append(row)
            if len(rows)%64==0: print(json.dumps({'completed':len(rows),'total':len(tasks),'failures':sum(x['status']!='completed' for x in rows),'seconds':time.perf_counter()-start}),flush=True)
    write(out/'index.json',{'rows':rows,'complete':len(rows)==len(tasks),'seconds':time.perf_counter()-start})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['freeze','run']); parser.add_argument('--out',required=True)
    parser.add_argument('--protocol',default=str(PHASE/'protocol-confirm.json')); parser.add_argument('--resume',action='store_true')
    a=parser.parse_args(); out=Path(a.out).resolve()
    if a.action=='freeze': freeze(out,a.protocol)
    else: run(out,a.resume)
