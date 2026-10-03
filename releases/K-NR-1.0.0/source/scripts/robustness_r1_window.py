"""Source-frozen, per-repetition resumable R1 runner; legacy runs untouched."""
from __future__ import annotations
from research_window import wait_start_gate,cooperative_stop
wait_start_gate()
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_io import write_json,write_json_gzip,content_digest
from sca3_compass.robustness_registry import update_registry
from sca3_compass.robustness_dispatch import bounded_results
from sca3_compass.robustness_pattern_test import pattern_test_candidates,_json_value
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_pivotal import evaluate as evaluate_r1
from sca3_compass.robustness_bootstrap_guard import evaluate as evaluate_guard
from sca3_compass.robustness_confirmation_protocol import validate_plan
from sca3_compass.molecular_methods import ebh
from robustness_screen import data

R0_MAIN='pilotc0.5_pattern_projection_support_gate_eBH'
H_BASES=[f'pilot{c}_{mode}_eBH' for c in ['c0.25','c0.5','c0.65','c0.8','cv']
         for mode in ['pattern_support_simes','ordinary_simes','ordinary_bonf']]


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    import gzip
    with (gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as s:return json.load(s)


def _r0(z,cal):
    p,diag=evaluate_candidates(z,cal,'frontier')
    p.update(energy_prior_candidates(z,cal,diag))
    p.update(pattern_test_candidates(z,diag,p['target_only_weighted_cone_PC'],pilot=True,pilot_selection=True))
    return {n:p[n] for n in [R0_MAIN,*H_BASES]},diag


def generate(seed,index,rep,case):
    rng=np.random.default_rng(np.random.SeedSequence([seed,index,rep]))
    working={**case}
    if working.get('mixed_sign_only'):working['truth']='global_null'
    z,cal,truth=data(rng,working,g=case.get('genes',256),k=case.get('pipelines',6))
    if case.get('mixed_sign_only'):
        z[:,0,:]+=case['effect'];z[:,1,:]-=case['effect']
        truth[:]=False # exactly one positive and one negative study, no signed replication
    return z,cal,truth


def one(root,protocol,case_index,rep):
    if cooperative_stop():return {'status':'NOT_STARTED','case_index':case_index,'rep':rep}
    out=Path(root)/'artifacts/robustness'/f"{protocol['run_id']}-repetitions"/f'case-{case_index:04}'
    out.mkdir(parents=True,exist_ok=True);path=out/f'rep-{rep:06}.json.gz'
    digest=content_digest(protocol)
    if path.exists():
        existing=read(path)
        if existing['protocol_digest']!=digest or existing['rep']!=rep or existing['case_index']!=case_index:
            raise ValueError('Existing checkpoint mismatch; never overwrite')
        if existing['status']!='completed':return {'status':'failed_existing','path':str(path)}
        return {'status':'completed','path':str(path),'sha256':sha(path),'reused':True,'case_index':case_index,'rep':rep}
    start=time.perf_counter();cpu=time.process_time()
    with threadpool_limits(1):
        z,cal,truth=generate(protocol['seed'],case_index,rep,protocol['cases'][case_index])
        npz=out/f'rep-{rep:06}-input.npz'
        if npz.exists():
            with np.load(npz) as old:
                if not all(np.array_equal(old[k],v) for k,v in [('z',z),('calibration',cal),('truth',truth)]):raise ValueError('Existing input conflict')
        else:
            with npz.open('xb') as stream:np.savez_compressed(stream,z=z,calibration=cal,truth=truth)
        record={'run_id':protocol['run_id'],'method_version':protocol['method_version'],'phase':protocol['phase'],
            'case_index':case_index,'rep':rep,'seed_sequence':[protocol['seed'],case_index,rep],
            'protocol_digest':digest,'input_sha256':sha(npz),'input_path':str(npz),
            'provenance':'SIMULATION_NOT_PATIENT_DATA','n_true_signed':int(truth.sum())}
        try:
            r0,d0=_r0(z,cal)
            r1,d1=(evaluate_guard(z,cal,draws=protocol['bootstrap_draws'],seed=int(np.random.SeedSequence([protocol['seed'],case_index,rep,913]).generate_state(1)[0]))
                if protocol.get('backend')=='bootstrap_guard' else evaluate_r1(z,cal))
            all_e={**r0,**r1};metrics={};claims={}
            for name,e in all_e.items():
                if e.shape!=truth.shape or not np.isfinite(e).all() or np.any(e<0):raise FloatingPointError('Invalid evidence:'+name)
                rejection=ebh(e,.05);total=int(rejection.sum());false=int((rejection&~truth).sum());true=int((rejection&truth).sum())
                metrics[name]={'fdp':false/max(1,total),'power':true/max(1,int(truth.sum())),
                    'tp':true,'fp':false,'discoveries':total,'power_defined':bool(truth.sum())}
                claims[name]=e
            evidence=out/f'rep-{rep:06}-evidence.npz'
            if evidence.exists():raise ValueError('Evidence exists without completed checkpoint; preserve for manual recovery')
            with evidence.open('xb') as stream:np.savez_compressed(stream,**claims,**{'p_'+k:v for k,v in d1['held_pvalues'].items()})
            null_input={}
            for mode,values in d1['held_pvalues'].items():
                x=values[~truth]
                null_input[mode]={'null_claims':len(x),'below_threshold':{str(q):int((x<=q).sum()) for q in [.0001,.001,.01,.05]},
                    'minimum':float(x.min()) if len(x) else None}
            d1={k:v for k,v in d1.items() if k!='held_pvalues'}
            record.update(status='completed',metrics=metrics,null_input=null_input,diagnostics_R1=_json_value(d1),
                diagnostics_R0=_json_value({k:d0[k] for k in ['fit','shape','energy_prior','pattern_testing'] if k in d0}),
                evidence_path=str(evidence),evidence_sha256=sha(evidence))
        except Exception as error:
            record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        record.update(elapsed_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
            finished_utc=datetime.now(timezone.utc).isoformat())
        write_json_gzip(path,record)
        return {'status':record['status'],'path':str(path),'sha256':sha(path),'case_index':case_index,'rep':rep}


def batch(root,protocol,index,reps):
    result=[]
    for rep in reps:
        if cooperative_stop():break
        result.append(one(root,protocol,index,rep))
    return result


def freeze(root,run_id,config,phase,repetitions,seed,workers,method_version,backend='pivotal',bootstrap_draws=16,analysis_plan=None):
    folder=root/'artifacts/robustness';protocol_path=folder/f'{run_id}-screen.protocol.json'
    if protocol_path.exists():raise ValueError('Run protocol already exists; use explicit --resume, never overwrite')
    source=folder/f'{run_id}-source';source.mkdir(exist_ok=False)
    sources=list((root/'src/sca3_compass').glob('*.py'))+[root/'scripts'/n for n in ['robustness_r1_window.py','robustness_screen.py','research_window.py','robustness_r1_analyze.py','robustness_r1_confirm.py','robustness_r1_replay.py']]
    hashes={}
    for src in sources:
        rel=src.relative_to(root);dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest);hashes[rel.as_posix()]=sha(dest)
    cases=read(config)['cases']
    protocol={'run_id':run_id,'phase':phase,'method_version':method_version,'seed':seed,'repetitions':repetitions,
        'cases':cases,'source_sha256':hashes,'source_snapshot':str(source),'created_utc':datetime.now(timezone.utc).isoformat(),
        'seed_rule':'SeedSequence([seed,case_index,rep]); same input for all methods',
        'H_baselines':H_BASES,'R0_main':R0_MAIN,'R1_main':'R1_pilotc0.5_projection_gate_eBH',
        'F_safe_kernels':['ordinary_bonf','support_bonf','weighted_bonf','support_simes_by'],
        'F_empirical_kernels':['ordinary_simes_empirical','support_simes_empirical'],
        'F_multipliers':[.25,.5,.65,.8],
        'alpha':.05,'workers':workers,'batch_size':4,'independent_confirmation':phase in ['C1','C2'],
        'window_file':str(root/'artifacts/robustness/R1R2-20260916/window.json'),
        'protocol_kind':'r1_window_per_repetition_v1','provenance':'SIMULATION_NOT_PATIENT_DATA',
        'backend':backend,'bootstrap_draws':bootstrap_draws}
    import platform,scipy
    from importlib.metadata import version
    protocol['environment']={'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
        'scipy':scipy.__version__,'threadpoolctl':version('threadpoolctl'),
        'blas_thread_limit_per_worker':1,'algorithm':'CPU; no GPU-specific kernels or paid services'}
    if backend=='bootstrap_guard':protocol['R1_main']='R1B_guard_pilotc0.5_projection_gate_eBH'
    protocol['repetition_counts']=[repetitions]*len(cases)
    if analysis_plan is not None:
        validate_plan(analysis_plan,cases)
        if analysis_plan['families']['H']!=H_BASES:raise ValueError('Historical15 baseline envelope changed')
        if phase not in ['C1','C2'] or analysis_plan['phase']!=phase:raise ValueError('Confirmation phase mismatch')
        if analysis_plan['candidate']!=protocol['R1_main'] or analysis_plan['bootstrap_draws']!=bootstrap_draws:raise ValueError('Candidate freeze mismatch')
        if analysis_plan['case_definition_digest']!=content_digest(cases):raise ValueError('Confirmation case definitions changed')
        if analysis_plan['candidate_source_sha256']!=sha(root/'src/sca3_compass/robustness_bootstrap_guard.py'):raise ValueError('Candidate changed after analysis-plan freeze')
        if analysis_plan['reporting_error_budget']!=.025 or sum(analysis_plan['error_allocation'].values())>.025+1e-14:raise ValueError('Confirmation reporting budget mismatch')
        if len(analysis_plan['repetition_counts'])!=len(cases) or min(analysis_plan['repetition_counts'])<2:raise ValueError('Fixed repetition counts required')
        protocol['repetition_counts']=analysis_plan['repetition_counts'];protocol['analysis_plan']=analysis_plan
        # No third attempt; completed/failed/interrupted confirmations still consume the batch.
        old=read(folder/'EXPERIMENT_REGISTRY.json')['experiments']
        if any(e.get('settings',{}).get('phase')==phase for e in old):raise ValueError('Confirmation batch already registered; no hidden repeat')
    write_json(protocol_path,protocol)
    entry={'id':run_id,'status':'prepared','method_version':method_version,'purpose':phase,'settings':protocol,
        'started_utc':protocol['created_utc'],'result':None,'elapsed_seconds':0}
    update_registry(folder/'EXPERIMENT_REGISTRY.json',entry,create=True)
    return protocol


def execute(root,run_id):
    folder=root/'artifacts/robustness';protocol=read(folder/f'{run_id}-screen.protocol.json')
    for rel,digest in protocol['source_sha256'].items():
        if sha(SOURCE/rel)!=digest:raise ValueError('Executing source snapshot mismatch:'+rel)
    registry=folder/'EXPERIMENT_REGISTRY.json';entry=next(r for r in read(registry)['experiments'] if r['id']==run_id)
    if entry['status']=='completed':raise ValueError('Run already completed; no duplicate launch')
    started=time.perf_counter();entry.update(status='running',pid=os.getpid(),cwd=str(SOURCE),execution_started_utc=datetime.now(timezone.utc).isoformat());update_registry(registry,entry)
    jobs=[]
    for i in range(len(protocol['cases'])):
        count=protocol.get('repetition_counts',[protocol['repetitions']]*len(protocol['cases']))[i]
        for start in range(0,count,protocol['batch_size']):
            reps=list(range(start,min(start+protocol['batch_size'],count)))
            jobs.append(((i,start),batch,(str(root),protocol,i,reps),{}))
    receipts=[]
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers']) as pool:
            for key,items in bounded_results(pool,jobs,max_pending=protocol['workers']):
                receipts.extend(items)
                write_json(folder/f'{run_id}-progress.json',{'run_id':run_id,'finished_utc':datetime.now(timezone.utc).isoformat(),
                    'completed':sum(x['status']=='completed' for x in receipts),'failed':sum(x['status']!='completed' for x in receipts),
                    'planned':sum(protocol.get('repetition_counts',[protocol['repetitions']]*len(protocol['cases']))),'last_batch':list(key)})
                print(run_id,key,len(receipts),'/',sum(protocol.get('repetition_counts',[protocol['repetitions']]*len(protocol['cases']))),flush=True)
                if cooperative_stop():break
        expected=sum(protocol.get('repetition_counts',[protocol['repetitions']]*len(protocol['cases'])))
        complete=len(receipts)==expected and all(r['status']=='completed' for r in receipts)
        index=folder/f'{run_id}-results-index.json'
        write_json(index,{'run_id':run_id,'settings':protocol,'complete':complete,'receipts':receipts,
            'elapsed_seconds':time.perf_counter()-started})
        entry.update(status='completed' if complete else ('interrupted' if cooperative_stop() else 'failed'),
            result=str(index),sha256=sha(index),elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        entry.update(status='failed',error=repr(error),elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat());raise
    finally:update_registry(registry,entry)
    if not complete:raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--config',type=Path)
    parser.add_argument('--phase',default='development');parser.add_argument('--method-version',default='R1-dev0')
    parser.add_argument('--repetitions',type=int,default=16);parser.add_argument('--seed',type=int,default=7005073)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--execute',action='store_true');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--backend',choices=['pivotal','bootstrap_guard'],default='pivotal');parser.add_argument('--bootstrap-draws',type=int,default=16)
    parser.add_argument('--analysis-plan',type=Path)
    parser.add_argument('--project-root',type=Path,default=SOURCE);args=parser.parse_args();root=args.project_root.resolve()
    if args.execute:return execute(root,args.run_id)
    if not args.resume:
        if args.phase!='development' and args.analysis_plan is None:raise ValueError('Confirmation requires a separately frozen analysis plan')
        freeze(root,args.run_id,args.config,args.phase,args.repetitions,args.seed,args.workers,args.method_version,args.backend,args.bootstrap_draws,read(args.analysis_plan) if args.analysis_plan else None)
    protocol=read(root/'artifacts/robustness'/f'{args.run_id}-screen.protocol.json')
    source=Path(protocol['source_snapshot']);env=os.environ.copy();env['PYTHONPATH']=str(source/'src')+os.pathsep+str(source/'scripts');env['PYTHONDONTWRITEBYTECODE']='1'
    child=subprocess.run([sys.executable,str(source/'scripts/robustness_r1_window.py'),'--run-id',args.run_id,'--execute','--project-root',str(root)],cwd=source,env=env)
    raise SystemExit(child.returncode)


if __name__=='__main__':main()
