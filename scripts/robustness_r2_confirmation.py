"""Fixed C2 runner, one K candidate, source-frozen and whole-family paired."""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_io import content_digest,write_json,write_json_gzip
from sca3_compass.robustness_registry import update_registry
from sca3_compass.robustness_dispatch import bounded_results
from sca3_compass.robustness_pattern_test import _json_value
from sca3_compass.robustness_bootstrap_guard import evaluate as reference
from sca3_compass.robustness_calibration_efficiency import evaluate as candidate
from robustness_r2_development import read,sha,score
from robustness_r1_window import generate,_r0
CONTEXT={}


def validate(plan):
    if plan['phase']!='C2' or plan['run_id']!='R0076' or plan['reporting_error_budget']!=.025:
        raise ValueError('Reserved C2 identity/budget required')
    if abs(sum(plan['error_allocation'].values())-.025)>1e-14:
        raise ValueError('C2 error allocation mismatch')
    if len(plan['cases'])!=84 or len(plan['repetition_counts'])!=84 or sum(plan['repetition_counts'])!=82800:
        raise ValueError('Fixed84 cases and82800 whole families required')
    if content_digest(plan['cases'])!=plan['case_definition_digest']:
        raise ValueError('Case definitions changed')
    if plan['candidate']!='R2K_pilotc0.5_projection_gate_eBH' or plan['reference']!='R1B_guard_pilotc0.5_projection_gate_eBH':
        raise ValueError('Candidate/reference changed')
    for label,indices in plan['strata'].items():
        if len({plan['repetition_counts'][i] for i in indices})!=1:
            raise ValueError('Equal-scene EB pooling requires equal counts')
    if len(plan['reported_methods'])!=124 or len(set(plan['reported_methods']))!=124:
        raise ValueError('Exactly124 distinct reported labels required')
    for members in plan['families'].values():
        if not set(members)<=set(plan['reported_methods']):
            raise ValueError('Unreported comparator')


def initialize(root,protocol):
    CONTEXT.update(root=Path(root),protocol=protocol,digest=content_digest(protocol))


def joint_evaluate(z,cal,seed):
    """Within-family memoization only; failed banks get the fresh K path."""
    r0,_=_r0(z,cal)
    r1,d1=reference(z,cal,seed=seed,draws=16)
    reusable=all(f['guard_success'] and len(f['bank'])==17 for f in d1['folds'])
    r2,d2=candidate(z,cal,seed=seed,draws=16,mode='K',
                    cached_folds=d1['folds'] if reusable else None)
    return {**r0,**r1,**r2},d1,d2,reusable


def one(case,rep):
    root,p=CONTEXT['root'],CONTEXT['protocol']
    folder=root/'artifacts/robustness'/f"{p['run_id']}-repetitions"/f'case-{case:04}'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'rep-{rep:06}.json.gz'
    if target.exists():
        old=read(target)
        if old['protocol_digest']!=CONTEXT['digest'] or old['case_index']!=case or old['rep']!=rep:
            raise ValueError('Existing record identity conflict')
        if old['status']=='completed':
            for k in ['input','evidence']:
                if sha(old[k+'_path'])!=old[k+'_sha256']:
                    raise ValueError('Resume array checksum mismatch')
        return {'status':old['status'],'case_index':case,'rep':rep,'path':str(target),'sha256':sha(target),'reused':True}
    record={'run_id':p['run_id'],'phase':'C2','method_version':p['method_version'],
        'protocol_digest':CONTEXT['digest'],'case_index':case,'rep':rep,
        'seed_sequence':[p['seed'],case,rep],'provenance':'SIMULATION_NOT_PATIENT_DATA'}
    started,cpu=time.perf_counter(),time.process_time()
    try:
        ip=folder/f'rep-{rep:06}-input.npz';ep=folder/f'rep-{rep:06}-evidence.npz'
        if ip.exists() or ep.exists():
            raise FileExistsError('Orphan arrays require separate audit; never overwrite')
        with threadpool_limits(1):
            z,cal,truth=generate(p['seed'],case,rep,p['cases'][case])
            with ip.open('xb') as stream:np.savez_compressed(stream,z=z,calibration=cal,truth=truth)
            seed=int(np.random.SeedSequence([p['seed'],case,rep,913]).generate_state(1)[0])
            e,d1,d2,reused=joint_evaluate(z,cal,seed)
            if set(e)!=set(p['analysis_plan']['reported_methods']):
                raise ValueError('Frozen method membership mismatch')
            metrics={m:score(v,truth) for m,v in e.items()}
            with ep.open('xb') as stream:
                np.savez_compressed(stream,**e,**{'p_R1_'+m:v for m,v in d1['held_pvalues'].items()},
                                    **{'p_R2_'+m:v for m,v in d2['held_pvalues'].items()})
            record.update(status='completed',input_path=str(ip),input_sha256=sha(ip),
                evidence_path=str(ep),evidence_sha256=sha(ep),n_true_signed=int(truth.sum()),metrics=metrics,
                diagnostics_R1=_json_value({k:v for k,v in d1.items() if k!='held_pvalues'}),
                diagnostics_R2=_json_value({k:v for k,v in d2.items() if k!='held_pvalues'}),
                numerical_reuse='FRESH_SAME_FAMILY_R1_INTERMEDIATES' if reused else 'FRESH_K_AFTER_INCOMPLETE_R1_BANK',
                reuse_notice='Legacy cached_development_only flag describes optional API cache; this run never reads development input/fits. Exact fresh K replay required.')
    except Exception as error:
        record.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    record.update(elapsed_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    write_json_gzip(target,record)
    return {'status':record['status'],'case_index':case,'rep':rep,'path':str(target),'sha256':sha(target)}


def batch(case,reps):
    result=[]
    for rep in reps:
        if cooperative_stop():break
        result.append(one(case,rep))
    return result


def freeze(root):
    base=root/'artifacts/robustness';plan=read(root/'configs/robustness_C2_analysis.json')
    validate(plan)
    if any(e.get('settings',{}).get('phase')=='C2' for e in read(base/'EXPERIMENT_REGISTRY.json')['experiments']):
        raise ValueError('C2 already registered; no third confirmation or hidden repeat')
    if sha(ROOT/'src/sca3_compass/robustness_calibration_efficiency.py')!=plan['candidate_source_sha256']:
        raise ValueError('K changed since plan')
    if sha(ROOT/'src/sca3_compass/robustness_bootstrap_guard.py')!=plan['reference_source_sha256']:
        raise ValueError('Frozen R1 changed')
    for rel,digest in plan['development_evidence_sha256'].items():
        if sha(root/rel)!=digest:raise ValueError('DEV or R1 closure evidence changed')
    source=base/'R0076-source';source.mkdir(exist_ok=False)
    scripts=['robustness_r2_confirmation.py','robustness_r2_confirm_analyze.py','robustness_r2_confirm_replay.py','robustness_r2_confirmation_plan.py',
             'robustness_r2_development.py','robustness_r2_analyze.py','robustness_r1_window.py','robustness_screen.py','research_window.py']
    paths=[*sorted((ROOT/'src/sca3_compass').glob('*.py')),*[ROOT/'scripts'/name for name in scripts]]
    hashes={}
    for path in paths:
        rel=path.relative_to(ROOT);dest=source/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);hashes[rel.as_posix()]=sha(dest)
    # Existing R1 scientific dependencies must be byte-identical to C1;
    # genuinely new R2 files are outside the C1 manifest.
    old=read(base/'R0073-screen.protocol.json')
    for rel,digest in old['source_sha256'].items():
        if rel.startswith('src/') and hashes.get(rel)!=digest:
            raise ValueError('R1 scientific dependency changed:'+rel)
    import platform,scipy
    p={'run_id':'R0076','phase':'C2','method_version':plan['method_version'],'seed':plan['seed'],
       'cases':plan['cases'],'repetition_counts':plan['repetition_counts'],'workers':8,'batch_size':4,
       'source_snapshot':str(source),'source_sha256':hashes,'analysis_plan':plan,
       'created_utc':datetime.now(timezone.utc).isoformat(),'provenance':'SIMULATION_NOT_PATIENT_DATA',
       'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
           'platform':platform.platform(),'BLAS_threads_per_worker':1},
       'replay_plan':{'cases':[0,27,59,63,75,77,78,83],'reps':[0,1],
           'extra':'All full-guard failure records and first K fallback record per affected case; no new samples'}}
    write_json(base/'R0076-screen.protocol.json',p)
    shutil.copy2(root/'configs/robustness_C2_analysis.json',source/'analysis_plan.json')
    update_registry(base/'EXPERIMENT_REGISTRY.json',{'id':'R0076','status':'frozen','purpose':'C2',
        'method_version':p['method_version'],'settings':p,'started_utc':p['created_utc']},create=True)
    return p


def execute(root):
    base=root/'artifacts/robustness';p=read(base/'R0076-screen.protocol.json');validate(p['analysis_plan'])
    for rel,digest in p['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source changed:'+rel)
    registry=base/'EXPERIMENT_REGISTRY.json'
    entry=next(r for r in read(registry)['experiments'] if r['id']=='R0076')
    if entry['status']=='completed':raise ValueError('Already completed, no relaunch')
    start=time.perf_counter();entry.update(status='running',pid=os.getpid(),execution_started_utc=datetime.now(timezone.utc).isoformat())
    update_registry(registry,entry);receipts=[]
    jobs=[((i,k),batch,(i,list(range(k,min(k+4,n)))),{}) for i,n in enumerate(p['repetition_counts']) for k in range(0,n,4)]
    try:
        with ProcessPoolExecutor(max_workers=8,initializer=initialize,initargs=(str(root),p)) as pool:
            for key,items in bounded_results(pool,jobs,max_pending=8):
                receipts.extend(items)
                progress={'run_id':'R0076','completed':sum(r['status']=='completed' for r in receipts),
                    'failed':sum(r['status']!='completed' for r in receipts),'planned':82800,
                    'updated_utc':datetime.now(timezone.utc).isoformat(),'last_batch':list(key)}
                write_json(base/'R0076-progress.json',progress)
                print(progress,flush=True)
                if cooperative_stop():break
        complete=len(receipts)==82800 and all(r['status']=='completed' for r in receipts)
        index=base/'R0076-results-index.json'
        write_json(index,{'run_id':'R0076','settings':p,'complete':complete,'receipts':receipts,'elapsed_seconds':time.perf_counter()-start})
        entry.update(status='completed' if complete else 'interrupted' if cooperative_stop() else 'failed',
            result=str(index),sha256=sha(index),elapsed_seconds=time.perf_counter()-start,finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        entry.update(status='failed',error=repr(error),elapsed_seconds=time.perf_counter()-start)
        raise
    finally:update_registry(registry,entry)
    if not complete:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=ROOT)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true');args=p.parse_args()
    root=args.project_root.resolve()
    if args.execute:return execute(root)
    protocol=read(root/'artifacts/robustness/R0076-screen.protocol.json') if args.resume else freeze(root)
    source=Path(protocol['source_snapshot'])
    env=dict(os.environ,PYTHONPATH=str(source/'src')+os.pathsep+str(source/'scripts'),PYTHONDONTWRITEBYTECODE='1')
    raise SystemExit(subprocess.run([sys.executable,str(source/'scripts/robustness_r2_confirmation.py'),
        '--project-root',str(root),'--execute'],cwd=source,env=env).returncode)


if __name__=='__main__':main()
