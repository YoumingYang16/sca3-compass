"""Two predeclared targeted R2 DEV batches, never C2 confirmation.

R0074 reuses ALL7600 R0072 families for paired protection decomposition.
R0075 uses the eight predeclared smaller-calibration diagnostic conditions.
Source frozen, bounded dispatch, explicit failures and no scientific overwrite.
"""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_calibration_efficiency import evaluate
from sca3_compass.robustness_bootstrap_guard import evaluate as reference
from sca3_compass.robustness_pattern_test import _json_value
from sca3_compass.robustness_dispatch import bounded_results
from sca3_compass.robustness_io import content_digest, write_json, write_json_gzip
from sca3_compass.robustness_registry import update_registry
from sca3_compass.molecular_methods import ebh
from robustness_r1_window import generate, _r0

CONTEXT = {}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz')
          else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)


def score(e, truth):
    if e.shape != truth.shape or not np.isfinite(e).all() or np.any(e < 0):
        raise ValueError('Invalid signed evidence')
    selected = ebh(e, .05)
    total, tp, fp = int(selected.sum()), int((selected&truth).sum()), int((selected&~truth).sum())
    return {'discoveries': total, 'tp': tp, 'fp': fp, 'fdp': fp/max(1,total),
            'power': tp/max(1,int(truth.sum())), 'power_defined': bool(truth.sum())}


def initialize(root, protocol):
    CONTEXT.update(root=Path(root), protocol=protocol, digest=content_digest(protocol))
    if protocol['kind'] == 'cached':
        path = Path(root)/'artifacts/robustness/R0072-results-index.json'
        if sha(path) != protocol['development_index_sha256']:
            raise ValueError('Original DEV index changed')
        index = read(path)
        CONTEXT['old'] = {(r['case_index'],r['rep']):r for r in index['receipts']}


def one(case, rep):
    root, protocol = CONTEXT['root'], CONTEXT['protocol']
    run = protocol['run_id']
    folder = root/'artifacts/robustness'/f'{run}-repetitions'/f'case-{case:04}'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder/f'rep-{rep:06}.json.gz'
    if target.exists():
        saved = read(target)
        if saved['protocol_digest'] != CONTEXT['digest'] or saved['case_index'] != case or saved['rep'] != rep:
            raise ValueError('Existing record conflict')
        return {'status': saved['status'], 'case_index': case, 'rep': rep, 'path': str(target), 'sha256': sha(target), 'reused': True}
    started, cpu = time.perf_counter(), time.process_time()
    evidence_path = folder/f'rep-{rep:06}-evidence.npz'
    record = {'run_id': run, 'phase': 'development', 'method_version': protocol['method_version'],
              'protocol_digest': CONTEXT['digest'], 'case_index': case, 'rep': rep,
              'seed_sequence': [protocol['seed'],case,rep], 'provenance': 'SIMULATION_NOT_PATIENT_DATA'}
    try:
        if evidence_path.exists():
            raise FileExistsError('Orphan evidence needs explicit audit; never overwrite')
        seed = int(np.random.SeedSequence([protocol['seed'],case,rep,913]).generate_state(1)[0])
        with threadpool_limits(1):
            if protocol['kind'] == 'cached':
                receipt = CONTEXT['old'][(case,rep)]
                if sha(receipt['path']) != receipt['sha256']:
                    raise ValueError('Cached original record checksum mismatch')
                old = read(receipt['path'])
                if (old['status'] != 'completed' or old['seed_sequence'] != [protocol['seed'],case,rep]
                        or old['protocol_digest'] != protocol['development_protocol_digest']):
                    raise ValueError('Original DEV identity mismatch')
                for key in ['input', 'evidence']:
                    if sha(old[key+'_path']) != old[key+'_sha256']:
                        raise ValueError('Original input/evidence checksum mismatch')
                input_path = Path(old['input_path'])
                with np.load(input_path) as archive:
                    z, cal, truth = archive['z'], archive['calibration'], archive['truth'].astype(bool)
                proposed, details = evaluate(z, cal, seed=seed, cached_folds=old['diagnostics_R1']['folds'])
                with np.load(old['evidence_path']) as archive:
                    for name, values in proposed.items():
                        if name.startswith('R2P_'):
                            np.testing.assert_array_equal(values, archive[name.replace('R2P_', 'R1B_plugin_')])
                all_e = proposed
                old_metrics = old['metrics']
                record.update(reference_record=receipt, reference_evidence=old['evidence_path'],
                              reference_evidence_sha256=old['evidence_sha256'], exact_plugin_replay=True)
            else:
                z, cal, truth = generate(protocol['seed'],case,rep,protocol['cases'][case])
                input_path = folder/f'rep-{rep:06}-input.npz'
                if input_path.exists():
                    raise FileExistsError('Orphan input needs explicit audit')
                with input_path.open('xb') as stream:
                    np.savez_compressed(stream,z=z,calibration=cal,truth=truth)
                r0, _ = _r0(z, cal)
                r1, d1 = reference(z,cal,seed=seed,draws=16)
                proposed, details = evaluate(z,cal,seed=seed,draws=16)
                for name, values in proposed.items():
                    if name.startswith('R2P_'):
                        np.testing.assert_array_equal(values, r1[name.replace('R2P_', 'R1B_plugin_')])
                all_e = {**r0,**r1,**proposed}
                old_metrics = {}
                record.update(diagnostics_R1=_json_value({k:v for k,v in d1.items() if k!='held_pvalues'}), exact_plugin_replay=True)
            metrics = {**old_metrics, **{name:score(values,truth) for name,values in all_e.items()}}
            with evidence_path.open('xb') as stream:
                np.savez_compressed(stream,**all_e,**{'p_'+name:values for name,values in details['held_pvalues'].items()})
            record.update(status='completed', input_path=str(input_path), input_sha256=sha(input_path),
                          evidence_path=str(evidence_path), evidence_sha256=sha(evidence_path),
                          n_true_signed=int(truth.sum()), metrics=metrics,
                          diagnostics_R2=_json_value({k:v for k,v in details.items() if k!='held_pvalues'}))
    except Exception as error:
        record.update(status='failed', error=repr(error), traceback=traceback.format_exc())
    record.update(elapsed_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu,
                  finished_utc=datetime.now(timezone.utc).isoformat())
    write_json_gzip(target,record)
    return {'status':record['status'],'case_index':case,'rep':rep,'path':str(target),'sha256':sha(target)}


def batch(case, reps):
    result=[]
    for rep in reps:
        if cooperative_stop():
            break
        result.append(one(case,rep))
    return result


def freeze(root, run):
    base=root/'artifacts/robustness'
    plan_path=root/'configs/robustness_R2_targeted_development.json'
    plan=read(plan_path)
    if sha(root/plan['R1_closed_evidence']) != plan['R1_analysis_sha256']:
        raise ValueError('R1 closure evidence changed')
    r1=read(root/plan['R1_closed_evidence'])
    replay=read(base/'R0073-replay.json')
    if not r1['complete'] or not r1['in_scope_candidate_fdr_all_upper_le_05'] or len(replay['receipts'])!=10:
        raise ValueError('R1 scoped reference closure required')
    if run not in ('R0074','R0075'):
        raise ValueError('Only the two predeclared DEV batches; no C2 or new search')
    protocol_path=base/f'{run}-screen.protocol.json'
    source=base/f'{run}-source'
    if protocol_path.exists() or source.exists():
        raise FileExistsError('Existing protocol/source; resume explicitly')
    source.mkdir()
    paths=[*sorted((SOURCE/'src/sca3_compass').glob('*.py')),
           SOURCE/'scripts/robustness_r2_development.py',SOURCE/'scripts/robustness_r1_window.py',
           SOURCE/'scripts/robustness_screen.py',SOURCE/'scripts/research_window.py']
    hashes={}
    for path in paths:
        rel=path.relative_to(SOURCE)
        destination=source/rel
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        hashes[str(rel).replace('\\','/')]=sha(destination)
    original=read(base/'R0072-screen.protocol.json')
    is_cached=run=='R0074'
    setup=plan['existing_development' if is_cached else 'targeted_fresh_development']
    protocol={'run_id':run,'phase':'development','kind':'cached' if is_cached else 'fresh',
              'method_version':'R2-protection-efficiency-dev1','plan_sha256':sha(plan_path),'plan':plan,
              'seed':original['seed'] if is_cached else setup['seed'],
              'cases':original['cases'] if is_cached else setup['cases'],
              'repetitions':100 if is_cached else setup['repetitions_per_case'],
              'workers':setup['workers_max'],'batch_size':4,'source_snapshot':str(source),'source_sha256':hashes,
              'development_index_sha256':sha(base/'R0072-results-index.json'),
              'development_protocol_digest':content_digest(original),'provenance':'SIMULATION_NOT_PATIENT_DATA'}
    if len(protocol['cases'])*protocol['repetitions']!=setup['whole_family_limit']:
        raise ValueError('Fixed DEV count mismatch')
    write_json(protocol_path,protocol)
    shutil.copy2(plan_path,source/'development_plan.json')
    update_registry(base/'EXPERIMENT_REGISTRY.json',{'id':run,'status':'frozen',
                    'method_version':protocol['method_version'],'purpose':'development',
                    'settings':protocol,'started_utc':datetime.now(timezone.utc).isoformat()},create=True)


def execute(root, run):
    base=root/'artifacts/robustness'
    protocol=read(base/f'{run}-screen.protocol.json')
    for rel,digest in protocol['source_sha256'].items():
        if sha(SOURCE/rel)!=digest:
            raise ValueError('Frozen source mismatch:'+rel)
    registry=base/'EXPERIMENT_REGISTRY.json'
    entry=next(r for r in read(registry)['experiments'] if r['id']==run)
    if entry['status']=='completed':
        raise ValueError('Already completed; no duplicate launch')
    start=time.perf_counter()
    entry.update(status='running',pid=os.getpid(),cwd=str(SOURCE),execution_started_utc=datetime.now(timezone.utc).isoformat())
    update_registry(registry,entry)
    jobs=[((i,first),batch,(i,list(range(first,min(first+protocol['batch_size'],protocol['repetitions'])))),{})
          for i in range(len(protocol['cases'])) for first in range(0,protocol['repetitions'],protocol['batch_size'])]
    receipts=[]
    try:
        with ProcessPoolExecutor(max_workers=protocol['workers'],initializer=initialize,initargs=(str(root),protocol)) as pool:
            for key,items in bounded_results(pool,jobs,max_pending=protocol['workers']):
                receipts.extend(items)
                progress={'run_id':run,'completed':sum(r['status']=='completed' for r in receipts),
                          'failed':sum(r['status']!='completed' for r in receipts),
                          'planned':len(protocol['cases'])*protocol['repetitions'],
                          'updated_utc':datetime.now(timezone.utc).isoformat(),'last_batch':list(key)}
                write_json(base/f'{run}-progress.json',progress)
                print(progress,flush=True)
                if cooperative_stop():
                    break
        complete=len(receipts)==len(protocol['cases'])*protocol['repetitions'] and all(r['status']=='completed' for r in receipts)
        index=base/f'{run}-results-index.json'
        write_json(index,{'run_id':run,'settings':protocol,'complete':complete,'receipts':receipts,'elapsed_seconds':time.perf_counter()-start})
        entry.update(status='completed' if complete else 'interrupted' if cooperative_stop() else 'failed',
                     result=str(index),sha256=sha(index),finished_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
    except BaseException as error:
        entry.update(status='failed',error=repr(error),elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        update_registry(registry,entry)
    if not complete:
        raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',choices=['R0074','R0075'],required=True)
    parser.add_argument('--project-root',type=Path,default=SOURCE)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=args.project_root.resolve()
    if args.execute:
        return execute(root,args.run_id)
    if not args.resume:
        freeze(root,args.run_id)
    protocol=read(root/'artifacts/robustness'/f'{args.run_id}-screen.protocol.json')
    source=Path(protocol['source_snapshot'])
    env=dict(os.environ,PYTHONPATH=str(source/'src')+os.pathsep+str(source/'scripts'),PYTHONDONTWRITEBYTECODE='1')
    raise SystemExit(subprocess.run([sys.executable,str(source/'scripts/robustness_r2_development.py'),
                     '--run-id',args.run_id,'--execute','--project-root',str(root)],cwd=source,env=env).returncode)


if __name__=='__main__':
    main()
