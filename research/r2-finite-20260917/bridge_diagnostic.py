"""One fixed predictive-bridge development rescore; no new target data."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import gzip
import json
import shutil
import time
import os
import sys
sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parent
DATA_PHASE = PHASE if (PHASE/'D001/index.json').exists() else PHASE.parent
sys.path.insert(0, str(PHASE))
import numpy as np
from threadpoolctl import threadpool_limits
from experiment import sha, read, write, score, js
from predictive_bridge import evaluate
CASES = [0, 1, 2, 4, 5, 7, 9, 10]
FILES = ['bridge_diagnostic.py', 'predictive_bridge.py', 'finite_calibration.py', 'experiment.py']


def prepare(out):
    if out.exists(): raise FileExistsError('preserve existing diagnostic')
    out.mkdir()
    p = {'id': 'R2FC-D002', 'stage': 'DEVELOPMENT_REUSE_D001', 'cases': CASES,
         'reps': 32, 'reference_draws': 4095, 'iterations': 2, 'workers': 6,
         'seed': 1709172202, 'total': len(CASES)*32,
         'stop': 'fixed256storedfamilies, no enlargement or confirmation claim',
         'error_budget': 'q=.05; exact rank p integrates auxiliary MC, no failure-probability deduction',
         'source_index_sha256': sha(PHASE/'D001/index.json')}
    write(out/'protocol.json', p)
    for file in FILES: shutil.copy2(PHASE/file, out/file)
    write(out/'freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
                             'files': {f: sha(out/f) for f in FILES+['protocol.json']}})


def one(task):
    folder, case, rep = task; out = Path(folder); p = read(out/'protocol.json')
    record = out/f'raw/case-{case:02}/rep-{rep:05}.json.gz'
    if record.exists(): raise FileExistsError('no duplicate attempt')
    record.parent.mkdir(parents=True, exist_ok=True)
    old = DATA_PHASE/f'D001/raw/case-{case:02}/rep-{rep:05}.json.gz'
    with gzip.open(old, 'rt', encoding='utf-8') as stream: prior = json.load(stream)
    source = DATA_PHASE/'D001'/prior['input_path']
    if sha(source) != prior['input_sha256']: raise ValueError('prior input changed')
    with np.load(source, allow_pickle=False) as a:
        z, cal, truth = a['z'], a['calibration'], a['truth']
    seed = int(np.random.SeedSequence([p['seed'], case, rep, 917]).generate_state(1)[0])
    started = time.perf_counter(); cpu = time.process_time()
    with threadpool_limits(1):
        result = evaluate(z, cal, seed=seed, reference_draws=p['reference_draws'], iterations=p['iterations'])
    ep = record.with_name(record.name.replace('.json.gz', '-evidence.npz'))
    arrays = {'decision_'+k:v for k,v in result['decisions'].items()}
    arrays.update({'e_'+k:v for k,v in result['evidence'].items()})
    arrays.update(p=result['p'], reference=result['reference'])
    with ep.open('xb') as stream: np.savez_compressed(stream, **arrays)
    row = {k:v for k,v in result.items() if k not in ['decisions', 'evidence', 'p', 'reference']}
    row.update(case=case, rep=rep, algorithm_seed=seed, input_path=str(source), input_sha256=sha(source),
               source_record_sha256=sha(old), evidence_path=ep.relative_to(out).as_posix(), evidence_sha256=sha(ep),
               metrics={**prior['metrics'], **{k:score(v, truth) for k,v in result['decisions'].items()}},
               wall_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
               finished_utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(record, 'xt', encoding='utf-8') as stream: json.dump(row, stream, default=js, allow_nan=False)
    return {'case': case, 'rep': rep, 'path': record.relative_to(out).as_posix(), 'sha256': sha(record), 'status': result['status']}


def run(out):
    p = read(out/'protocol.json'); fr = read(out/'freeze.json')
    for file in FILES:
        if sha(PHASE/file) != fr['files'][file] or sha(out/file) != fr['files'][file]: raise ValueError('source freeze differs')
    if sha(out/'protocol.json') != fr['files']['protocol.json']: raise ValueError('protocol changed')
    if (out/'started.json').exists(): raise FileExistsError('inspect original state; no automatic restart')
    write(out/'started.json', {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat()})
    tasks = [(str(out), i, rep) for i in p['cases'] for rep in range(p['reps'])]
    rows = []; start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        for r in pool.map(one, tasks):
            rows.append(r)
            if len(rows)%32 == 0: print(json.dumps({'attempts':len(rows), 'total':len(tasks), 'seconds':time.perf_counter()-start}), flush=True)
    write(out/'index.json', {'rows': rows, 'seconds': time.perf_counter()-start, 'complete': len(rows)==p['total']})


def analyze(out):
    p = read(out/'protocol.json'); index = read(out/'index.json'); results = []
    for case in p['cases']:
        rows = []
        for item in index['rows']:
            if item['case'] != case: continue
            if sha(out/item['path']) != item['sha256']: raise ValueError('changed record')
            with gzip.open(out/item['path'], 'rt', encoding='utf-8') as stream: rows.append(json.load(stream))
        metrics = {}
        for method in rows[0]['metrics']:
            metrics[method] = {key: None if rows[0]['metrics'][method][key] is None else float(np.mean([r['metrics'][method][key] for r in rows])) for key in ['power', 'fdp']}
        result = {'case': case, 'reps': len(rows), 'metrics': metrics,
                  'conservative_failures': sum(r['status'] != 'completed' for r in rows),
                  'mean_seconds': float(np.mean([r['wall_seconds'] for r in rows])),
                  'mean_pilot_gamma': np.mean([x['gamma_from_pilot'] for r in rows for x in r['folds']], axis=0)}
        results.append(result); print(json.dumps(result, default=js), flush=True)
    write(out/'summary.json', {'stage': p['stage'], 'rows': results, 'index_sha256': sha(out/'index.json')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run', 'analyze']); parser.add_argument('--out', required=True)
    a = parser.parse_args(); out = Path(a.out).resolve()
    if a.action == 'prepare': prepare(out)
    elif a.action == 'run': run(out)
    else: analyze(out)
