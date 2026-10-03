"""Small R2 experiments, archived code/protocol and raw families, no V1 writes."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import traceback
import shutil
sys.dont_write_bytecode = True
PHASE = Path(__file__).resolve().parent
ROOT = next(p for p in PHASE.parents if (p/'releases/K-NR-1.0.0/release.json').exists())
sys.path[:0] = [str(PHASE), str(ROOT/'releases/K-NR-1.0.0/source/src'), str(ROOT/'releases/K-NR-1.0.0/source/scripts')]
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from finite_calibration import evaluate, matrix_kappa_upper, shape_fit, contrasts, reference_cutoff
from sca3_compass.molecular_v1 import evaluate as evaluate_v1
from sca3_compass.molecular_methods import covariance_root
from robustness_screen import data, radial, simulation_equicorrelation


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def js(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, default=js, ensure_ascii=False, indent=2, allow_nan=False)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def generate(p, i, rep):
    c = p['cases'][i]
    rng = np.random.default_rng(np.random.SeedSequence([p['seed'], i, rep]))
    z, original_calibration, truth = data(rng, c, g=p['family_G'])
    r = simulation_equicorrelation(4, c.get('study_rho', .65))
    calibration = original_calibration.transpose(1, 0, 2).copy()
    if c.get('mixed_sign_null'):
        # Regenerate this special null directly, not by subtracting guessed effects.
        rt = simulation_equicorrelation(6, c['rho'])
        z = covariance_root(r) @ rng.normal(size=(p['family_G'], 4, 6)) @ covariance_root(rt).T
        z *= radial(rng, (p['family_G'], 1, 1), c['distribution'])
        mu = np.zeros((p['family_G'], 4))
        mu[:, 0] = c['effect']; mu[:, 1] = -c['effect']
        z += mu[..., None]
        truth = np.zeros((p['family_G'], 2), bool)
    kappa = (1+5*c['rho'])/(6*(1-c['rho']))
    return z, calibration, truth, r, kappa


def score(decision, truth):
    if decision.dtype != bool or decision.shape != truth.shape:
        raise ValueError('signed decision shape')
    tp = int((decision & truth).sum()); fp = int((decision & ~truth).sum())
    return {'power': tp/int(truth.sum()) if truth.any() else None,
            'fdp': fp/max(1, tp+fp), 'tp': tp, 'fp': fp, 'discoveries': tp+fp}


def one(task):
    output, i, rep = task
    out = Path(output); p = read(out/'protocol.json'); freeze = read(out/'freeze.json')
    record = out/f'raw/case-{i:02}/rep-{rep:05}.json.gz'
    if record.exists():
        with gzip.open(record, 'rt', encoding='utf-8') as stream: row = json.load(stream)
        if row['freeze_sha256'] != sha(out/'freeze.json'): raise ValueError('checkpoint freeze differs')
        for key in ['input', 'evidence']:
            if key+'_path' in row and sha(out/row[key+'_path']) != row[key+'_sha256']: raise ValueError('checkpoint content differs')
        return {'case': i, 'rep': rep, 'status': row['status'], 'path': record.relative_to(out).as_posix(), 'sha256': sha(record)}
    started = time.perf_counter(); cpu = time.process_time()
    row = {'case': i, 'rep': rep, 'freeze_sha256': sha(out/'freeze.json'), 'provenance': 'SIMULATION_NOT_PATIENT_DATA'}
    record.parent.mkdir(parents=True, exist_ok=True)
    try:
        with threadpool_limits(1):
            z, cal, truth, r, kappa = generate(p, i, rep)
            ip = record.with_name(record.name.replace('.json.gz', '-input.npz'))
            with ip.open('xb') as stream: np.savez_compressed(stream, z=z, calibration=cal, truth=truth, shape=r, kappa=kappa)
            row.update(input_path=ip.relative_to(out).as_posix(), input_sha256=sha(ip))
            seed = int(np.random.SeedSequence([p['seed'], i, rep, 913]).generate_state(1)[0])
            begin = time.perf_counter()
            result = evaluate(z, cal, seed=seed, reference_draws=p['reference_draws'], delta_kappa=p['delta_kappa'], oracle=(r, kappa))
            r2_seconds = time.perf_counter()-begin
            begin = time.perf_counter()
            legacy = evaluate_v1(z, cal.transpose(1, 0, 2), seed=seed, acknowledge_scope=True)
            v1_seconds = time.perf_counter()-begin
            decisions = {**result['decisions'], **{k: legacy['discoveries'][k] for k in p['methods'] if k in legacy['discoveries']}}
            ep = record.with_name(record.name.replace('.json.gz', '-evidence.npz'))
            arrays = {'decision_'+k: v for k, v in decisions.items()}
            arrays.update({'e_'+k: v for k, v in result['evidence'].items()})
            arrays.update(p=result['p'], bb_p=result['bb_p'], shape_reference=result['shape_reference'])
            with ep.open('xb') as stream: np.savez_compressed(stream, **arrays)
            row.update(status='completed', metrics={k: score(decisions[k], truth) for k in p['methods']},
                       evidence_path=ep.relative_to(out).as_posix(), evidence_sha256=sha(ep),
                       algorithm_seed=seed, kappa_upper=result['kappa_upper'], kappa_true=kappa,
                       kappa_covered=result['kappa_upper'] >= kappa, kappa_ratio=result['kappa_upper']/kappa,
                       shape_covered=[x['true_shape_distortion'] <= x['cutoff'] for x in result['folds']],
                       r2_folds=result['folds'], calibration={k:v for k,v in result['calibration'].items() if k != 'ratios'},
                       r2_seconds=r2_seconds, v1_seconds=v1_seconds,
                       r2_internal_q=result['q_internal'], v1_status=legacy['status'])
    except Exception as err:
        row.update(status='failed', error=repr(err), traceback=traceback.format_exc())
    row.update(seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu,
               finished_utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(record, 'xt', encoding='utf-8') as stream:
        json.dump(row, stream, default=js, allow_nan=False)
    return {'case': i, 'rep': rep, 'status': row['status'], 'path': record.relative_to(out).as_posix(), 'sha256': sha(record)}


def freeze(protocol, out):
    if out.exists(): raise FileExistsError('do not overwrite phase')
    out.mkdir(parents=True)
    shutil.copy2(protocol, out/'protocol.json')
    shutil.copy2(__file__, out/'experiment.py')
    shutil.copy2(PHASE/'finite_calibration.py', out/'finite_calibration.py')
    # Frozen scripts remain under the same parent depth by importing ROOT explicitly at run time.
    write(out/'freeze.json', {'created_utc': datetime.now(timezone.utc).isoformat(),
         'source_root': str(ROOT), 'files': {x: sha(out/x) for x in ['protocol.json', 'experiment.py', 'finite_calibration.py']},
         'v1_source_sha256': sha(ROOT/'releases/K-NR-1.0.0/source/src/sca3_compass/molecular_v1.py'),
         'v1_zip_sha256': sha(ROOT/'releases/K-NR-1.0.0.zip')})


def run(out):
    p = read(out/'protocol.json'); fr = read(out/'freeze.json')
    for file, digest in fr['files'].items():
        if sha(out/file) != digest: raise ValueError('frozen file changed')
    if sha(PHASE/'finite_calibration.py') != fr['files']['finite_calibration.py'] or sha(__file__) != fr['files']['experiment.py']:
        raise ValueError('working executor differs from frozen copy')
    if (out/'index.json').exists(): raise FileExistsError('batch already ended; inspect instead of restarting')
    tasks = [(str(out), i, r) for i in range(len(p['cases'])) for r in range(p['repetitions'])]
    write(out/'started.json', {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(), 'total': len(tasks)})
    started = time.perf_counter(); rows = []
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        for row in pool.map(one, tasks, chunksize=1):
            rows.append(row)
            if len(rows)%64 == 0:
                print(json.dumps({'completed': len(rows), 'planned': len(tasks), 'failed': sum(x['status'] != 'completed' for x in rows), 'seconds': time.perf_counter()-started}), flush=True)
    write(out/'index.json', {'rows': rows, 'complete': len(rows) == len(tasks), 'failed': sum(x['status'] != 'completed' for x in rows), 'seconds': time.perf_counter()-started})


def analyze(out):
    p = read(out/'protocol.json'); index = read(out/'index.json'); result = []
    for i, case in enumerate(p['cases']):
        rows = []
        for item in index['rows']:
            if item['case'] != i: continue
            if sha(out/item['path']) != item['sha256']: raise ValueError('record changed')
            with gzip.open(out/item['path'], 'rt', encoding='utf-8') as stream: rows.append(json.load(stream))
        failures = [r for r in rows if r['status'] != 'completed']
        if failures:
            result.append({'case': case, 'failures': failures, 'status': 'failed_no_success_only_means'}); continue
        stats = {}
        for method in p['methods']:
            stats[method] = {}
            for metric in ['power', 'fdp', 'discoveries']:
                v = [r['metrics'][method][metric] for r in rows]
                stats[method][metric] = None if v[0] is None else float(np.mean(v))
        result.append({'case': case, 'repetitions': len(rows), 'methods': stats,
                       'kappa_noncoverage': sum(not r['kappa_covered'] for r in rows),
                       'shape_noncoverage': sum(not all(r['shape_covered']) for r in rows),
                       'median_kappa_ratio': float(np.median([r['kappa_ratio'] for r in rows])),
                       'mean_r2_seconds': float(np.mean([r['r2_seconds'] for r in rows])),
                       'mean_v1_seconds': float(np.mean([r['v1_seconds'] for r in rows]))})
    write(out/'summary.json', {'stage': p['stage'], 'rows': result, 'index_sha256': sha(out/'index.json')})
    for r in result:
        print(r['case']['name'], json.dumps(r.get('methods', r.get('status'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['freeze', 'run', 'analyze'])
    parser.add_argument('--out', required=True); parser.add_argument('--protocol', default=str(PHASE/'protocol-dev.json'))
    args = parser.parse_args(); out = Path(args.out).resolve()
    if args.action == 'freeze': freeze(args.protocol, out)
    elif args.action == 'run': run(out)
    else: analyze(out)
