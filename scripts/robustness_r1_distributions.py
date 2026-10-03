"""Complete-C1 descriptive family distributions and compute accounting.

No new samples, inference, candidate selection or intervals. Quantiles use
empirical inverse CDF and complete families, never individual correlated genes.
Must follow the frozen whole-array confirmation analyzer.
"""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from sca3_compass.robustness_io import write_json


def read(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def distribution(values):
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Finite nonempty family vector required')
    return {'mean': float(values.mean()), 'sd': float(values.std(ddof=1)) if len(values)>1 else None,
            'min': float(values.min()), 'max': float(values.max()),
            'quantiles': {str(q): float(np.quantile(values, q, method='inverted_cdf'))
                          for q in [.01, .05, .5, .9, .95, .99]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    folder = ROOT/'artifacts/robustness'
    analysis_path = folder/f'{args.run_id}-confirmation-analysis.json'
    analysis = read(analysis_path)
    if not analysis['complete'] or analysis['phase'] != 'C1':
        raise ValueError('Complete frozen C1 analysis required; no partial-result inspection')
    index_path = folder/f'{args.run_id}-results-index.json'
    if sha(index_path) != analysis['index_sha256']:
        raise ValueError('Index changed after frozen analysis')
    index = read(index_path)
    if not index['complete'] or len(index['receipts']) != analysis['whole_family_repetitions']:
        raise ValueError('Complete fixed sample required')
    out = folder/f'{args.run_id}-distributions'
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out/'generate_source.py')
    names = analysis['analysis_plan']['reported_methods']
    grouped = {i: [] for i in range(len(index['settings']['cases']))}
    for receipt in index['receipts']:
        grouped[receipt['case_index']].append(receipt)
    result = {'run_id': args.run_id, 'complete': False,
              'status': 'SECONDARY_DESCRIPTIVE_NO_ADDITIONAL_CONFIDENCE_CLAIMS',
              'provenance': 'SIMULATION_NOT_PATIENT_DATA', 'analysis_sha256': sha(analysis_path),
              'index_sha256': sha(index_path), 'script_sha256': sha(__file__),
              'unit': 'whole independently generated target/calibration/algorithm family',
              'quantile_definition': 'numpy empirical inverted_cdf; no quantile confidence intervals',
              'rows': [], 'whole_family_count': 0, 'total_recorded_cpu_seconds': 0.,
              'total_recorded_family_wall_seconds': 0.,
              'cost_limits': ['All88 labels plus record preparation, not per-method deployment cost',
                              'Sum of family wall times is not elapsed batch time under parallelism',
                              'Lost unfinished work at reboot, recovery audits, import/cache rebuild and other jobs are excluded',
                              'Single-family wall time can include contention from concurrent CPU jobs']}
    started = time.monotonic()
    summary_rows = {r['case_index']: r for r in analysis['rows']}
    for case, receipts in grouped.items():
        if cooperative_stop():
            write_json(out/'partial.json', result)
            raise InterruptedError('Bounded postprocessing checkpoint; not complete')
        n = len(receipts)
        if n != index['settings']['repetition_counts'][case]:
            raise ValueError('Unexpected per-case repeat count')
        fdp, power = np.empty((n, len(names))), np.empty((n, len(names)))
        cpu, wall = np.empty(n), np.empty(n)
        no_discovery = np.empty_like(fdp)
        guard_failures = pattern_fallbacks = 0
        for pos, receipt in enumerate(sorted(receipts, key=lambda r: r['rep'])):
            if receipt['rep'] != pos or receipt['status'] != 'completed' or sha(receipt['path']) != receipt['sha256']:
                raise ValueError('Changed/incomplete/duplicate family')
            r = read(receipt['path'])
            if r['run_id'] != args.run_id or r['case_index'] != case or r['rep'] != pos or r['status'] != 'completed':
                raise ValueError('Wrong archived family identity')
            for col, method in enumerate(names):
                m = r['metrics'][method]
                fdp[pos,col], power[pos,col] = m['fdp'], m['power']
                no_discovery[pos,col] = m['discoveries'] == 0
            cpu[pos], wall[pos] = r['cpu_seconds'], r['elapsed_seconds']
            folds = r['diagnostics_R1']['folds']
            guard_failures += sum(not f['guard_success'] for f in folds)
            pattern_fallbacks += sum(f['pattern']['fallback'] for f in folds)
        if not np.isfinite(fdp).all() or not np.isfinite(power).all() or np.any((fdp<0)|(fdp>1)) or np.any((power<0)|(power>1)):
            raise ValueError('Invalid bounded family outcomes')
        methods = {}
        for col, method in enumerate(names):
            ref = summary_rows[case]['methods'][method]
            p = None if ref['power'] is None else distribution(power[:,col])
            f = distribution(fdp[:,col])
            if abs(f['mean']-ref['fdp']) > 1e-12 or (p is not None and abs(p['mean']-ref['power'])>1e-12):
                raise ValueError('Disagreement with frozen analysis')
            methods[method] = {'fdp': f, 'power': p,
                               'zero_discovery_fraction': float(no_discovery[:,col].mean()),
                               'fdp_gt_nominal_fraction': float((fdp[:,col]>.05).mean())}
        result['rows'].append({'case_index': case, 'case': index['settings']['cases'][case], 'n': n,
                               'methods': methods, 'cpu_seconds': distribution(cpu),
                               'family_wall_seconds': distribution(wall),
                               'guard_failed_folds': guard_failures, 'pattern_fallback_folds': pattern_fallbacks})
        result['whole_family_count'] += n
        result['total_recorded_cpu_seconds'] += float(cpu.sum())
        result['total_recorded_family_wall_seconds'] += float(wall.sum())
        write_json(out/'progress.json', {'complete_cases': len(result['rows']), 'complete_families': result['whole_family_count']})
        print('Descriptive whole-family distributions:', case, n, flush=True)
    result.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                  analysis_elapsed_seconds=time.monotonic()-started)
    write_json(out/'summary.json', result)
    print(out)


if __name__ == '__main__':
    main()
