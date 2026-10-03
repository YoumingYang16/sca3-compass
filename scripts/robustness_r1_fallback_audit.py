"""Audit all recorded fallback families after complete frozen confirmation.

No fitting, sampling, threshold adjustment, or additional statistical tests.
Only cases with a nonzero fallback count in the frozen full analyzer need a
second diagnostic read; equality of the recovered count is required.
"""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz')
          else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    base = ROOT/'artifacts/robustness'
    analysis_path = base/f'{args.run_id}-confirmation-analysis.json'
    analysis = read(analysis_path)
    index_path = base/f'{args.run_id}-results-index.json'
    if not analysis['complete'] or sha(index_path) != analysis['index_sha256']:
        raise ValueError('Completed hash-bound confirmation required')
    index = read(index_path)
    cases = {r['case_index']: r['fallback_folds'] for r in analysis['rows'] if r['fallback_folds']}
    found = {i: 0 for i in cases}
    receipts = []
    for entry in index['receipts']:
        if cooperative_stop():
            raise InterruptedError('Finite audit stop; no completed output')
        if entry['case_index'] not in cases:
            continue
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Record checksum mismatch')
        record = read(entry['path'])
        for fold in record['diagnostics_R1']['folds']:
            if not fold['fallback']:
                continue
            found[entry['case_index']] += 1
            receipts.append({'case_index': entry['case_index'], 'rep': entry['rep'],
                             'path': entry['path'], 'record_sha256': entry['sha256'],
                             'fold': fold['fold'], 'guard_success': fold['guard_success'],
                             'pattern_fallback': fold['pattern']['fallback'],
                             'pattern': fold['pattern'] if fold['pattern']['fallback'] else None,
                             'failed_bootstrap': [r for r in fold['bootstrap_receipts'] if r['status'] != 'completed']})
    if found != cases:
        raise ValueError('Raw fallback count differs from frozen analyzer')
    out = base/f'{args.run_id}-fallback-audit'
    out.mkdir(exist_ok=False)
    result = {'run_id': args.run_id, 'complete': True, 'analysis_sha256': sha(analysis_path),
              'source_sha256': sha(__file__), 'purpose': 'DESCRIPTIVE_FAILURE_AUDIT_NO_NEW_EXPERIMENT',
              'fallback_folds': sum(found.values()), 'guard_failed_folds': sum(not r['guard_success'] for r in receipts),
              'pattern_fallback_folds': sum(r['pattern_fallback'] for r in receipts), 'receipts': receipts}
    with (out/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    shutil.copy2(__file__, out/'generate_source.py')
    print({k: v for k, v in result.items() if k != 'receipts'})
    print(out)


if __name__ == '__main__':
    main()
