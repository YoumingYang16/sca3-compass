"""Post-reboot integrity audit; no metrics, fits, thresholds or new samples.

Completed records must match the original run/seed/protocol and all array
hashes. Orphan inputs/evidence are inventoried for recoverable quarantine.
This script never moves/deletes a file and never inspects Power/FDP values.
"""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime, timezone
import argparse, gzip, hashlib, json, re, shutil, time

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode('utf-8')).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    folder = ROOT/'artifacts/robustness'
    protocol_path = folder/f'{args.run_id}-screen.protocol.json'
    protocol = read(protocol_path)
    if protocol['phase'] != 'C1' or protocol['run_id'] != args.run_id:
        raise ValueError('Recovery of the original C1 only')
    source = folder/f'{args.run_id}-source'
    for rel, value in protocol['source_sha256'].items():
        if sha(source/rel) != value:
            raise ValueError('Frozen source damaged: ' + rel)
    out = args.out.resolve()
    if not out.is_relative_to((folder/'R1R2-20260916').resolve()):
        raise ValueError('Audit output must stay in this research archive')
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, out/'audit_source.py')
    repetitions = (folder/f'{args.run_id}-repetitions').resolve()
    paths = sorted(repetitions.rglob('*.json.gz'))
    expected_digest = digest(protocol)
    verified = []; issues = []; scientific_failures = []; stems = set(); start = time.perf_counter()
    examined = 0
    for path in paths:
        if cooperative_stop():
            break
        relative = path.relative_to(repetitions).as_posix()
        match = re.fullmatch(r'case-(\d{4})/rep-(\d{6})\.json\.gz', relative)
        if not match:
            issues.append({'path': str(path), 'issue': 'unexpected record path'}); examined += 1; continue
        case, rep = map(int, match.groups())
        stem = path.with_name(path.name.removesuffix('.json.gz'))
        stems.add(str(stem))
        try:
            r = read(path)  # gzip CRC/JSON integrity, but never select/read scientific metrics
            if r.get('status') != 'completed':
                scientific_failures.append({'path': str(path), 'status': r.get('status'),
                                            'instruction': 'Preserve; do not automatically rerun/drop a scientific failure'})
            else:
                if not 0 <= case < len(protocol['cases']) or not 0 <= rep < protocol['repetition_counts'][case]:
                    raise ValueError('Record outside frozen sample grid')
                identities = {'run_id': args.run_id, 'phase': 'C1', 'method_version': protocol['method_version'],
                              'case_index': case, 'rep': rep, 'protocol_digest': expected_digest,
                              'seed_sequence': [protocol['seed'], case, rep]}
                if any(r.get(key) != value for key, value in identities.items()):
                    raise ValueError('Record identity mismatch')
                for kind in ['input', 'evidence']:
                    expected_path = stem.with_name(stem.name + '-' + kind + '.npz')
                    actual_path = Path(r[kind+'_path']).resolve()
                    if actual_path != expected_path or not actual_path.is_relative_to(repetitions):
                        raise ValueError('Array path outside the matching frozen family')
                    if sha(actual_path) != r[kind+'_sha256']:
                        raise ValueError('Array hash mismatch: ' + kind)
                verified.append({'case_index': case, 'rep': rep, 'path': str(path), 'sha256': sha(path),
                                 'input_sha256': r['input_sha256'], 'evidence_sha256': r['evidence_sha256'],
                                 'status': 'IDENTITY_AND_ARRAY_HASHES_VERIFIED_NOT_METRICS'})
        except (OSError, ValueError, EOFError, KeyError) as error:
            issues.append({'path': str(path), 'issue': repr(error)})
        examined += 1
        if examined % 500 == 0:
            progress = {'examined': examined, 'record_files': len(paths), 'verified': len(verified),
                        'issues': len(issues), 'scientific_failures': len(scientific_failures),
                        'updated_utc': datetime.now(timezone.utc).isoformat()}
            (out/'progress.json').write_text(json.dumps(progress, indent=2), encoding='utf-8')
            print(json.dumps(progress), flush=True)
    quarantine = []
    for path in sorted(repetitions.rglob('*')):
        if not path.is_file():
            continue
        if path.name.endswith('.tmp'):
            reason = 'interrupted atomic write temporary'
        elif path.name.endswith('-input.npz') or path.name.endswith('-evidence.npz'):
            suffix = '-input.npz' if path.name.endswith('-input.npz') else '-evidence.npz'
            stem = path.with_name(path.name.removesuffix(suffix))
            if str(stem) in stems:
                continue
            reason = 'array without a completed record filename; preserve and regenerate same frozen seed'
        else:
            continue
        quarantine.append({'source': str(path), 'relative_path': path.relative_to(repetitions).as_posix(),
                           'bytes': path.stat().st_size, 'sha256': sha(path), 'reason': reason})
    complete = examined == len(paths)
    result = {'run_id': args.run_id, 'complete_audit': complete,
              'safe_after_quarantining_listed_orphans': complete and not issues and not scientific_failures,
              'frozen_protocol_sha256': sha(protocol_path), 'frozen_protocol_digest': expected_digest,
              'record_files': len(paths), 'verified_existing_families': len(verified),
              'planned_families': sum(protocol['repetition_counts']),
              'remaining_to_generate': sum(protocol['repetition_counts']) - len(verified),
              'issues': issues, 'scientific_failures': scientific_failures,
              'quarantine_plan': quarantine, 'elapsed_seconds': time.perf_counter()-start,
              'script_sha256': sha(__file__), 'provenance': 'INTEGRITY_ONLY_NOT_INDEPENDENT_CONFIRMATION',
              'scientific_metrics_inspected': False,
              'record_scope': 'Original C1; preserve fixed counts, source, input seeds, algorithm seeds and analysis. No reselection.',
              'finished_utc': datetime.now(timezone.utc).isoformat()}
    (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with gzip.open(out/'verified-receipts.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(verified, stream)
    print(json.dumps({key: result[key] for key in ['complete_audit', 'safe_after_quarantining_listed_orphans', 'verified_existing_families', 'remaining_to_generate', 'elapsed_seconds']}, indent=2))
    if not result['safe_after_quarantining_listed_orphans']:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
