"""Read-only scientific progress extraction; runs NO simulation or fitting.

Creates one new receipt. Registry, source archives and historical results stay
unchanged. Complete aggregates are checked against registered file hashes;
reported averages are recomputed from all repetition arrays. Partial raw case
digests are verified. This is a status audit, not independent confirmation.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import robustness_replay as replay
from sca3_compass.robustness_analysis import diagnostic_counts


def read(path):
    return replay.read_json(path)


def compact_rows(result):
    rows = []
    for row in result['rows']:
        for key, array in [('fdr', 'fdp_by_repetition'), ('power', 'power_by_repetition')]:
            values = np.asarray(row[array], float)
            if len(values) != result['repetitions'] or not np.isfinite(values).all():
                raise ValueError('Invalid repetition array')
            if np.any((values < 0) | (values > 1)):
                raise ValueError('Out-of-range bounded metric')
            if not np.isclose(values.mean(), row[key]['mean'], rtol=0, atol=2e-14):
                raise ValueError('Metric mean disagrees with raw repetition array')
        rows.append({k: row[k] for k in ['method', 'fdp_q95', 'fdp_q99']} |
                    {'power': row['power']['mean'], 'fdr': row['fdr']['mean']})
    return {'case': result['case'], 'repetitions': result['repetitions'],
            'power_defined': result['power_defined'], 'rows': rows,
            'elapsed_seconds': result['elapsed_seconds']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior checkpoints')
    folder = ROOT / 'artifacts/robustness'
    registry = read(folder / 'EXPERIMENT_REGISTRY.json')
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'phase': 'PROGRESS_AUDIT_NOT_EXPERIMENT_OR_CONFIRMATION',
              'registry_sha256_at_read': replay.sha(folder / 'EXPERIMENT_REGISTRY.json'),
              'registry_status_counts': dict(Counter(e['status'] for e in registry['experiments'])),
              'registry_entries': [{k: e.get(k) for k in ['id', 'status', 'started_at', 'completed_at',
                  'elapsed_seconds', 'failure_type', 'failure']} for e in registry['experiments']],
              'runs': {}, 'earlier_analysis_checks': {}, 'pc_weight_source_checks': {}}
    for run in ['R0067', 'R0068', 'R0069']:
        entry = next(e for e in registry['experiments'] if e['id'] == run)
        protocol = read(folder / f'{run}-screen.protocol.json')
        if protocol != entry['settings']:
            raise ValueError('Registry/protocol mismatch')
        source_audit, _ = replay.audit_source(folder / f'{run}-source', protocol)
        partial = read(folder / f'{run}-screen.partial.json')
        item = {'status': entry['status'], 'source_audit': source_audit,
                'planned_cases': len(protocol['cases']), 'repetitions_per_case': protocol['repetitions'],
                'completed_indices': partial['completed_case_indices'], 'scenarios': []}
        if entry['status'] == 'completed':
            path = folder / f'{run}-screen.json'
            digest = replay.sha(path)
            if digest != entry['result_sha256']:
                raise ValueError('Registered aggregate checksum mismatch')
            artifact = read(path)
            if artifact['settings'] != protocol:
                raise ValueError('Aggregate protocol mismatch')
            item['aggregate_sha256_verified'] = digest
            item['elapsed_seconds'] = artifact['elapsed_seconds']
            item['scenarios'] = [compact_rows(x) for x in artifact['scenarios']]
            item['diagnostics'] = diagnostic_counts([d for x in artifact['scenarios'] for d in x['diagnostics']])
            item['verification_scope'] = 'Registered aggregate hash, archived sources, every metric mean/array; compact diagnostic event trees. Full raw diagnostic replay not performed by this status audit.'
        else:
            path = folder / f'{run}-screen.partial.json'
            for index in partial['completed_case_indices']:
                raw_path = folder / f'{run}-cases/case-{index:04}.json.gz'
                raw = read(raw_path)
                if (raw['settings_digest'] != replay.content_digest(protocol)
                        or raw['result_digest'] != replay.content_digest(raw['result'])
                        or raw['result']['case'] != protocol['cases'][index]):
                    raise ValueError('Partial raw checkpoint mismatch')
                item['scenarios'].append(compact_rows(raw['result']))
            item['verification_scope'] = 'Every presently completed raw case checksum/protocol and metric mean/array; incomplete cases excluded.'
        item['last_result_write_utc'] = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
        item['method_configuration_label_count'] = len(item['scenarios'][0]['rows'])
        result['runs'][run] = item
        print(run, item['status'], len(item['scenarios']), '/', item['planned_cases'], flush=True)
    for name in ['R0062-support-gate-checkpoint-analysis.json', 'R0064-free-cap-checkpoint-analysis.json']:
        analysis = read(folder / name)
        receipts = analysis['checkpoint_analysis_receipt']['checkpoint_receipts']
        for receipt in receipts:
            if replay.sha(receipt['path']) != receipt['file_sha256']:
                raise ValueError('Previously analyzed raw file changed')
        result['earlier_analysis_checks'][name] = {'analysis_sha256': replay.sha(folder / name),
                                                   'raw_file_hashes_reverified': len(receipts)}
    weight = read(folder / 'pc-weight-development/D002-final-source-replay/result.json')
    for relative, digest in weight['source_sha256'].items():
        if replay.sha(ROOT / relative) != digest:
            raise ValueError('PC-weight delivered code differs from its receipt')
        result['pc_weight_source_checks'][relative] = digest
    label_sets = [{r['method'] for r in x['scenarios'][0]['rows']} for x in result['runs'].values()]
    result['current_three_runs_unique_configuration_labels'] = len(set.union(*label_sets))
    result['independent_confirmation_completed'] = False
    result['source_sha256'] = replay.sha(__file__)
    replay.write_new(args.output, result)
    print('receipt', args.output, flush=True)


if __name__ == '__main__':
    main()
