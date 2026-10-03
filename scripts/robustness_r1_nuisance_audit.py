"""Descriptive audit of existing DEV nuisance banks; no refit/new data/C1 read."""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
import argparse, gzip, hashlib, json, shutil, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with (gzip.open(path, 'rt', encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='R0072')
    args = parser.parse_args()
    path = ROOT/'artifacts/robustness'/f'{args.run_id}-results-index.json'
    index = read(path)
    if not index['complete'] or index['settings']['phase'] != 'development':
        raise ValueError('Complete DEVELOPMENT evidence only; do not open confirmation records')
    start = time.perf_counter()
    rows = {}
    completed = 0
    for receipt in index['receipts']:
        if cooperative_stop():
            break
        if sha(receipt['path']) != receipt['sha256']:
            raise ValueError('Record checksum mismatch')
        r = read(receipt['path'])
        if r['status'] != 'completed':
            raise ValueError('Do not omit failed families')
        for field in ['input', 'evidence']:
            if sha(r[field+'_path']) != r[field+'_sha256']:
                raise ValueError('Array checksum mismatch')
        case = r['case_index']
        row = rows.setdefault(case, {'case_index': case, 'case': index['settings']['cases'][case], 'families': []})
        with np.load(r['input_path'], allow_pickle=False) as archive:
            truth = archive['truth'].astype(bool)
        with np.load(r['evidence_path'], allow_pickle=False) as archive:
            plugin = archive['p_plugin_projection']
            guard = archive['p_guard_projection']
        if np.any(guard < plugin) or not np.isfinite([plugin, guard]).all():
            raise ValueError('Saved guard dominance/bounded evidence failed')
        groups = {}
        for label, mask in [('true_signed', truth), ('null_signed', ~truth)]:
            use = mask & (plugin > 0) & (plugin < 1)
            groups[label] = {
                'claims': int(mask.sum()), 'strict_interior_plugin_claims': int(use.sum()),
                'zero_plugin_claims': int((mask & (plugin == 0)).sum()),
                'one_plugin_claims': int((mask & (plugin == 1)).sum()),
                'mean_log10_guard_over_plugin_interior': float(np.mean(np.log10(guard[use])-np.log10(plugin[use]))) if use.any() else None,
                'rate_guard_exceeds_plugin': float((guard[mask] > plugin[mask]).mean()) if mask.any() else None,
                'null_tail_001': float((guard[mask] <= .001).mean()) if label == 'null_signed' and mask.any() else None,
            }
        folds = []
        for fold in r['diagnostics_R1']['folds']:
            bank = fold['bank']; base = bank[0]; gaussian = [b['prior']['gaussian_bic_selected'] for b in bank]
            rho = np.array([b['rho'] for b in bank]); scatter = np.array([b['prior']['scatter'] for b in bank])
            folds.append({'rho_original': float(rho[0]), 'rho_range': float(rho.max()-rho.min()),
                          'scatter_max_over_original': float(scatter.max()/scatter[0]),
                          'scatter_min_over_original': float(scatter.min()/scatter[0]),
                          'mixed_gaussian_student_bank': len(set(gaussian)) > 1,
                          'original_gaussian': bool(gaussian[0]), 'bank_entries': len(bank),
                          'mean_gamma': float(np.mean(fold['gamma'])),
                          'zero_gamma_count': int(np.sum(np.asarray(fold['gamma']) == 0)),
                          'pilot_projection_plugin': fold['pilot']['plugin_projection']['pilot_discoveries'],
                          'pilot_projection_guard': fold['pilot']['guard_projection']['pilot_discoveries'],
                          'fallback': bool(fold['fallback'])})
        row['families'].append({'rep': r['rep'], 'groups': groups, 'folds': folds})
        completed += 1
        if completed % 500 == 0:
            print('DEV audit families', completed, flush=True)
    summaries = []
    for case, row in sorted(rows.items()):
        families = row['families']; folds = [f for r in families for f in r['folds']]
        s = {'case_index': case, 'case': row['case'], 'n': len(families),
             'rho_range_mean': float(np.mean([f['rho_range'] for f in folds])),
             'mixed_model_bank_rate': float(np.mean([f['mixed_gaussian_student_bank'] for f in folds])),
             'mean_scatter_max_ratio': float(np.mean([f['scatter_max_over_original'] for f in folds])),
             'gamma_mean': float(np.mean([f['mean_gamma'] for f in folds])),
             'pilot_plugin_mean': float(np.mean([f['pilot_projection_plugin'] for f in folds])),
             'pilot_guard_mean': float(np.mean([f['pilot_projection_guard'] for f in folds])),
             'fallback_folds': sum(f['fallback'] for f in folds), 'groups': {}}
        for label in ['true_signed', 'null_signed']:
            values = [r['groups'][label]['mean_log10_guard_over_plugin_interior'] for r in families]
            defined = [v for v in values if v is not None]
            s['groups'][label] = {'defined_families': len(defined),
                                  'mean_family_log_ratio': float(np.mean(defined)) if defined else None,
                                  'zero_plugin_claims_total': sum(r['groups'][label]['zero_plugin_claims'] for r in families),
                                  'interior_plugin_claims_total': sum(r['groups'][label]['strict_interior_plugin_claims'] for r in families)}
        summaries.append(s)
    out = ROOT/'artifacts/robustness'/f'{args.run_id}-nuisance-audit'
    out.mkdir(exist_ok=False)
    result = {'run_id': args.run_id, 'purpose': 'EXISTING_DEV_MECHANISM_AUDIT_NOT_CONFIRMATION',
              'complete': completed == len(index['receipts']), 'whole_families': completed,
              'index_sha256': sha(path), 'script_sha256': sha(__file__),
              'elapsed_seconds': time.perf_counter()-start, 'rows': summaries,
              'warning': 'Family averages are descriptive. Interior-p log ratios explicitly exclude reported zero/one endpoints. Joint nuisance ranges and model switches do not isolate causal parameter contributions. No null-claim independence or finite-sample bootstrap validity inferred.'}
    (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    with gzip.open(out/'family-diagnostics.json.gz', 'wt', encoding='utf-8') as stream:
        json.dump(rows, stream)
    shutil.copy2(__file__, out/'audit_source.py')
    print(json.dumps({k:result[k] for k in ['complete','whole_families','elapsed_seconds']}, indent=2))
    print(out)

if __name__ == '__main__':
    main()
