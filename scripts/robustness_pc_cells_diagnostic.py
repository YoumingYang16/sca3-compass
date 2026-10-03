"""Exclusive-output, bounded KNOWN-MODEL DEVELOPMENT diagnostic for PC cells.

No registry edits, raw clinical data, production integration, or certification.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

# Apply before NumPy/SciPy import: post-import threadpool_limits is too late
# to prevent OpenBLAS thread-buffer allocation on a busy shared workstation.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import scipy
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sca3_compass.robustness_pc_cells import (  # noqa: E402
    DEFAULT_CELLS, equicorrelation, evaluate_summary, fit_cell_lp,
    outer_null_bound, required_outer_radius, student_location_lipschitz,
    summarize_bank_many,
)

ARTIFACT_ROOT = ROOT / 'artifacts' / 'robustness' / 'pc-cells-known-model-dev'
TAUS = (.001, .003)
KINDS = ('projection', 'simes')
SEEDS = {'fit_noise': 772501, 'fit_alternative_noise': 772503,
         'adversarial_means': 772509, 'adversarial_noise': 772511,
         'selected_null_fresh_noise': 772517, 'alternative_fresh_noise': 772519}
SOURCE_PATHS = ('src/sca3_compass/robustness_pc_cells.py',
                'tests/test_robustness_pc_cells.py',
                'scripts/robustness_pc_cells_diagnostic.py',
                'docs/robustness_pc_cells.md')
ARRAY_KEYS = ('cell_counts', 'overlap_tau_counts', 'baseline_beta_counts',
              'baseline_tau_counts', 'inside_box_counts')


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, default=plain, allow_nan=False)


def write_npz(path, **arrays):
    with path.open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(str(array.dtype).encode() + repr(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def noise_bank(n, seed, output, name):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, 4)) @ np.linalg.cholesky(equicorrelation(.65)).T
    noise /= np.sqrt(rng.chisquare(25., size=(n, 1)) / 25.)
    write_npz(output / f'{name}-bank.npz', noise=noise)
    write_json(output / f'{name}-bank.json', {'n': n, 'seed': seed, 'sha256_array': array_digest(noise),
                                            'construction': 'Normal(0,Sigma)/sqrt(ChiSquare(25)/25)',
                                            'df': 25., 'rho': .65, 'shared_radial': True})
    return noise


def null_fit_design():
    free = (-4., -1., 0., .5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 8., 12.)
    nonpositive = (0., -.25, -1., -3.)
    rows = []
    for j in range(4):
        keep = np.delete(np.arange(4), j)
        for a, others in product(free, product(nonpositive, repeat=3)):
            mu = np.zeros(4)
            mu[j], mu[keep] = a, others
            rows.append(mu)
    means = np.unique(rows, axis=0)
    if np.any(np.sum(means > 0, axis=1) > 1):
        raise ArithmeticError('Null design changed the directional null')
    membership = np.array([np.all(means[:, np.arange(4) != j] <= 0, axis=1) for j in range(4)]).T
    return means, membership, {'free_coordinate_levels': free, 'other_coordinate_levels': nonpositive,
                               'all_faces_enumerated': True, 'unique_mean_count': len(means)}


def adversarial_design(n, seed, fit_means):
    """Fixed out-of-grid boundary sweep plus unequal negative interior means."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(4):
        keep = np.delete(np.arange(4), j)
        for a in np.linspace(.125, 15.875, 64):
            mu = np.zeros(4)
            mu[j] = a
            rows.append(mu)
        for _ in range(max(0, n // 4 - 64)):
            mu = np.zeros(4)
            mu[j] = rng.uniform(0, 8) if rng.uniform() < .75 else rng.uniform(-4, 16)
            negatives = -rng.exponential(size=3) * rng.choice([.15, .5, 1.5, 4.], size=3)
            negatives[rng.uniform(size=3) < .4] = 0
            mu[keep] = negatives
            rows.append(mu)
        for a in (-20., 20.):
            mu = np.zeros(4)
            mu[j] = a
            rows.append(mu)
    means = np.unique(rows, axis=0)
    seen = {tuple(row) for row in fit_means}
    means = np.array([row for row in means if tuple(row) not in seen])
    assert np.all(np.sum(means > 0, axis=1) <= 1)
    return means


def alternative_design():
    rows, labels = [], []
    for effect in (.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6.):
        rows.append([effect] * 4)
        labels.append(f'all4_{effect:g}')
    for support in (2, 3):
        for effect in (2.5, 3.5, 4.5):
            rows.append([effect] * support + [0.] * (4 - support))
            labels.append(f'{support}positive_{effect:g}')
    rows.extend([[1., 2., 3., 4.], [.5, 2.5, 2.5, 2.5]])
    labels.extend(['unequal_all4', 'one_weak_all4'])
    return np.array(rows), labels


def summarize_phase(noise, means, output, phase, batch=32):
    started = time.monotonic()
    pieces = {f'{kind}_{tau:g}': [] for kind, tau in product(KINDS, TAUS)}
    for lo in range(0, len(means), batch):
        piece = summarize_bank_many(noise, means[lo:lo+batch], TAUS, KINDS)
        for key, value in piece.items():
            pieces[key].append(value)
        if lo % (batch * 4) == 0 or lo + batch >= len(means):
            print(json.dumps({'phase': phase, 'done': min(lo+batch, len(means)),
                              'total': len(means), 'elapsed_s': round(time.monotonic()-started, 2)}), flush=True)
    summaries, arrays = {}, {'means': means, 'n': np.array(len(noise))}
    for key, parts in pieces.items():
        summaries[key] = {'n': len(noise)}
        for field in ARRAY_KEYS:
            value = np.concatenate([part[field] for part in parts])
            summaries[key][field] = value
            arrays[f'{key}__{field}'] = value
    write_npz(output / f'{phase}-counts.npz', **arrays)
    return summaries


def evaluate_models(summaries, models):
    return {name: evaluate_summary(summaries[item['summary_key']], item['weights'], item['tau'])
            for name, item in models.items()}


def peak_working_set():
    if sys.platform != 'win32':
        return None
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not ctypes.windll.psapi.GetProcessMemoryInfo(process, ctypes.byref(value), value.cb):
            return {'error': 'GetProcessMemoryInfo returned false'}
        return int(value.PeakWorkingSetSize)
    except Exception as error:
        return {'error': str(error)}


def verified_run(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ARTIFACT_ROOT.resolve()):
        raise ValueError('Read only scoped cell-development artifacts')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    for relative, info in manifest.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or digest(path) != info['sha256']:
            raise RuntimeError('Artifact checksum mismatch')
    return directory


def audit_frozen(args):
    """Additional DEVELOPMENT regression audit. No fitting or candidate changes."""
    started, cpu_started = time.monotonic(), time.process_time()
    parent = verified_run(args.audit_frozen)
    models = json.loads((parent / 'frozen-models.json').read_text(encoding='utf-8'))
    prior = json.loads((parent / 'summary.json').read_text(encoding='utf-8'))
    rows = [[0.] * 4, [-.25] * 4, [-1.] * 4, [-3.] * 4]
    for j in range(4):
        for a in (1.125, 1.375, 2., 3., 6., 12., 20.):
            mu = np.zeros(4)
            mu[j] = a
            rows.append(mu)
        rows.extend([np.roll([3., -.125, -.5, -2.], j), np.roll([0., -.1, -1., -3.], j)])
    rows.extend(item['worst_fresh_null_mean'] for item in prior['models'].values())
    means = np.unique(rows, axis=0)
    if np.any(np.sum(means > 0, axis=1) > 1):
        raise ArithmeticError('Sentinel design changed the directional null')
    name = args.run_name or ('D002-frozen-audit-' + uuid.uuid4().hex[:8])
    output = (ARTIFACT_ROOT / name).resolve()
    if not output.is_relative_to(ARTIFACT_ROOT.resolve()) or output == ARTIFACT_ROOT.resolve():
        raise ValueError('Output must be a new scoped child')
    output.mkdir(exist_ok=False)
    print(json.dumps({'output': str(output), 'parent': str(parent), 'status': 'FROZEN_CANDIDATE_DEVELOPMENT_AUDIT'}), flush=True)
    sources = {}
    source_dir = output / 'source'
    source_dir.mkdir()
    for relative in SOURCE_PATHS:
        data = (ROOT / relative).read_bytes()
        with (source_dir / relative.replace('/', '__')).open('xb') as stream:
            stream.write(data)
        sources[relative] = hashlib.sha256(data).hexdigest()
    protocol = {'status': 'FROZEN_CANDIDATE_DEVELOPMENT_AUDIT_NOT_CERTIFIED',
                'parent': str(parent), 'parent_manifest_sha256': digest(parent / 'manifest.json'),
                'parent_models_sha256': digest(parent / 'frozen-models.json'),
                'source_sha256': sources, 'seed': 772523, 'n': args.fresh_n, 'means': means,
                'design': 'Global null, nonpositive sentinels, single-positive boundary, unequal-negative nulls and prior observed failures; no refit.',
                'data_reuse_disclosure': 'Parameters include known failures; only Monte Carlo bank is new. Not main-project confirmation.',
                'continuous_null_certified': False}
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'frozen-models.json', models)
    try:
        bank = noise_bank(args.fresh_n, 772523, output, 'sentinel-fresh')
        summaries = summarize_phase(bank, means, output, 'sentinel-fresh', batch=8)
        results = evaluate_models(summaries, models)
        write_json(output / 'sentinel-fresh-metrics.json', results)
        zero = int(np.flatnonzero(np.all(means == 0, axis=1))[0])
        compact = {}
        for key, item in results.items():
            worst = int(np.argmax(item['Ephi_over_tau']))
            compact[key] = {'max_Ephi_over_tau': float(item['Ephi_over_tau'][worst]),
                            'mc_se_over_tau': float(item['mc_se'][worst] / models[key]['tau']),
                            'worst_mean': means[worst],
                            'global_null_Ephi_over_tau': float(item['Ephi_over_tau'][zero]),
                            'global_null_mc_se_over_tau': float(item['mc_se'][zero] / models[key]['tau']),
                            'worst_baseline_tau_ratio': float(item['baseline_tau'][worst] / models[key]['tau']),
                            'empirically_above_tau_count': int(np.sum(item['Ephi_over_tau'] > 1))}
        for relative, expected in sources.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f'Source changed during frozen audit: {relative}')
        write_json(output / 'summary.json', {'status': protocol['status'], 'models': compact,
                                            'resources': {'wall_seconds': time.monotonic()-started,
                                                          'process_cpu_seconds': time.process_time()-cpu_started,
                                                          'peak_working_set_bytes': peak_working_set()}})
        write_json(output / 'manifest.json', {str(path.relative_to(output)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                                             for path in output.rglob('*') if path.is_file()})
        print(json.dumps({'output': str(output), 'models': compact}, default=plain), flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'FAILED_FROZEN_AUDIT', 'error': str(error)})
        raise


def run(args):
    start, cpu_start = time.monotonic(), time.process_time()
    run_name = args.run_name or ('D001-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    output = (ARTIFACT_ROOT / run_name).resolve()
    if not output.is_relative_to(ARTIFACT_ROOT.resolve()) or output == ARTIFACT_ROOT.resolve():
        raise ValueError('Output must be a new child of the exclusive scoped artifact root')
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    print(json.dumps({'output': str(output), 'status': 'DEVELOPMENT_ONLY_NOT_CERTIFIED'}), flush=True)
    source_hashes = {}
    source_dir = output / 'source'
    source_dir.mkdir()
    for relative in SOURCE_PATHS:
        data = (ROOT / relative).read_bytes()
        destination = source_dir / relative.replace('/', '__')
        with destination.open('xb') as stream:
            stream.write(data)
        source_hashes[relative] = hashlib.sha256(data).hexdigest()
    fit_means, faces, grid_receipt = null_fit_design()
    fit_alt = np.array([[effect] * 4 for effect in (2., 2.5, 3., 3.5, 4.)])
    scan_means = adversarial_design(args.audit_vectors, SEEDS['adversarial_means'], fit_means)
    alt_means, alt_labels = alternative_design()
    protocol = {'status': 'DEVELOPMENT_ONLY_NOT_CERTIFIED', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'run_name': run_name, 'source_sha256': source_hashes, 'seeds': SEEDS,
                'model': {'df': 25., 'rho': .65, 'shape': equicorrelation(.65),
                          'known_nuisances': True, 'shared_radial': True, 'profile': [1, 1, 1, 1]},
                'taus': TAUS, 'beta_fraction': .9, 'cell_spec': DEFAULT_CELLS.receipt(),
                'grid': grid_receipt, 'null_fit_n': args.fit_n, 'alternative_fit_n': args.fit_alt_n,
                'out_of_grid_scan_n': args.audit_n, 'out_of_grid_scan_count': len(scan_means),
                'selected_fresh_n': args.fresh_n, 'selected_count_limit': args.selected,
                'alternative_fresh_n': args.alt_n, 'fit_alternative_means': fit_alt,
                'alternative_audit_means': alt_means, 'alternative_audit_labels': alt_labels,
                'accounting': ['strict_empirical', 'clipped_estimate', 'bound'],
                'clipping_disclosure': 'Only the MC baseline-probability estimate is capped by its known beta bound; it is NOT a confidence upper bound and does not change observations or the baseline decision.',
                'selection_rule': 'Freeze all LPs; rank unseen-null scan vectors by max candidate Ephi/tau; fresh independent bank at top selected vectors; no refit.',
                'certification': False, 'helpers_only_not_complete_cover': True,
                'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                                'platform': platform.platform(), 'threadpools': threadpool_info()}}
    write_json(output / 'protocol.json', protocol)
    write_npz(output / 'designs.npz', fit_means=fit_means, fit_faces=faces, fit_alternative=fit_alt,
              scan_means=scan_means, alternative_audit=alt_means)
    try:
        fit_noise = noise_bank(args.fit_n, SEEDS['fit_noise'], output, 'fit-null')
        fit_summaries = summarize_phase(fit_noise, fit_means, output, 'fit-null')
        del fit_noise
        alt_noise = noise_bank(args.fit_alt_n, SEEDS['fit_alternative_noise'], output, 'fit-alternative')
        alt_fit_summaries = summarize_phase(alt_noise, fit_alt, output, 'fit-alternative')
        del alt_noise
        models, fits = {}, {}
        for kind, tau in product(KINDS, TAUS):
            key = f'{kind}_{tau:g}'
            for accounting in ('strict_empirical', 'clipped_estimate', 'bound'):
                name = f'{key}_{accounting}'
                weights, receipt, constraint = fit_cell_lp(fit_summaries[key], alt_fit_summaries[key], tau,
                                                         baseline_accounting=accounting)
                fits[name] = receipt
                arrays = {k: v for k, v in constraint.items() if v is not None}
                if weights is not None:
                    arrays['weights'] = weights
                write_npz(output / f'constraints-{name}.npz', **arrays)
                write_json(output / f'fit-{name}.json', receipt)
                if receipt['success']:
                    models[name] = {'summary_key': key, 'tau': tau, 'weights': weights,
                                    'baseline': kind, 'accounting': accounting}
                print(json.dumps({'phase': 'fit', 'name': name, **receipt}, default=plain), flush=True)
        if not models:
            raise RuntimeError('Every finite-grid LP failed; raw failure records retained')
        write_json(output / 'frozen-models.json', models)
        del fit_summaries, alt_fit_summaries
        audit_noise = noise_bank(args.audit_n, SEEDS['adversarial_noise'], output, 'out-of-grid-scan')
        scanned = summarize_phase(audit_noise, scan_means, output, 'out-of-grid-scan')
        scan_results = evaluate_models(scanned, models)
        write_json(output / 'out-of-grid-scan-metrics.json', scan_results)
        scores = np.max(np.array([value['Ephi_over_tau'] for value in scan_results.values()]), axis=0)
        selected_ids = np.argsort(-scores, kind='stable')[:args.selected]
        selected_means = scan_means[selected_ids]
        write_json(output / 'selected-null-design.json', {'indices': selected_ids, 'means': selected_means,
                                                        'scan_selection_scores': scores[selected_ids],
                                                        'selected_before_fresh_noise': True})
        del audit_noise, scanned, scan_results
        fresh_noise = noise_bank(args.fresh_n, SEEDS['selected_null_fresh_noise'], output, 'selected-null-fresh')
        fresh = summarize_phase(fresh_noise, selected_means, output, 'selected-null-fresh', batch=8)
        fresh_results = evaluate_models(fresh, models)
        write_json(output / 'selected-null-fresh-metrics.json', fresh_results)
        del fresh_noise, fresh
        alt_noise = noise_bank(args.alt_n, SEEDS['alternative_fresh_noise'], output, 'alternative-fresh')
        alt_summary = summarize_phase(alt_noise, alt_means, output, 'alternative-fresh', batch=4)
        alt_results = evaluate_models(alt_summary, models)
        write_json(output / 'alternative-fresh-metrics.json', alt_results)
        del alt_noise
        main_alt_ids = np.array([alt_labels.index(f'all4_{x:g}') for x in (2., 2.5, 3., 3.5, 4.)])
        result = {'status': 'DEVELOPMENT_COMPLETE_NOT_CERTIFIED', 'continuous_null_certified': False,
                  'main_project_integrated': False, 'pipeline_heterogeneity_guarantee': False,
                  'models': {}, 'out_of_grid_selected_means': selected_means,
                  'alternative_labels': alt_labels, 'fit_receipts': fits,
                  'known_model_helpers': {str(tau): {'L_df25': student_location_lipschitz(25),
                                                    'required_outer_radius': required_outer_radius(tau, .9*tau, 25),
                                                    'outer_bound_at_radius': outer_null_bound(required_outer_radius(tau, .9*tau, 25), 25, beta=.9*tau)}
                                          for tau in TAUS}}
        for name, item in models.items():
            a, f = alt_results[name], fresh_results[name]
            worst = int(np.argmax(f['Ephi_over_tau']))
            result['models'][name] = {'mean_power_fit_profile_fresh': float(a['mean_phi'][main_alt_ids].mean()),
                                      'baseline_tau_power_fit_profile_fresh': float(a['baseline_tau'][main_alt_ids].mean()),
                                      'gain_pp_fit_profile_fresh': float(100 * a['gain_vs_tau'][main_alt_ids].mean()),
                                      'baseline_beta_cost_pp': float(100 * a['baseline_reduction_cost'][main_alt_ids].mean()),
                                      'added_cell_mass_pp': float(100 * a['added_cell_mass'][main_alt_ids].mean()),
                                      'all_alternative_gains_pp': 100 * a['gain_vs_tau'],
                                      'worst_fresh_null_Ephi_over_tau': float(f['Ephi_over_tau'][worst]),
                                      'worst_fresh_null_mc_se_over_tau': float(f['mc_se'][worst] / item['tau']),
                                      'worst_fresh_null_mean': selected_means[worst],
                                      'worst_fresh_null_baseline_tau_ratio': float(f['baseline_tau'][worst] / item['tau']),
                                      'worst_fresh_null_added_mass': float(f['added_cell_mass'][worst]),
                                      'fresh_null_vectors_above_nominal_empirically': int(np.sum(f['Ephi_over_tau'] > 1)),
                                      'no_uniform_claim': True}
        for relative, expected in source_hashes.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f'Source changed during run: {relative}; do not treat this run as frozen')
        result['resources'] = {'wall_seconds': time.monotonic()-start, 'process_cpu_seconds': time.process_time()-cpu_start,
                               'peak_working_set_bytes': peak_working_set(), 'blas_threads': 1, 'highs_threads': 1}
        write_json(output / 'summary.json', result)
        write_json(output / 'manifest.json', {str(path.relative_to(output)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                                             for path in output.rglob('*') if path.is_file()})
        print(json.dumps({'output': str(output), 'summary': result}, default=plain, allow_nan=False), flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'FAILED_DEVELOPMENT_RUN', 'exception': type(error).__name__,
                                            'message': str(error), 'elapsed_seconds': time.monotonic()-start})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--fit-n', type=int, default=32768)
    parser.add_argument('--fit-alt-n', type=int, default=131072)
    parser.add_argument('--audit-n', type=int, default=65536)
    parser.add_argument('--audit-vectors', type=int, default=1024)
    parser.add_argument('--fresh-n', type=int, default=524288)
    parser.add_argument('--selected', type=int, default=64)
    parser.add_argument('--alt-n', type=int, default=262144)
    parser.add_argument('--read-run', type=Path)
    parser.add_argument('--audit-frozen', type=Path)
    args = parser.parse_args()
    if args.read_run is not None:
        directory = verified_run(args.read_run)
        print((directory / 'summary.json').read_text(encoding='utf-8'))
        return
    if min(args.fit_n, args.fit_alt_n, args.audit_n, args.fresh_n, args.alt_n, args.selected) < 1 or args.audit_vectors < 256:
        raise ValueError('Positive Monte Carlo sizes and at least 256 adversarial means required')
    with threadpool_limits(limits=1):
        if args.audit_frozen is not None:
            audit_frozen(args)
        else:
            run(args)


if __name__ == '__main__':
    main()
