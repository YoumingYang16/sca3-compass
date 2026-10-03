"""Exclusive, bounded known-model diagnostic. No registry or cohort access."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

for variable in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sca3_compass.robustness_pc_cells import DEFAULT_CELLS, evaluate_summary  # noqa: E402
from sca3_compass.robustness_pc_exchange import (  # noqa: E402
    ARRAY_FIELDS, BETA_FRACTIONS, KINDS, PAIRS, TAUS, alternative_design,
    audit_frozen, audit_sentinels, calibrate_once, exchange_fit, face_membership,
    noise_batches, null_design, power_contrast, summarize_batches,
)

ARTIFACT_ROOT = ROOT / 'artifacts/robustness/pc-exchange-development'
SOURCES = ('src/sca3_compass/robustness_pc_exchange.py',
           'tests/test_robustness_pc_exchange.py',
           'scripts/robustness_pc_exchange_diagnostic.py',
           'docs/robustness_pc_exchange.md',
           'src/sca3_compass/robustness_pc_cells.py',
           'docs/robustness_pc_cells.md', 'docs/robustness_conditional_pc_route.md')
SIZES = {'train': 65536, 'train_alternative': 131072, 'development_scan': 65536,
         'calibration': 524288, 'untouched_audit': 1048576, 'untouched_power': 524288}
SEEDS = dict(zip(SIZES, (99172001, 99172003, 99172007, 99172009, 99172013, 99172017)))
STATUS = 'DEVELOPMENT_FINITE_ROW_MC_BOUNDS_NOT_CONTINUOUS_NULL_CERTIFIED'


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
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def peak_working_set():
    if sys.platform != 'win32':
        return None
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
        raise RuntimeError('Cannot measure process memory')
    return int(value.PeakWorkingSetSize)


def save_counts(output, phase, means, summaries):
    arrays = {'means': means, 'n': np.array(next(iter(summaries.values()))['n'])}
    for key, summary in summaries.items():
        arrays.update({f'{key}__{field}': summary[field] for field in ARRAY_FIELDS})
    write_npz(output / f'{phase}-counts.npz', **arrays)


def load_counts(directory, phase):
    with np.load(directory / f'{phase}-counts.npz', allow_pickle=False) as arrays:
        summaries = {key: {'n': int(arrays['n']), **{field: arrays[f'{key}__{field}']
                                                  for field in ARRAY_FIELDS}}
                     for key, _, _, _ in PAIRS}
        return arrays['means'], summaries


def run_phase(output, phase, means, sizes, seeds):
    started, last_print = time.monotonic(), [0.]
    print(json.dumps({'phase': phase, 'rows': len(means), 'n': sizes[phase]}), flush=True)
    bank_dir = output / f'{phase}-bank'
    bank_dir.mkdir()
    chunks = []

    def saved_batches():
        for i, noise in enumerate(noise_batches(sizes[phase], seeds[phase])):
            path = bank_dir / f'chunk-{i:04d}.npy'
            with path.open('xb') as stream:
                np.save(stream, noise, allow_pickle=False)
            chunks.append({'file': path.name, 'n': len(noise), 'sha256': digest(path)})
            yield noise

    def progress(done):
        elapsed = time.monotonic() - started
        if elapsed - last_print[0] >= 15 or done == sizes[phase]:
            print(json.dumps({'phase': phase, 'draws_done': done, 'elapsed_s': round(elapsed, 2)}), flush=True)
            last_print[0] = elapsed

    summaries = summarize_batches(saved_batches(), means, progress)
    write_json(bank_dir / 'receipt.json', {'n': sizes[phase], 'seed': seeds[phase], 'batch_size': 16384,
                                          'generator': 'numpy.PCG64, SeedSequence.spawn(2): normal/radial',
                                          'model': 'shared-radial t25, unit-diagonal shape equicorrelation .65',
                                          'independent_phase_bank': True, 'chunks': chunks})
    save_counts(output, phase, means, summaries)
    return summaries


def select_adverse(pool, summaries_by_phase, weights):
    """Same union for every model: four largest additions/model/phase + zero."""
    indices = set(np.flatnonzero(np.all(pool == 0, axis=1)).tolist())
    receipts = {}
    for phase, summaries in summaries_by_phase.items():
        receipts[phase] = {}
        for key, _, _, _ in PAIRS:
            added = summaries[key]['cell_counts'] @ weights[key] / summaries[key]['n']
            selected = np.argsort(-added, kind='stable')[:4]
            indices.update(selected.tolist())
            receipts[phase][key] = selected
    indices = np.array(sorted(indices), dtype=int)
    return pool[indices], {'pool_indices': indices, 'ranked_indices_by_phase_model': receipts,
                         'selection_uses_no_calibration_or_audit_draws': True}


def select_beta(train_alt, calibrated):
    """Prespecified TRAIN mean power AFTER calibration shrink; ties prefer .9."""
    selected, utility = {}, {}
    for kind in KINDS:
        for tau in TAUS:
            candidates = []
            for fraction in BETA_FRACTIONS:
                key = f'{kind}_{tau:g}_b{fraction:g}'
                value = float(evaluate_summary(train_alt[key], calibrated[key]['weights'], tau)['mean_phi'].mean())
                utility[key] = value
                candidates.append((value, fraction, key))
            selected[f'{kind}_{tau:g}'] = max(candidates)[2]
    return {'selected': selected, 'train_utility': utility,
            'rule': 'largest equal-mixture TRAIN alternative mean power after one calibration shrink; ties prefer larger beta',
            'calibration_is_development': True, 'untouched_audit_or_power_used': False}


def power_results(summaries, weights, labels):
    # Four tails, all 12 candidates, 19 means and two comparator types.
    alpha = .01 / (4 * len(PAIRS) * len(labels) * 2)
    return {key: {comparison: power_contrast(summaries[key], weights[key], tau,
                                           comparison=comparison, alpha_per_tail=alpha)
                  for comparison in ('tau', 'simple')}
            for key, _, tau, _ in PAIRS}


def compact_results(audit, power, calibrated, labels):
    macro = [labels.index(f'all4_{a:g}') for a in (2., 2.5, 3., 3.5, 4.)]
    support2 = [i for i, label in enumerate(labels) if label.startswith('support2_')]
    result = {}
    for key, _, _, fraction in PAIRS:
        row, p, simple = audit[key], power[key]['tau'], power[key]['simple']
        result[key] = {'beta_fraction': fraction, 'shrink': calibrated[key]['shrink'],
                       'max_null_empirical_Ephi_over_tau': float(np.max(row['empirical']['Ephi_over_tau'])),
                       'max_null_99pct_family_upper_over_tau': float(np.max(row['total_upper_over_tau'])),
                       'null_rows_upper_above_tau': row['rows_upper_above_tau'],
                       'macro_same_tau_gain_pp': 100 * float(np.mean(np.asarray(p['gain'])[macro])),
                       'macro_simple_enhanced_gain_pp': 100 * float(np.mean(np.asarray(simple['gain'])[macro])),
                       'positive_gain_profile_count': int(np.sum(np.asarray(p['gain']) > 0)),
                       'simultaneously_positive_profile_labels': [labels[i] for i in np.flatnonzero(np.asarray(p['simultaneous_gain_lower']) > 0)],
                       'simultaneously_negative_profile_labels': [labels[i] for i in np.flatnonzero(np.asarray(p['simultaneous_gain_upper']) < 0)],
                       'support2_same_tau_gain_pp': (100 * np.asarray(p['gain'])[support2]),
                       'support2_loss_upper_pp': (100 * np.asarray(p['simultaneous_gain_upper'])[support2]),
                       'all_profile_same_tau_gain_pp': 100 * np.asarray(p['gain'])}
    return result


def verified_run(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to(ARTIFACT_ROOT.resolve()) or directory == ARTIFACT_ROOT.resolve():
        raise ValueError('Read only exclusive PC exchange children')
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    actual = {str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(manifest):
        raise RuntimeError('Manifest inventory mismatch')
    for relative, entry in manifest.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or digest(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise RuntimeError(f'Artifact checksum mismatch: {relative}')
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    for relative, expected in protocol['source_sha256'].items():
        if digest(directory / 'source' / relative.replace('/', '__')) != expected:
            raise RuntimeError('Frozen source receipt mismatch')
    frozen = json.loads((directory / 'frozen-candidates.json').read_text(encoding='utf-8'))
    weights = {key: np.asarray(item['weights']) for key, item in frozen['calibrated'].items()}
    _, summaries = load_counts(directory, 'untouched_audit')
    audit = audit_frozen(summaries, weights)
    del summaries
    _, summaries = load_counts(directory, 'untouched_power')
    power = power_results(summaries, weights, protocol['alternative_labels'])
    compact = compact_results(audit, power, frozen['calibrated'], protocol['alternative_labels'])
    saved = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if json.loads(json.dumps(compact, default=plain)) != saved['models']:
        raise RuntimeError('Count-based summary reproduction mismatch')
    print(json.dumps({'verified_files': len(manifest), 'counts_reproduced': True,
                      'output': str(directory), 'selected_models': saved['selection']['selected'],
                      'resources': saved['resources']}, default=plain), flush=True)
    return saved


def run(args):
    start, cpu_start = time.monotonic(), time.process_time()
    name = args.run_name or ('E001-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('run-name must name one new direct child')
    output = (ARTIFACT_ROOT / name).resolve()
    if output.parent != ARTIFACT_ROOT.resolve():
        raise ValueError('Exclusive direct child required')
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    print(json.dumps({'output': str(output), 'status': STATUS}), flush=True)
    try:
        (output / 'source').mkdir()
        hashes = {}
        for relative in SOURCES:
            data = (ROOT / relative).read_bytes()
            with (output / 'source' / relative.replace('/', '__')).open('xb') as stream:
                stream.write(data)
            hashes[relative] = hashlib.sha256(data).hexdigest()
        sizes = {k: 256 for k in SIZES} if args.smoke else SIZES.copy()
        seeds = {k: v + (1000000 if args.smoke else 0) for k, v in SEEDS.items()}
        pool, initial, faces = null_design()
        sentinels = audit_sentinels()
        alt, labels = alternative_design()
        train_alt_means = np.array([[a] * 4 for a in (2., 2.5, 3., 3.5, 4.)])
        protocol = {'status': STATUS, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'source_sha256': hashes, 'command': [sys.executable, *sys.argv],
                    'smoke_only': args.smoke, 'sizes': sizes, 'seeds': seeds, 'batch_size': 16384,
                    'model': {'df': 25, 'rho': .65, 'known': True, 'shared_radial': True},
                    'cells': DEFAULT_CELLS.receipt(), 'beta_fractions': BETA_FRACTIONS,
                    'taus': TAUS, 'kinds': KINDS, 'initial_rows': len(initial), 'train_pool_rows': len(pool),
                    'null_boundary_spacing': .0625, 'all_four_faces': True, 'unequal_negatives': True,
                    'exchange': {'max_rounds': 6, 'add_per_pair': 24, 'shared_active_row_union': True},
                    'calibration': {'family_alpha': .01, 'guard': .8, 'single_shrink': True,
                                    'selection': 'union top four added masses per model per TRAIN/development-scan bank plus zero',
                                    'becomes_development': True},
                    'audit': {'family_alpha': .01, 'frozen_scalar_Hoeffding_KL_bounds': True,
                              'rows': 'calibration rows plus fixed offset/negative/tail sentinels',
                              'retuning_after_audit': False},
                    'power': {'family_alpha': .01, 'paired_signed_component_KL_bounds': True,
                              'beta_selection': 'TRAIN alternative utility after calibration; ties prefer larger beta'},
                    'alternative_means': alt, 'alternative_labels': labels,
                    'simple_enhancement': 'own baseline at beta OR other baseline at tau-beta; fixed union bound',
                    'continuous_null_coverage': False, 'unconditional_evalidity_claim': False,
                    'environment': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__,
                                    'platform': platform.platform(), 'threadpools': threadpool_info()}}
        write_json(output / 'protocol.json', protocol)
        write_npz(output / 'designs.npz', train_pool=pool, initial_indices=initial, faces=faces,
                  audit_sentinels=sentinels, train_alternatives=train_alt_means, power_alternatives=alt)
        train = run_phase(output, 'train', pool, sizes, seeds)
        train_alt = run_phase(output, 'train_alternative', train_alt_means, sizes, seeds)
        rounds = exchange_fit(train, train_alt, initial)
        for record in rounds:
            iteration = record['iteration']
            directory = output / f'exchange-{iteration:02d}'
            directory.mkdir()
            write_json(directory / 'rows.json', {k: v for k, v in record.items() if k != 'fits'})
            for key, item in record['fits'].items():
                write_json(directory / f'{key}-receipt.json', {k: v for k, v in item.items() if k != 'arrays'})
                write_npz(directory / f'{key}-constraints.npz', **{k: v for k, v in item['arrays'].items() if v is not None})
        if any(not item['receipt']['success'] for item in rounds[-1]['fits'].values()):
            raise RuntimeError('Recorded TRAIN LP solver failure')
        initial_weights = {key: rounds[0]['fits'][key]['weights'] for key, _, _, _ in PAIRS}
        weights = {key: rounds[-1]['fits'][key]['weights'] for key, _, _, _ in PAIRS}
        exchange_trace = [{key: {'max_added_budget_ratio': float(np.max(record['fits'][key]['pool_added_budget_ratio'])),
                                 'pool_violation_count': record['fits'][key]['pool_violation_count'],
                                 'alternative_added_mass': record['fits'][key]['receipt']['alternative_added_mass']}
                           for key, _, _, _ in PAIRS} for record in rounds]
        write_json(output / 'frozen-before-calibration.json', {'weights': weights, 'initial_weights': initial_weights,
                                                             'train_exchange_trace': exchange_trace})
        del rounds
        scan = run_phase(output, 'development_scan', pool, sizes, seeds)
        gap = {}
        for key, _, tau, fraction in PAIRS:
            gap[key] = {}
            for phase, summaries in (('train', train), ('independent_development_scan', scan)):
                summary = summaries[key]
                added = summary['cell_counts'] @ weights[key] / summary['n']
                budget_ratio = added / ((1 - fraction) * tau)
                empirical = evaluate_summary(summary, weights[key], tau)
                gap[key][phase] = {'max_added_budget_ratio': float(budget_ratio.max()),
                                    'worst_mean': pool[int(np.argmax(budget_ratio))],
                                    'added_budget_exceedance_count': int(np.sum(budget_ratio > 1 + 1e-7)),
                                    'max_empirical_Ephi_over_tau': float(empirical['Ephi_over_tau'].max())}
        write_json(output / 'grid-versus-MC-gap.json', gap)
        calibration_means, selection_receipt = select_adverse(pool, {'train': train, 'development_scan': scan}, weights)
        audit_means = np.unique(np.vstack((calibration_means, sentinels)), axis=0)
        write_json(output / 'frozen-null-selection.json', {**selection_receipt,
                                                         'calibration_means': calibration_means,
                                                         'calibration_faces': face_membership(calibration_means),
                                                         'audit_means': audit_means,
                                                         'audit_faces': face_membership(audit_means)})
        del train, scan
        calibration_summary = run_phase(output, 'calibration', calibration_means, sizes, seeds)
        calibrated = calibrate_once(calibration_summary, weights)
        del calibration_summary
        selection = select_beta(train_alt, calibrated)
        final_weights = {key: calibrated[key]['weights'] for key, _, _, _ in PAIRS}
        write_json(output / 'frozen-candidates.json', {'calibrated': calibrated, 'selection': selection,
                                                       'audit_and_power_banks_not_yet_opened': True})
        del train_alt
        audit_summary = run_phase(output, 'untouched_audit', audit_means, sizes, seeds)
        audit = audit_frozen(audit_summary, final_weights)
        write_json(output / 'untouched-audit-metrics.json', audit)
        del audit_summary
        power_summary = run_phase(output, 'untouched_power', alt, sizes, seeds)
        power = power_results(power_summary, final_weights, labels)
        write_json(output / 'untouched-power-metrics.json', power)
        stages = {stage: {key: evaluate_summary(power_summary[key], table[key], tau)
                           for key, _, tau, _ in PAIRS}
                  for stage, table in (('initial_grid', initial_weights), ('train_exchanged_unshrunk', weights),
                                       ('calibrated', final_weights))}
        write_json(output / 'power-stage-decomposition.json', stages)
        # Cross-method comparison at the same tau, using selections frozen above.
        cross = {}
        for tau in TAUS:
            pkey, skey = (selection['selected'][f'{kind}_{tau:g}'] for kind in KINDS)
            cross[f'{tau:g}'] = {'projection_selected': pkey, 'simes_selected': skey,
                                'projection_minus_enhanced_simes_pp': 100 * (power[pkey]['tau']['candidate_power']
                                                                           - power[skey]['tau']['candidate_power']),
                                'descriptive_CRN_comparison_only': True}
        for relative, expected in hashes.items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError(f'Source changed during run: {relative}')
        memory = peak_working_set()
        summary = {'status': STATUS, 'smoke_only': args.smoke, 'selection': selection,
                   'models': compact_results(audit, power, calibrated, labels),
                   'cross_method_same_tau': cross, 'train_exchange_round_count': len(exchange_trace),
                   'calibration_row_count': len(calibration_means), 'audit_row_count': len(audit_means),
                   'continuous_null_coverage': False, 'task_acceptance_claim': False,
                   'resources': {'wall_seconds': time.monotonic() - start,
                                 'process_cpu_seconds': time.process_time() - cpu_start,
                                 'peak_working_set_bytes': memory, 'blas_threads': 1, 'highs_threads': 1,
                                 'threadpools': threadpool_info(),
                                 'within_1GiB': memory is None or memory <= 1024**3}}
        write_json(output / 'summary.json', summary)
        print(json.dumps({'output': str(output), 'selection': selection['selected'],
                          'models': {key: summary['models'][key] for key in selection['selected'].values()},
                          'resources': summary['resources']}, default=plain), flush=True)
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'FAILED_DEVELOPMENT_RUN_PRESERVED',
                                             'error': str(error), 'traceback': traceback.format_exc()})
        raise
    finally:
        write_json(output / 'manifest.json', {str(path.relative_to(output)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                                             for path in output.rglob('*') if path.is_file()})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name')
    parser.add_argument('--smoke', action='store_true', help='256 draws/phase, independent seeds; never scientific evidence')
    parser.add_argument('--read-run', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.read_run:
            verified_run(args.read_run)
        else:
            run(args)


if __name__ == '__main__':
    main()
