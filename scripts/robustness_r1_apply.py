"""Apply a HASH-VERIFIED frozen R1 to supplied summary arrays, without truth.

Research-use adapter, not clinical software or a new candidate. It evaluates
the frozen full bank so that outputs can be compared exactly with raw evidence.
It never derives FDR or Power from an unlabelled application dataset.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time

ROOT = Path(__file__).resolve().parents[1]
MAIN = 'R1B_guard_pilotc0.5_projection_gate_eBH'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    from research_window import wait_start_gate
    wait_start_gate()
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-run', default='R0073')
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--algorithm-seed', type=int, required=True)
    parser.add_argument('--provenance', choices=['SIMULATION_NOT_PATIENT_DATA', 'OBSERVATIONAL_RESEARCH_UNVALIDATED'], required=True)
    parser.add_argument('--acknowledge-research-only', action='store_true', required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    source = root / 'artifacts/robustness' / f'{args.source_run}-source'
    protocol_path = root / 'artifacts/robustness' / f'{args.source_run}-screen.protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['backend'] != 'bootstrap_guard' or protocol['R1_main'] != MAIN or protocol['bootstrap_draws'] != 16:
        raise ValueError('This adapter targets the fixed R1 bootstrap-guard, not arbitrary new variants')
    for rel, digest in protocol['source_sha256'].items():
        if sha(source / rel) != digest:
            raise ValueError('Frozen source checksum mismatch: ' + rel)
    data_path = args.data.resolve(strict=True)
    if 'gse320100' in str(data_path).casefold():
        raise ValueError('Protected holdout GSE320100 must remain sealed')
    # Do not accidentally expose active confirmation data through an adapter.
    for parent in data_path.parents:
        if parent.name.endswith('-repetitions'):
            run = parent.name.removesuffix('-repetitions')
            plan_path = parent.parent / f'{run}-screen.protocol.json'
            if plan_path.exists():
                plan = json.loads(plan_path.read_text(encoding='utf-8'))
                if plan['phase'] in ['C1', 'C2']:
                    index_path = parent.parent / f'{run}-results-index.json'
                    if not index_path.exists() or not json.loads(index_path.read_text(encoding='utf-8'))['complete']:
                        raise ValueError('Do not read incomplete confirmation samples')
            break
    if args.out.exists():
        raise FileExistsError('Preserve previous application outputs: ' + str(args.out))
    if args.algorithm_seed < 0:
        raise ValueError('Nonnegative algorithm seed required')
    sys.path.insert(0, str(source / 'src'))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    import numpy as np
    from threadpoolctl import threadpool_limits
    from sca3_compass.robustness_bootstrap_guard import evaluate
    from sca3_compass.robustness_pattern_test import _json_value
    from sca3_compass.molecular_methods import ebh
    import sca3_compass.robustness_bootstrap_guard as implementation
    if sha(implementation.__file__) != protocol['source_sha256']['src/sca3_compass/robustness_bootstrap_guard.py']:
        raise ValueError('Imported wrong method version')
    with np.load(data_path, allow_pickle=False) as archive:
        z = np.array(archive['z'], dtype=float)
        calibration = np.array(archive['calibration'], dtype=float)
        ignored_truth = 'truth' in archive.files  # never load/use labels in the algorithm
    if z.shape != (256, 4, 6) or calibration.ndim != 3 or calibration.shape[0] != 4 or calibration.shape[2] != 6:
        raise ValueError('Adapter restricted to the evaluated G=256/S=4/K=6 setting; no automatic extrapolation')
    if not np.isfinite(z).all() or not np.isfinite(calibration).all():
        raise ValueError('Missing/nonfinite input is not silently imputed')
    start = time.perf_counter()
    with threadpool_limits(1):
        evidence, diagnostics = evaluate(z, calibration, draws=16, seed=args.algorithm_seed)
        rejected = ebh(evidence[MAIN], .05)
    elapsed = time.perf_counter() - start
    target = args.out.resolve()
    target.mkdir(parents=True, exist_ok=False)
    arrays = target / 'evidence.npz'
    with arrays.open('xb') as stream:
        np.savez_compressed(stream, **evidence,
                            **{'p_' + key: value for key, value in diagnostics.pop('held_pvalues').items()},
                            primary_rejected=rejected)
    receipt = {'purpose': 'RESEARCH_APPLICATION_NOT_NEW_CONFIRMATION', 'provenance': args.provenance,
               'candidate': MAIN, 'source_run': args.source_run, 'method_version': protocol['method_version'],
               'protocol_sha256': sha(protocol_path), 'method_sha256': sha(implementation.__file__),
               'adapter_sha256': sha(__file__), 'input_path': str(data_path), 'input_sha256': sha(data_path),
               'evidence_sha256': sha(arrays), 'algorithm_seed': args.algorithm_seed,
               'bootstrap_draws': 16, 'algorithm_nominal_fdr': .05, 'shape': list(z.shape),
               'calibration_shape': list(calibration.shape), 'truth_present_but_not_loaded': ignored_truth,
               'discoveries': int(rejected.sum()), 'elapsed_seconds_all72_labels': elapsed,
               'power': None, 'actual_fdr': None,
               'scope_warning': 'Finite bootstrap perturbations have no general nuisance-coverage/FDR theorem. Common pipeline means, transferable angular covariance and independent calibration/training units are assumptions, not checked facts. Matching array dimensions does not verify these assumptions.',
               'not_claimed': ['clinical utility', 'real-data FDR validation', 'publication readiness', 'current confirmation passed'],
               'diagnostics': _json_value(diagnostics)}
    with (target / 'receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({key: receipt[key] for key in ['candidate', 'discoveries', 'elapsed_seconds_all72_labels', 'power', 'actual_fdr', 'scope_warning']}, indent=2))
    print(target)

if __name__ == '__main__':
    main()
