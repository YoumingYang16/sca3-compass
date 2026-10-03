"""Same-input frozen numerical forensics; no scoring or confirmation repair."""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import shutil
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else str(value)
    return value


def gaussian_interior(x, numerical_calls):
    n, k = x.shape
    norm2 = np.sum(x*x, axis=1)
    common = k*x.mean(axis=1)**2
    orth = norm2-common  # exactly the frozen objective's arithmetic
    c, o = float(common.mean()), float(orth.mean())
    if min(c, o) <= 0:
        return {'interior_solution_available': False, 'reason': 'nonpositive eigen-energy'}
    scale = (c+o)/k
    rho = (c-o/(k-1))/(c+o)
    log_scale = float(np.log(scale))
    inside = -.95/(k-1) < rho < .995 and -7 < log_scale < 5

    def value_gradient(params):
        r, ls = np.asarray(params, float)
        s = np.exp(ls); a, b = 1-r, 1+(k-1)*r
        q = (orth/a+common/b)/s
        value = .5*(np.mean(q)+(k-1)*np.log(a)+np.log(b)+k*ls+k*np.log(2*np.pi))
        gradient = [.5*(-(k-1)/a+(k-1)/b+(o/a**2-(k-1)*c/b**2)/s),
                    .5*(k-float(np.mean(q)))]
        return {'objective': float(value), 'analytic_gradient': gradient}

    analytic = value_gradient([rho, log_scale])
    comparisons = []
    for call in numerical_calls:
        if len(call['x']) == 2:
            check = value_gradient(call['x'])
            comparisons.append({'success': call['success'], 'message': call['message'],
                'optimizer_fd_jac': call.get('jac'), 'analytic_gradient_at_numerical_fit': check['analytic_gradient'],
                'objective_excess_over_analytic': check['objective']-analytic['objective'],
                'rho_error_vs_analytic': call['x'][0]-rho,
                'log_scale_error_vs_analytic': call['x'][1]-log_scale})
    return {'interior_solution_available': inside, 'rho': rho, 'scatter': scale,
        'lambda_parallel': c, 'lambda_orthogonal': o/(k-1), 'at_analytic': analytic,
        'numerical_Gaussian_comparisons': comparisons,
        'interpretation': 'Observed-data Gaussian subproblem only; not true noise/oracle, not changed frozen fit or general Student optimum'}


def main():
    begun = time.perf_counter()
    base = ROOT/'artifacts/robustness'
    protocol_path = base/'R0076-screen.protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    snapshot = base/'R0076-source'
    for rel, digest in protocol['source_sha256'].items():
        if sha(snapshot/rel) != digest:
            raise ValueError('Frozen source changed: '+rel)
    failures = []
    for case in [80, 81]:
        for path in sorted((base/'R0076-repetitions'/f'case-{case:04}').glob('rep-*.json.gz')):
            record = json.loads(gzip.decompress(path.read_bytes()))
            if record['status'] != 'completed':
                failures.append((path, record))
    if not failures:
        raise ValueError('No observed failure in the prespecified diagnostic cases')
    out = base/'R0076-failure-forensics-initial'
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__, out/'diagnose_source.py')
    sys.path[:0] = [str(snapshot/'src'), str(snapshot/'scripts')]
    from sca3_compass import robustness_methods as methods
    from robustness_r1_window import generate
    if not Path(methods.__file__).resolve().is_relative_to(snapshot.resolve()):
        raise ValueError('Forensic import is not the frozen module')
    original_minimize = methods.minimize
    records = []
    with threadpool_limits(1):
        for path, failure in failures:
            case, rep = failure['case_index'], failure['rep']
            ip = path.parent/f'rep-{rep:06}-input.npz'
            with np.load(ip, allow_pickle=False) as saved:
                regenerated = generate(protocol['seed'], case, rep, protocol['cases'][case])
                exact = {key: bool(np.array_equal(saved[key], value))
                         for key, value in zip(['z', 'calibration', 'truth'], regenerated)}
                x = saved['calibration'].reshape(-1, saved['calibration'].shape[-1])
            if not all(exact.values()):
                raise ValueError('Failure input does not regenerate exactly')
            all_calls = []; replay = []

            def capture(*args, **kwargs):
                result = original_minimize(*args, **kwargs)
                all_calls.append({key: clean(getattr(result, key, None))
                    for key in ['success', 'status', 'message', 'nit', 'nfev', 'njev', 'fun', 'x', 'jac']})
                return result  # transparent logging, never edits/replaces OptimizeResult

            methods.minimize = capture
            try:
                for iteration in range(2):
                    before = len(all_calls)
                    fit = methods.fit_calibration(x)
                    replay.append({'attempt': iteration+1, 'fit': vars(fit),
                                   'optimizer_calls': all_calls[before:]})
            finally:
                methods.minimize = original_minimize
            records.append({'case': case, 'rep': rep, 'failure_record': str(path),
                'failure_record_sha256': sha(path), 'failure_error': failure['error'],
                'failure_traceback': failure['traceback'], 'input_path': str(ip), 'input_sha256': sha(ip),
                'input_regeneration_exact': exact, 'calibration_shape': list(x.shape),
                'frozen_exact_fit_replays': replay, 'Gaussian_subproblem': gaussian_interior(x, all_calls)})
    payload = {'run_id': 'R0076', 'kind': 'POST_FAILURE_NUMERICAL_DIAGNOSTIC_NOT_CONFIRMATION',
        'complete': True, 'protocol_sha256': sha(protocol_path), 'script_sha256': sha(__file__),
        'records': records, 'elapsed_seconds': time.perf_counter()-begun,
        'finished_utc': datetime.now(timezone.utc).isoformat(), 'provenance': 'SIMULATION_NOT_PATIENT_DATA',
        'limits': ['Same saved inputs/seeds only; two frozen fit calls per observed failure',
                   'No Power/FDR/true-parameter use in fitting; truth checked only for exact regeneration',
                   'No change/replacement of original method, failure record, index, or confirmation status',
                   'Gaussian closed form is a diagnostic subproblem, not a post-hoc C2 repair']}
    with (out/'summary.json').open('x', encoding='utf-8') as stream:
        json.dump(clean(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
    print(out, flush=True)


if __name__ == '__main__':
    main()
