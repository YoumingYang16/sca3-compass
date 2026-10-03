"""Engineering replay/safety tests only; no fresh scientific simulation."""
from pathlib import Path
import json, os, subprocess, sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/robustness/R0072-repetitions/case-0000/rep-000000-input.npz'
OLD = DATA.with_name('rep-000000-evidence.npz')
pytestmark = pytest.mark.skipif(not DATA.exists(), reason='Local archived DEV replay fixture not distributed')

def command(data, out):
    return [sys.executable, str(ROOT/'scripts/robustness_r1_apply.py'),
            '--source-run', 'R0073', '--data', str(data), '--out', str(out),
            '--algorithm-seed', '1707322967', '--provenance', 'SIMULATION_NOT_PATIENT_DATA',
            '--acknowledge-research-only']

def run(data, out):
    env = dict(os.environ)
    for key in ['RESEARCH_START_GATE', 'RESEARCH_STOP_FILE', 'RESEARCH_WINDOW_FILE']:
        env.pop(key, None)
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.run(command(data, out), cwd=ROOT, env=env, capture_output=True,
                          text=True, encoding='utf-8', timeout=30)

def test_exact_frozen_output_and_truth_independence(tmp_path):
    first = tmp_path / 'original'
    result = run(DATA, first)
    assert result.returncode == 0, result.stderr
    with np.load(first/'evidence.npz') as new, np.load(OLD) as old:
        compared = [key for key in new.files if key != 'primary_rejected']
        assert len(compared) == 88  # 72 e arrays +16 p-component arrays
        for key in compared:
            np.testing.assert_array_equal(new[key], old[key])
    changed = tmp_path / 'changed-labels.npz'
    with np.load(DATA) as data:
        np.savez_compressed(changed, z=data['z'], calibration=data['calibration'], truth=~data['truth'])
    second = tmp_path / 'changed'
    result = run(changed, second)
    assert result.returncode == 0, result.stderr
    with np.load(first/'evidence.npz') as a, np.load(second/'evidence.npz') as b:
        assert a.files == b.files
        for key in a.files:
            np.testing.assert_array_equal(a[key], b[key])
    receipt = json.loads((second/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['truth_present_but_not_loaded'] is True
    assert receipt['power'] is None and receipt['actual_fdr'] is None
    assert receipt['discoveries'] == 49

def test_refuses_overwriting_application(tmp_path):
    out = tmp_path / 'exists'
    out.mkdir()
    marker = out / 'user.txt'
    marker.write_text('Preserve me', encoding='utf-8')
    result = run(DATA, out)
    assert result.returncode != 0 and 'Preserve previous application outputs' in result.stderr
    assert marker.read_text(encoding='utf-8') == 'Preserve me'

def test_refuses_incomplete_confirmation_before_reading_arrays(tmp_path):
    path = tmp_path/'RTEST-repetitions/case-0000/input.npz'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'Not a real scientific input')
    (tmp_path/'RTEST-screen.protocol.json').write_text(json.dumps({'phase': 'C1'}), encoding='utf-8')
    result = run(path, tmp_path/'no-output')
    assert result.returncode != 0 and 'Do not read incomplete confirmation samples' in result.stderr
    assert not (tmp_path/'no-output').exists()

def test_refuses_unassessed_dimensions(tmp_path):
    path = tmp_path/'wrong-shape.npz'
    np.savez_compressed(path, z=np.zeros((8,4,6)), calibration=np.zeros((4,8,6)))
    result = run(path, tmp_path/'no-output')
    assert result.returncode != 0 and 'restricted to the evaluated' in result.stderr
