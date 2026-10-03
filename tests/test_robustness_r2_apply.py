"""Adapter tests ONLY on an already exposed development input."""
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from robustness_r2_apply import check_input_path
from robustness_r2_development import read


def test_protected_path_and_incomplete_confirmation_are_rejected(tmp_path):
    with pytest.raises(ValueError):check_input_path(tmp_path/'GSE320100'/'data.npz')
    folder=tmp_path/'RTEST-repetitions'/'case-0000';folder.mkdir(parents=True)
    (tmp_path/'RTEST-screen.protocol.json').write_text(json.dumps({'phase':'C2'}))
    with pytest.raises(ValueError):check_input_path(folder/'input.npz')


def run_adapter(data,out):
    return subprocess.run([sys.executable,str(ROOT/'scripts/robustness_r2_apply.py'),'--data',str(data),
        '--out',str(out),'--algorithm-seed','1707322967','--provenance','SIMULATION_NOT_PATIENT_DATA',
        '--acknowledge-research-only'],cwd=ROOT,capture_output=True,text=True,timeout=45)


def test_exact_fresh_DEV_outputs_and_ignored_truth(tmp_path):
    if not (ROOT/'artifacts/robustness/R0076-screen.protocol.json').exists():pytest.skip('Frozen C2 source unavailable')
    original=ROOT/'artifacts/robustness/R0072-repetitions/case-0000/rep-000000-input.npz'
    expected=read(ROOT/'artifacts/robustness/R0074-repetitions/case-0000/rep-000000.json.gz')
    first=run_adapter(original,tmp_path/'first')
    assert first.returncode==0,first.stderr
    with np.load(original) as z:
        changed=tmp_path/'changed-truth.npz'
        np.savez_compressed(changed,z=z['z'],calibration=z['calibration'],truth=~z['truth'])
    second=run_adapter(changed,tmp_path/'second')
    assert second.returncode==0,second.stderr
    with np.load(tmp_path/'first/evidence.npz') as a,np.load(tmp_path/'second/evidence.npz') as b,np.load(expected['evidence_path']) as old:
        for name in a.files:
            np.testing.assert_array_equal(a[name],b[name])
            if name.startswith('R2K_'):np.testing.assert_array_equal(a[name],old[name])
    receipt=read(tmp_path/'first/receipt.json')
    assert receipt['power'] is None and receipt['actual_fdr'] is None
    assert receipt['truth_present_but_not_loaded'] and receipt['fresh_no_cached_nuisance']
    repeated=run_adapter(original,tmp_path/'first')
    assert repeated.returncode!=0 and 'Preserve previous' in repeated.stderr
