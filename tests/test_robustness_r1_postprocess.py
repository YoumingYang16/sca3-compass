"""No tasks launched: deterministic completion-gate tests only."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('r1_postprocess_fixture', SCRIPTS/'robustness_r1_postprocess.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('mutation', [None, 'partial', 'failed', 'plan', 'index', 'protocol', 'receipts', 'wrong_run'])
def test_requires_original_complete_fixed_batch(mutation):
    protocol = {'run_id': 'R0073', 'repetition_counts': [2, 3]}
    progress = {'run_id': 'R0073', 'completed': 5, 'failed': 0, 'planned': 5}
    index = {'run_id': 'R0073', 'complete': True, 'settings': deepcopy(protocol), 'receipts': [None]*5}
    if mutation == 'partial': progress['completed'] = 4
    elif mutation == 'failed': progress['failed'] = 1
    elif mutation == 'plan': progress['planned'] = 6
    elif mutation == 'index': index['complete'] = False
    elif mutation == 'protocol': index['settings']['repetition_counts'] = [3, 2]
    elif mutation == 'receipts': index['receipts'].pop()
    elif mutation == 'wrong_run': index['run_id'] = 'ROTHER'
    assert module.fixed_completion(progress, index, protocol) is (mutation is None)


def test_transient_read_denial_retries_without_modifying_input(monkeypatch):
    calls = []
    def read_text(path, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            raise PermissionError('fixture sharing violation')
        return '{"completed": 5}'
    monkeypatch.setattr(Path, 'read_text', read_text)
    monkeypatch.setattr(module.time, 'sleep', lambda delay: None)
    assert module.read('fixture.json') == {'completed': 5}
    assert len(calls) == 3


def test_permanent_denial_is_bounded(monkeypatch):
    calls = []
    def denied():
        calls.append(1)
        raise PermissionError('fixture persistent denial')
    monkeypatch.setattr(module.time, 'sleep', lambda delay: None)
    with pytest.raises(PermissionError):
        module.retry_sharing_operation(denied)
    assert len(calls) == 10


def test_malformed_json_not_retried(monkeypatch):
    calls = []
    def malformed(path, **kwargs):
        calls.append(path)
        return '{not valid json'
    monkeypatch.setattr(Path, 'read_text', malformed)
    with pytest.raises(module.json.JSONDecodeError):
        module.read('fixture.json')
    assert len(calls) == 1


def test_atomic_replace_denial_retries(monkeypatch, tmp_path):
    original_replace = module.os.replace
    calls = []
    def replace(src, dst):
        calls.append((src, dst))
        if len(calls) == 1:
            raise PermissionError('fixture temporary reader lock')
        return original_replace(src, dst)
    monkeypatch.setattr(module.os, 'replace', replace)
    monkeypatch.setattr(module.time, 'sleep', lambda delay: None)
    target = tmp_path / 'state.json'
    module.save(target, {'status': 'fixture'})
    assert module.read(target) == {'status': 'fixture'}
    assert len(calls) == 2
