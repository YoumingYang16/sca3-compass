"""Synthetic file-integrity fixtures, not new C2 observations."""
from pathlib import Path
import json
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from robustness_r2_rerender import fresh_output, verified_inputs, RENDERER_SHA256
from robustness_r2_development import sha


def fixture(base, complete=True):
    report = base / 'R0076-evidence-report'
    report.mkdir()
    analysis = base / 'R0076-confirmation-analysis.json'
    replay = base / 'R0076-replay.json'
    c1 = base / 'R0073-confirmation-analysis.json'
    analysis.write_text(json.dumps({'complete': complete}), encoding='utf-8')
    replay.write_text(json.dumps({'complete': True, 'analysis_sha256': sha(analysis)}), encoding='utf-8')
    c1.write_text(json.dumps({'complete': True}), encoding='utf-8')
    text = report / 'README_ZH.md'
    text.write_text('SYNTHETIC_UNIT_FIXTURE_NOT_RESEARCH', encoding='utf-8')
    manifest = {'run_id': 'R0076', 'script_sha256': RENDERER_SHA256,
                'analysis_sha256': sha(analysis), 'replay_sha256': sha(replay),
                'C1_analysis_sha256': sha(c1), 'generated_files': {text.name: sha(text)}}
    (report / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return report


def test_new_output_is_scoped_and_does_not_create_anything(tmp_path):
    path = tmp_path / 'R0076-rerender-example'
    assert fresh_output(tmp_path, path) == path.resolve()
    assert not path.exists()
    for bad in [tmp_path / 'R0076-evidence-report', tmp_path / 'nested/R0076-rerender-new',
                tmp_path.parent / 'R0076-rerender-new', tmp_path / 'R0076-rerender-']:
        with pytest.raises(ValueError):
            fresh_output(tmp_path, bad)
    path.mkdir()
    with pytest.raises(FileExistsError):
        fresh_output(tmp_path, path)


def test_complete_hash_bound_inputs_are_accepted(tmp_path):
    fixture(tmp_path)
    data, receipts = verified_inputs(tmp_path, ROOT / 'scripts/robustness_r2_evidence_report.py')
    assert set(data) == {'analysis', 'replay', 'C1'}
    assert all(item['complete'] for item in data.values())
    assert len(receipts) == 5


def test_original_report_mutation_is_rejected(tmp_path):
    report = fixture(tmp_path)
    (report / 'README_ZH.md').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='Original report file changed'):
        verified_inputs(tmp_path, ROOT / 'scripts/robustness_r2_evidence_report.py')


def test_even_hash_bound_incomplete_analysis_is_rejected(tmp_path):
    fixture(tmp_path, complete=False)
    with pytest.raises(ValueError, match='Completed evidence required'):
        verified_inputs(tmp_path, ROOT / 'scripts/robustness_r2_evidence_report.py')
