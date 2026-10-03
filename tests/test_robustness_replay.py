"""Tiny replay fixtures only: no research experiments or runner modifications."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/robustness_replay.py"
SPEC = importlib.util.spec_from_file_location("robustness_replay_sidecar_test", SCRIPT)
replay = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(replay)

FAKE_RUNNER = '''
import argparse
import json
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from sca3_compass.molecular_data import PROJECT_ROOT
from sca3_compass.helper import VALUE

def one(task):
    case, repetitions = task
    return {"case": case, "repetitions": repetitions, "power_defined": True,
            "replicated_signed_truths": 3,
            "rows": [{"method": "tiny", "fdp_by_repetition": [VALUE]*repetitions,
                      "power_by_repetition": [case['effect']]*repetitions}]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id')
    parser.add_argument('--repetitions',type=int)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--workers',type=int)
    parser.add_argument('--config')
    parser.add_argument('--draws',type=int)
    parser.add_argument('--profile')
    parser.add_argument('--mc',action='store_true')
    parser.add_argument('--soft',action='store_true')
    parser.add_argument('--loading',action='store_true')
    parser.add_argument('--prior',action='store_true')
    args=parser.parse_args()
    original=json.loads((PROJECT_ROOT/'original-protocol.json').read_text(encoding='utf-8'))
    settings={**original,'run_id':args.run_id,'repetitions':args.repetitions,
              'cases':json.loads(Path(args.config).read_text(encoding='utf-8'))['cases']}
    sources=[Path(__file__),*sorted((PROJECT_ROOT/'src/sca3_compass').glob('*.py'))]
    settings['source_sha256']={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with ProcessPoolExecutor(args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        results=list(pool.map(one,[(case,args.repetitions) for case in settings['cases']]))
    output=PROJECT_ROOT/f'artifacts/robustness/{args.run_id}-screen.json'
    with output.open('x',encoding='utf-8') as stream:
        json.dump({'settings':settings,'scenarios':results},stream)

if __name__=='__main__':
    main()
'''


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


@pytest.fixture
def archive(tmp_path_factory):
    # Keep nested replay paths below Windows MAX_PATH without changing OS policy.
    project = tmp_path_factory.mktemp("r") / "p"
    folder = project / "artifacts/robustness"
    snapshot = folder / "RTEST-source"
    sources = {replay.RUNNER: FAKE_RUNNER,
               "src/sca3_compass/__init__.py": '"""Frozen fixture."""\n',
               "src/sca3_compass/molecular_data.py": "from pathlib import Path\nPROJECT_ROOT=Path(__file__).resolve().parents[2]\n",
               "src/sca3_compass/helper.py": "VALUE=.125\n"}
    hashes = {}
    for path, text in sources.items():
        put(snapshot / path, text)
        hashes[path.replace("/", "\\")] = replay.sha(snapshot / path)
    protocol = {"run_id": "RTEST", "source_sha256": hashes,
                "phase": "DEVELOPMENT_ONLY", "provenance": "SIMULATION_NOT_PATIENT_DATA",
                "independent_confirmation": False, "seed": 123, "repetitions": 3,
                "cases": [{"name": "a", "effect": .25}, {"name": "b", "effect": .5}],
                "mc": False, "soft": False, "loading": False, "prior": False,
                "draws": 31, "profile": "frontier"}
    results = [{"case": case, "repetitions": 3, "power_defined": True,
                "replicated_signed_truths": 3,
                "rows": [{"method": "tiny", "fdp_by_repetition": [.125]*3,
                          "power_by_repetition": [case["effect"]]*3}]} for case in protocol["cases"]]
    put(folder / "RTEST-screen.protocol.json", protocol)
    put(folder / "RTEST-screen.json", {"settings": protocol, "scenarios": results})
    put(folder / "EXPERIMENT_REGISTRY.json", {"experiments": [{"id": "RTEST", "status": "completed"}]})
    return project, folder, snapshot, protocol, results


def test_full_audit_and_transitive_closure(archive):
    project, _, _, _, _ = archive
    *_, audit, sources = replay.load_archive(project, "RTEST")
    assert audit["verified_files"] == 4
    assert "sca3_compass.helper" in audit["reachable_local_modules"]
    assert "concurrent" in audit["external_import_roots"]
    assert set(audit["source_sha256"]) == set(sources)


@pytest.mark.parametrize("missing", ["src/sca3_compass/__init__.py", "src/sca3_compass/helper.py"])
def test_old_leading_only_archive_rejected(archive, missing):
    _, _, snapshot, protocol, _ = archive
    protocol = {**protocol, "source_sha256": {k: v for k, v in protocol["source_sha256"].items()
                                             if k.replace("\\", "/") != missing}}
    (snapshot / missing).unlink()
    with pytest.raises(replay.ReplayError, match="transitive module"):
        replay.audit_source(snapshot, protocol)


def test_missing_file_hash_change_and_extra_are_rejected(archive):
    _, _, snapshot, protocol, _ = archive
    target = snapshot / "src/sca3_compass/helper.py"
    target.write_text("VALUE=.9\n", encoding="utf-8")
    with pytest.raises(replay.ReplayError, match="hash mismatch"):
        replay.audit_source(snapshot, protocol)
    target.unlink()
    with pytest.raises(replay.ReplayError, match="Missing frozen source"):
        replay.audit_source(snapshot, protocol)
    put(target, "VALUE=.125\n")
    put(snapshot / "src/sca3_compass/unpinned.py", "VALUE=1\n")
    with pytest.raises(replay.ReplayError, match="Unhashed Python"):
        replay.audit_source(snapshot, protocol)


@pytest.mark.parametrize("bad", ["../evil.py", "C:/evil.py", "/evil.py", "src/../evil.py",
                                  "src//sca3_compass/a.py", "src/sca3_compass/a.py:stream",
                                  "external/a.py"])
def test_unsafe_source_paths(bad):
    with pytest.raises(replay.ReplayError):
        replay.relative_source(bad)


@pytest.mark.parametrize("bad", ["../RUN", "/R001", "C:foo", "R 1", "R.1", "", "1"])
def test_unsafe_ids(bad):
    with pytest.raises(replay.ReplayError):
        replay.run_id(bad)


def test_normalized_path_duplicates_rejected(archive):
    _, _, snapshot, protocol, _ = archive
    hashes = dict(protocol["source_sha256"])
    hashes[replay.RUNNER] = hashes[replay.RUNNER.replace("/", "\\")]
    with pytest.raises(replay.ReplayError, match="Duplicate normalized"):
        replay.audit_source(snapshot, {**protocol, "source_sha256": hashes})


def test_dynamic_import_requires_explicit_audit(archive):
    _, _, snapshot, protocol, _ = archive
    target = snapshot / "src/sca3_compass/helper.py"
    put(target, "import importlib\nVALUE=importlib.import_module('math').pi\n")
    hashes = {**protocol["source_sha256"], "src\\sca3_compass\\helper.py": replay.sha(target)}
    with pytest.raises(replay.ReplayError, match="Dynamic source loading"):
        replay.audit_source(snapshot, {**protocol, "source_sha256": hashes})


def test_exclusive_stage_preserves_original(archive):
    project, folder, snapshot, _, _ = archive
    before = {str(p): replay.sha(p) for p in folder.rglob("*") if p.is_file()}
    root = replay.stage_replay(project, "RTEST", "RREPLAY", workers=2, smoke_cases=1, smoke_repetitions=1)
    manifest, _ = replay.verify_stage(root)
    assert manifest["mode"] == "smoke_prefix"
    assert manifest["independent_confirmation"] is False
    assert manifest["python_environment"]["site_hooks_disabled"]
    for path, expected in before.items():
        assert replay.sha(path) == expected
    for path in snapshot.rglob("*.py"):
        assert path.read_bytes() == (root / path.relative_to(snapshot)).read_bytes()
    with pytest.raises(FileExistsError):
        replay.stage_replay(project, "RTEST", "RREPLAY")
    with pytest.raises(replay.ReplayError, match="new run ID"):
        replay.stage_replay(project, "RTEST", "RTEST")


def test_stage_refuses_registered_id_and_bad_prefix(archive):
    project, folder, _, _, _ = archive
    put(folder / "EXPERIMENT_REGISTRY.json", {"experiments": [{"id": "RUSED"}]})
    with pytest.raises(replay.ReplayError, match="already registered"):
        replay.stage_replay(project, "RTEST", "rused")
    for kwargs in ({"smoke_cases": 0}, {"smoke_cases": 3}, {"smoke_repetitions": 4}, {"workers": 0}):
        with pytest.raises(replay.ReplayError):
            replay.stage_replay(project, "RTEST", "RFRESH", **kwargs)


def test_staged_hash_and_argument_tampering_rejected(archive):
    project, _, _, _, _ = archive
    root = replay.stage_replay(project, "RTEST", "RMUTATION")
    manifest = replay.read_json(root / "replay-manifest.json")
    manifest["runner_arguments"][manifest["runner_arguments"].index("--seed") + 1] = "999"
    put(root / "replay-manifest.json", manifest)
    with pytest.raises(replay.ReplayError, match="arguments differ"):
        replay.verify_stage(root)
    put(root / "src/sca3_compass/helper.py", "VALUE=.99\n")
    with pytest.raises(replay.ReplayError, match="source hash mismatch"):
        replay.verify_stage(root)


def test_isolated_subprocess_and_spawn_workers_ignore_live_source(archive, monkeypatch):
    project, folder, _, _, _ = archive
    root = replay.stage_replay(project, "RTEST", "RSPAWN", workers=2)
    # If PYTHONPATH/current/editable sources contaminate replay this raises.
    put(project / "src/sca3_compass/__init__.py", "raise RuntimeError('LIVE SOURCE LEAK')\n")
    monkeypatch.setenv("PYTHONPATH", str(project / "src"))
    result = replay.execute_replay(root, compare=True)
    assert result["exact_match"] and result["arrays_compared"] == 4
    assert result["all_original_cases_compared"] and result["full_repetition_arrays"]
    audits = [replay.read_json(p) for p in (root / "import-audits").glob("*.json")]
    assert {a["role"] for a in audits} == {"parent", "spawn_worker"}
    assert len(audits) >= 2
    for audit in audits:
        assert audit["isolated"] and audit["no_site"]
        assert all(Path(origin).is_relative_to(root) for origin in audit["origins"].values())
    assert not (folder / "RSPAWN-screen.json").exists()
    with pytest.raises(replay.ReplayError, match="already been executed"):
        replay.execute_replay(root)


def test_comparison_detects_numeric_mismatch_and_method_omission(archive):
    project, _, _, _, _ = archive
    root = replay.stage_replay(project, "RTEST", "RCOMPARE", smoke_repetitions=1, smoke_cases=1)
    replay.execute_replay(root)
    path = root / "artifacts/robustness/RCOMPARE-screen.json"
    document = replay.read_json(path)
    document["scenarios"][0]["rows"][0]["fdp_by_repetition"][0] += 1e-15
    put(path, document)
    result = replay.compare_replay(root)
    assert not result["exact_match"] and result["mismatches"][0]["first_difference"] == 0
    assert not result["all_original_cases_compared"] and not result["full_repetition_arrays"]
    document["scenarios"][0]["rows"][0]["method"] = "substituted"
    put(path, document)
    assert "method set" in replay.compare_replay(root)["mismatches"][0]["reason"]


@pytest.mark.parametrize("schema", [1, 2])
def test_partial_reference_checkpoints_and_exact_coverage(archive, schema):
    project, folder, _, protocol, results = archive
    (folder / "RTEST-screen.json").unlink()
    if schema == 1:
        partial = {"settings": protocol, "completed": [results[1]]}
    else:
        partial = {"schema_version": 2, "settings": protocol, "completed_case_indices": [1],
                   "case_checkpoint_directory": "C:/UNTRUSTED/PATH"}
        put(folder / "RTEST-cases/case-0001.json",
            {"case_index": 1, "settings_digest": replay.content_digest(protocol),
             "result_digest": replay.content_digest(results[1]), "result": results[1]})
    put(folder / "RTEST-screen.partial.json", partial)
    root = replay.stage_replay(project, "RTEST", f"RPARTIAL{schema}")
    result = replay.execute_replay(root, compare=True)
    assert result["exact_match"] and result["reference_status"] == "partial"
    assert result["case_indices_compared"] == [1] and not result["all_original_cases_compared"]
    assert result["arrays_compared"] == 2
    if schema == 2:
        target = folder / "RTEST-cases/case-0001.json"
        item = replay.read_json(target)
        item["result_digest"] = "0" * 64
        put(target, item)
        with pytest.raises(replay.ReplayError, match="Checkpoint digest"):
            replay.reference_results(folder, protocol)


def test_runtime_guard_rejects_unknown_and_modified_modules(archive):
    _, _, snapshot, protocol, _ = archive
    audit, _ = replay.audit_source(snapshot, protocol)
    guard = replay.FrozenFinder(snapshot, audit["source_sha256"])
    with pytest.raises(ImportError, match="Unfrozen"):
        guard.find_spec("sca3_compass.missing")
    spec = guard.find_spec("sca3_compass.helper")
    module = importlib.util.module_from_spec(spec)
    put(snapshot / "src/sca3_compass/helper.py", "VALUE=.9\n")
    with pytest.raises(ImportError, match="hash mismatch"):
        guard.exec_module(module)


def test_duplicate_json_and_nonfinite_raw_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for payload in ('{"a":1,"a":2}', '{"value":NaN}'):
        put(path, payload)
        with pytest.raises(replay.ReplayError):
            replay.read_json(path)
    with pytest.raises(replay.ReplayError, match="Invalid raw"):
        replay.raw_case({"case": {}, "repetitions": 1,
                         "rows": [{"method": "x", "fdp_by_repetition": [float("nan")],
                                   "power_by_repetition": [0]}]})


def test_environment_probe_does_not_import_project():
    result = replay.environment(sys.executable)
    assert result["site_hooks_disabled"]
    assert len(result["executable_sha256"]) == 64
    assert result["packages"] and result["dependency_paths"]
    assert result["executable_sha256"] == hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()


def test_unreachable_manifest_file_is_still_verified(archive):
    _, _, snapshot, protocol, _ = archive
    extra = snapshot / "src/sca3_compass/unused.py"
    put(extra, "UNUSED=True\n")
    hashes = {**protocol["source_sha256"], "src/sca3_compass/unused.py": replay.sha(extra)}
    protocol = {**protocol, "source_sha256": hashes}
    audit, _ = replay.audit_source(snapshot, protocol)
    assert audit["verified_files"] == 5
    assert "sca3_compass.unused" not in audit["reachable_local_modules"]
    put(extra, "UNUSED=False\n")
    with pytest.raises(replay.ReplayError, match="hash mismatch"):
        replay.audit_source(snapshot, protocol)


def test_unknown_protocol_setting_never_silently_dropped(archive):
    _, _, _, protocol, _ = archive
    with pytest.raises(replay.ReplayError, match="Unrecognized protocol"):
        replay.runner_arguments({**protocol, "new_tuning_parameter": 7}, FAKE_RUNNER,
                                "RNEW", 1, 3, "cases.json")


def test_missing_integrity_entry_and_unsafe_destination_rejected(archive):
    project, _, _, _, _ = archive
    root = replay.stage_replay(project, "RTEST", "RINTEGRITY")
    original = replay.read_json(root / "replay-manifest.json")
    manifest = json.loads(json.dumps(original))
    del manifest["integrity"]["reference-arrays.json"]
    put(root / "replay-manifest.json", manifest)
    with pytest.raises(replay.ReplayError, match="required input integrity"):
        replay.verify_stage(root)
    manifest = json.loads(json.dumps(original))
    manifest["new_run_id"] = "../RTEST"
    put(root / "replay-manifest.json", manifest)
    with pytest.raises(replay.ReplayError, match="safe single path"):
        replay.verify_stage(root)


def test_new_staged_source_cannot_be_imported_or_enter_source_glob(archive):
    project, _, _, _, _ = archive
    root = replay.stage_replay(project, "RTEST", "REXTRA")
    put(root / "src/sca3_compass/extra.py", "VALUE=1\n")
    with pytest.raises(replay.ReplayError, match="Unhashed Python"):
        replay.verify_stage(root)


def test_original_settings_mismatch_rejected(archive):
    _, folder, _, protocol, results = archive
    put(folder / "RTEST-screen.json", {"settings": {**protocol, "seed": 999}, "scenarios": results})
    with pytest.raises(replay.ReplayError, match="settings disagree"):
        replay.reference_results(folder, protocol)
