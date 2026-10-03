"""Isolated contract fixtures; none of these are scientific experiment data."""
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from sca3_compass import molecular_api as api
from sca3_compass.molecular_data import digest


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(api, "REGISTRY", tmp_path / "configs/registry.json")
    monkeypatch.setattr(api, "PROTOCOL", tmp_path / "configs/protocol.json")
    save(api.REGISTRY, {"fixture_only": True})
    save(api.PROTOCOL, {"fixture_only": True})
    save(tmp_path / "configs/molecular_shrinkage_protocol.json", {"fixture_only": True})
    source = Path(api.__file__).parent
    audit = {"registry_sha256": digest(api.REGISTRY), "summary": {"download_complete": 1}, "datasets": [
        {"accession": "GSE320100", "role": "sealed_candidate", "family": "fixture", "organism": "mouse", "download_status": "complete", "source_url": "https://example.invalid/fixture", "metadata": {"sample_count": 1}}]}
    save(tmp_path / "artifacts/molecular-data-audit.json", audit)
    qc = {"acquisition_sha256": digest(tmp_path / "artifacts/molecular-data-audit.json"),
          "source_sha256": digest(source / "molecular_qc.py"), "summary": {"matrix_qc_complete": 0},
          "datasets": [{"accession": "GSE320100", "status": "sealed_not_parsed", "features": None}]}
    save(tmp_path / "artifacts/molecular-qc.json", qc)
    app = FastAPI()
    app.include_router(api.router)
    return TestClient(app), tmp_path, source


def test_overview_and_sealed_profile_do_not_expose_expression(store):
    client, _, _ = store
    overview = client.get("/api/molecular/overview").json()
    assert overview["qc_current"] and overview["registry_current"]
    assert overview["datasets"][0]["qc_status"] == "sealed_not_parsed"
    response = client.get("/api/molecular/datasets/GSE320100")
    assert response.status_code == 200
    assert "pca" not in response.json()["qc"] and "counts" not in response.json()["qc"]
    assert client.get("/api/molecular/datasets/GSE999").status_code == 404
    assert client.post("/api/molecular/datasets/GSE320100").status_code == 405


@pytest.mark.parametrize("bad", [None, "{broken", "[]"])
def test_missing_or_invalid_artifact_fails_closed(store, bad):
    client, root, _ = store
    if bad is not None:
        (root / "artifacts/molecular-benchmark-validation.json").write_text(bad, encoding="utf-8")
    assert client.get("/api/molecular/benchmark").status_code == 503


def test_phase_is_validated(store):
    assert store[0].get("/api/molecular/benchmark?phase=unseal").status_code == 422


def test_benchmark_detects_stale_or_missing_source_manifest(store):
    client, root, source = store
    report = {"protocol_sha256": digest(api.PROTOCOL), "source_sha256": {
        name: digest(source / name) for name in ("molecular_benchmark.py", "molecular_methods.py")},
        "rows": [{"method": "test_fixture", "fdp_by_repetition": [0.1]}]}
    path = root / "artifacts/molecular-benchmark-validation.json"
    save(path, report)
    returned = client.get("/api/molecular/benchmark").json()
    assert returned["current"] and "fdp_by_repetition" not in returned["rows"][0]
    assert "fdp_by_repetition" in json.loads(path.read_text())["rows"][0]  # Read-only API.
    report["source_sha256"] = {}
    save(path, report)
    assert not client.get("/api/molecular/benchmark").json()["current"]


def test_qc_detects_upstream_change(store):
    client, root, _ = store
    audit = root / "artifacts/molecular-data-audit.json"
    report = json.loads(audit.read_text())
    report["fixture_amendment"] = True
    save(audit, report)
    assert not client.get("/api/molecular/overview").json()["qc_current"]


def test_model_detects_changed_processed_input(store):
    client, root, source = store
    matrix = root / "data/processed/molecular/fixture-input.npz"
    matrix.parent.mkdir(parents=True)
    matrix.write_bytes(b"fixture; not a real scientific matrix")
    output = root / "data/processed/molecular/GSE107958-shrinkage.npz"
    output.write_bytes(b"fixture result")
    qc_path = root / "artifacts/molecular-qc.json"
    qc = json.loads(qc_path.read_text())
    qc["datasets"].append({"accession": "GSE107958", "derived_path": str(matrix.relative_to(root)), "derived_sha256": digest(matrix)})
    save(qc_path, qc)
    report = {"accession": "GSE107958", "source_sha256": digest(source / "molecular_shrinkage.py"),
              "protocol_sha256": digest(root / "configs/molecular_shrinkage_protocol.json"),
              "qc_sha256": digest(qc_path), "input_sha256": digest(matrix), "result_sha256": digest(output)}
    save(root / "artifacts/molecular-shrinkage.json", report)
    assert client.get("/api/molecular/shrinkage").json()["current"]
    matrix.write_bytes(b"changed fixture")
    assert not client.get("/api/molecular/shrinkage").json()["current"]


@pytest.mark.parametrize("endpoint", ["envelope", "transport", "animal-blocks"])
def test_new_endpoints_require_artifacts_and_are_read_only(store, endpoint):
    client, _, _ = store
    assert client.get(f"/api/molecular/{endpoint}").status_code == 503
    assert client.post(f"/api/molecular/{endpoint}").status_code == 405


def test_envelope_current_and_whitelisted_manifest(store):
    client, root, source = store
    config = root / "configs/molecular_envelope_protocol.json"
    save(config, {"fixture_only": True})
    report = {"effective": {"fingerprint": {"protocol_sha256": digest(config), "sources": {
        name: digest(source / name) for name in ["molecular_envelope.py", "molecular_envelope_benchmark.py",
                                              "molecular_methods.py", "molecular_benchmark.py"]}}},
        "scenarios": [{"rows": [{"method": "fixture", "fdp_by_repetition": [0.0]}]}]}
    path = root / "artifacts/molecular-envelope-validation.json"
    save(path, report)
    got = client.get("/api/molecular/envelope").json()
    assert got["current"] and "fdp_by_repetition" not in got["scenarios"][0]["rows"][0]
    assert "fdp_by_repetition" in json.loads(path.read_text())["scenarios"][0]["rows"][0]
    report["effective"]["fingerprint"]["sources"] = {}
    save(path, report)
    assert not client.get("/api/molecular/envelope").json()["current"]
    assert client.get("/api/molecular/envelope?phase=unseal").status_code == 422


def test_transport_stale_expression_and_gate(store, monkeypatch):
    from sca3_compass import molecular_transport as transport

    client, root, _ = store
    frozen = root / "artifacts/molecular-transport-freeze.json"
    expression = root / "fixture-expression.npz"
    expression.write_bytes(b"FIXTURE_NOT_RESEARCH_DATA")
    monkeypatch.setattr(transport, "fingerprint", lambda: {"fixture": "same"})
    save(frozen, {"fingerprint": {"fixture": "same"}, "created_at": "fixture"})
    save(root / "artifacts/molecular-transport.json", {
        "freeze_sha256": digest(frozen), "expression_path": expression.name, "expression_sha256": digest(expression)})
    assert client.get("/api/molecular/transport").json()["current"]
    def gate():
        return next(r["status"] for r in client.get("/api/molecular/overview").json()["release_gates"]
                    if r["name"] == "Frozen public-cohort transport")
    assert gate() == "evaluated_current"
    expression.write_bytes(b"CHANGED_FIXTURE")
    assert not client.get("/api/molecular/transport").json()["current"]
    assert gate() == "stale"


def test_animal_blocks_current_and_stale(store):
    client, root, source = store
    config = root / "configs/molecular_block_validation.json"
    matrix = root / "data/processed/molecular/GSE107958-counts.npz"
    matrix.parent.mkdir(parents=True)
    matrix.write_bytes(b"FIXTURE_NOT_RESEARCH_DATA")
    save(config, {"fixture_only": True})
    save(root / "artifacts/molecular-block-validation.json", {
        "protocol_sha256": digest(config), "source_sha256": digest(source / "molecular_block_validation.py"),
        "shrinkage_source_sha256": digest(source / "molecular_shrinkage.py"),
        "qc_sha256": digest(root / "artifacts/molecular-qc.json"), "input_sha256": digest(matrix)})
    assert client.get("/api/molecular/animal-blocks").json()["current"]
    matrix.write_bytes(b"CHANGED_FIXTURE")
    assert not client.get("/api/molecular/animal-blocks").json()["current"]
