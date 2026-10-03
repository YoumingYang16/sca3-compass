"""Read-only research results. No endpoint can unseal or train on holdout data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException

from .molecular_data import PROJECT_ROOT, PROTOCOL, REGISTRY, digest

router = APIRouter(prefix="/api/molecular", tags=["Molecular methods research"])


def load_report(name: str) -> dict:
    path = PROJECT_ROOT / "artifacts" / name
    if not path.exists():
        raise HTTPException(503, detail=f"Research artifact not generated: {name}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise TypeError("Report must be an object")
        return result
    except (OSError, ValueError, TypeError) as exc:
        raise HTTPException(503, detail=f"Research artifact unavailable: {name}") from exc


def artifact_state(name: str) -> dict:
    path = PROJECT_ROOT / "artifacts" / name
    return {"available": path.exists(), "sha256": digest(path) if path.exists() else None}


def matches(path: Path, expected: str | None) -> bool:
    """A missing input is stale, never a successful hash comparison."""
    return bool(expected and path.is_file() and digest(path) == expected)


def qc_is_current(qc: dict) -> bool:
    return matches(PROJECT_ROOT / "artifacts/molecular-data-audit.json", qc.get("acquisition_sha256")) and matches(
        Path(__file__).with_name("molecular_qc.py"), qc.get("source_sha256"))


@router.get("/overview")
def molecular_overview() -> dict:
    audit = load_report("molecular-data-audit.json")
    qc = load_report("molecular-qc.json")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    datasets = []
    profiles = {d["accession"]: d for d in qc["datasets"]}
    for data in audit["datasets"]:
        profile = profiles.get(data["accession"], {})
        datasets.append({"accession": data["accession"], "role": data["role"], "family": data["family"],
            "organism": data["organism"], "title": data.get("metadata", {}).get("series", {}).get("title", [""])[0],
            "samples": data.get("metadata", {}).get("sample_count"), "resolved_units": profile.get("resolved_units"),
            "features": profile.get("features"), "download_status": data["download_status"],
            "qc_status": profile.get("status", "not_run"), "source_url": data["source_url"],
            "issues": profile.get("issues", [])})
    links = []
    for i, left in enumerate(datasets):
        for right in datasets[i + 1:]:
            left_units = {s["unit_id"] for s in profiles.get(left["accession"], {}).get("samples", []) if s.get("unit_id")}
            right_units = {s["unit_id"] for s in profiles.get(right["accession"], {}).get("samples", []) if s.get("unit_id")}
            # IDs may only be compared inside a documented study family.
            if left["family"] == right["family"]:
                links.append({"source": left["accession"], "target": right["accession"],
                    "relationship": "related_study_family", "recorded_shared_unit_ids": len(left_units & right_units) if left_units and right_units else None})
    transport_status = "not_completed"
    if (PROJECT_ROOT / "artifacts/molecular-transport.json").is_file():
        try:
            transport_status = "evaluated_current" if molecular_transport()["current"] else "stale"
        except (HTTPException, ValueError, KeyError, TypeError, OSError):
            transport_status = "unavailable"
    return {"protocol": protocol, "datasets": datasets, "relationships": links,
        "acquisition_summary": audit.get("summary", {}), "qc_summary": qc["summary"],
        "registry_current": audit["registry_sha256"] == digest(REGISTRY),
        "qc_current": qc_is_current(qc),
        "release_gates": [
            {"name": "Registered acquisition", "status": "complete" if audit.get("summary", {}).get("download_complete") == len(datasets) else "incomplete"},
            {"name": "Development expression QC", "status": "scope_qualified", "detail": f"{qc['summary']['matrix_qc_complete']} matrices checked; CEL/FPKM sensitivity sources are excluded from the frozen cerebellar analysis"},
            {"name": "Method implementation", "status": "reference_and_candidate_components"},
            {"name": "Original contribution", "status": "not_established"},
            {"name": "Frozen public-cohort transport", "status": transport_status, "detail": "GSE261670 untreated cerebellum; separate from clinical validation and CE-maxT assumption validation"},
            {"name": "Independent clinical validation", "status": "not_in_scope"},
            {"name": "Paper drafting", "status": "await_user_before_writing"}],
        "artifacts": {name: artifact_state(name) for name in ["molecular-benchmark-development.json", "molecular-benchmark-validation.json", "molecular-shrinkage.json"]},
        "boundary": "Reproducible methods and public-cohort research. Completed experiments do not themselves establish methodological originality or clinical validity."}


@router.get("/datasets/{accession}")
def molecular_dataset(accession: str) -> dict:
    audit = load_report("molecular-data-audit.json")
    found = next((d for d in audit["datasets"] if d["accession"] == accession), None)
    if found is None:
        raise HTTPException(404, detail="Unknown registered dataset")
    qc = load_report("molecular-qc.json")
    return {"registration": found, "qc": next((d for d in qc["datasets"] if d["accession"] == accession), None)}


@router.get("/benchmark")
def molecular_benchmark(phase: Literal["development", "validation"] = "validation") -> dict:
    result = load_report(f"molecular-benchmark-{phase}.json")
    expected_sources = {"molecular_benchmark.py", "molecular_methods.py"}
    sources = result.get("source_sha256", {})
    result["current"] = matches(PROTOCOL, result.get("protocol_sha256")) and set(sources) == expected_sources and all(
        matches(Path(__file__).with_name(name), expected) for name, expected in sources.items())
    # The public view omits thousands of repetition-level values, retained in the
    # downloadable artifact and local run archive for reproducible inspection.
    result["rows"] = [{key: value for key, value in row.items() if not key.endswith("_by_repetition")} for row in result["rows"]]
    return result


@router.get("/envelope")
def molecular_envelope(phase: Literal["development", "validation"] = "validation") -> dict:
    result = load_report(f"molecular-envelope-{phase}.json")
    fp = result.get("effective", {}).get("fingerprint", {})
    sources = fp.get("sources", {})
    expected = {"molecular_envelope.py", "molecular_envelope_benchmark.py", "molecular_methods.py", "molecular_benchmark.py"}
    result["current"] = (matches(PROJECT_ROOT / "configs/molecular_envelope_protocol.json", fp.get("protocol_sha256"))
        and set(sources) == expected and all(matches(Path(__file__).with_name(name), sha) for name, sha in sources.items()))
    for case in result["scenarios"]:
        case["rows"] = [{k: v for k, v in row.items() if not k.endswith("_by_repetition")} for row in case["rows"]]
    return result


@router.get("/transport")
def molecular_transport() -> dict:
    from .molecular_transport import fingerprint

    result = load_report("molecular-transport.json")
    freeze = load_report("molecular-transport-freeze.json")
    try:
        result["current"] = (freeze.get("fingerprint") == fingerprint()
            and matches(PROJECT_ROOT / "artifacts/molecular-transport-freeze.json", result.get("freeze_sha256"))
            and matches(PROJECT_ROOT / result.get("expression_path", ""), result.get("expression_sha256")))
    except OSError:
        result["current"] = False
    result["nominations_frozen_at"] = freeze.get("created_at")
    result["nominations"] = freeze.get("nominations")
    return result


@router.get("/animal-blocks")
def molecular_animal_blocks() -> dict:
    result = load_report("molecular-block-validation.json")
    result["current"] = (matches(PROJECT_ROOT / "configs/molecular_block_validation.json", result.get("protocol_sha256"))
        and matches(Path(__file__).with_name("molecular_block_validation.py"), result.get("source_sha256"))
        and matches(Path(__file__).with_name("molecular_shrinkage.py"), result.get("shrinkage_source_sha256"))
        and matches(PROJECT_ROOT / "artifacts/molecular-qc.json", result.get("qc_sha256"))
        and matches(PROJECT_ROOT / "data/processed/molecular/GSE107958-counts.npz", result.get("input_sha256")))
    return result


@router.get("/shrinkage")
def molecular_shrinkage() -> dict:
    result = load_report("molecular-shrinkage.json")
    qc = load_report("molecular-qc.json")
    profile = next((d for d in qc["datasets"] if d["accession"] == result["accession"]), {})
    result["current"] = (matches(Path(__file__).with_name("molecular_shrinkage.py"), result.get("source_sha256"))
        and matches(PROJECT_ROOT / "configs/molecular_shrinkage_protocol.json", result.get("protocol_sha256"))
        and matches(PROJECT_ROOT / "artifacts/molecular-qc.json", result.get("qc_sha256"))
        and qc_is_current(qc)
        and result.get("input_sha256") == profile.get("derived_sha256")
        and matches(PROJECT_ROOT / profile.get("derived_path", ""), profile.get("derived_sha256"))
        and matches(PROJECT_ROOT / "data/processed/molecular/GSE107958-shrinkage.npz", result.get("result_sha256")))
    return result
