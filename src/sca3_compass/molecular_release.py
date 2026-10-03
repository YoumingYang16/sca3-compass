"""Check real release files and computation integrity, not scientific originality."""
from __future__ import annotations

import itertools
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .molecular_api import (
    load_report,
    matches,
    molecular_animal_blocks,
    molecular_benchmark,
    molecular_envelope,
    molecular_overview,
    molecular_shrinkage,
    molecular_transport,
)
from .molecular_data import PROJECT_ROOT, digest, write_json


def extended_checks(record, qc: dict) -> None:
    """Audit completed experiments without regenerating frozen inputs."""
    envelope_freeze = load_report("molecular-envelope-freeze.json")
    phases = {}
    for phase, repetitions in [("development", 100), ("validation", 600)]:
        report = load_report(f"molecular-envelope-{phase}.json")
        phases[phase] = report
        record(f"envelope {phase}: current sources", molecular_envelope(phase)["current"])
        record(f"envelope {phase}: simulation not patient data", report["provenance"] == "SIMULATION_NOT_PATIENT_DATA")
        record(f"envelope {phase}: same frozen fingerprint", report["effective"]["fingerprint"] == envelope_freeze["fingerprint"])
        record(f"envelope {phase}: full non-smoke experiment", len(report["scenarios"]) == 37 and not report["effective"]["smoke"])
        methods = set(report["effective"]["protocol"]["methods"])
        for case in report["scenarios"]:
            ok = len(case["rows"]) == 8 and {r["method"] for r in case["rows"]} == methods
            for row in case["rows"]:
                fdp = np.asarray(row["fdp_by_repetition"])
                power = np.asarray(row["power_by_repetition"])
                ok &= (len(fdp) == len(power) == repetitions and np.isfinite(fdp).all()
                       and np.isfinite(power).all() and ((fdp >= 0) & (fdp <= 1)).all()
                       and ((power >= 0) & (power <= 1)).all()
                       and np.isclose(fdp.mean(), row["fdr"]["mean"], atol=1e-14)
                       and np.isclose(power.mean(), row["power"]["mean"], atol=1e-14))
            record(f"envelope {phase}: {case['scenario']} complete numerical record", ok)
    protocol = phases["validation"]["effective"]["protocol"]
    record("envelope: independent seed schedules", protocol["seed_development"] != protocol["seed_validation"])
    record("envelope: freeze before validation output", datetime.fromisoformat(envelope_freeze["created_at"]) < datetime.fromisoformat(phases["validation"]["created_at"]))
    stress = {s["scenario"]: s for s in phases["validation"]["scenarios"]}
    record("envelope: out-of-assumption failures retained", all(not stress[name]["candidate_assumptions"]
        for name in ["heavy_tail_both", "calibration_covariance_shift", "dependent_calibration_rows"]))
    parity = load_report("molecular-efilter-conformance.json")
    record("eFilter: author and port fingerprints", parity["passed"] and len(parity["cases"]) == 20
        and sum(r["hypotheses"] for r in parity["cases"]) == 5230
        and matches(PROJECT_ROOT / "external/efilter/funcs.R", parity["author_function_sha256"])
        and matches(Path(__file__).with_name("molecular_envelope.py"), parity["port_sha256"])
        and matches(PROJECT_ROOT / "scripts/check_efilter_conformance.py", parity["runner_sha256"]))
    record("eFilter: license retained", (PROJECT_ROOT / parity["license"]).is_file())
    for case in parity["cases"]:
        record(f"eFilter: case {case['case']} input/output hashes",
            matches(PROJECT_ROOT / f"artifacts/efilter-conformance/input-{case['case']}.txt", case["input_sha256"])
            and matches(PROJECT_ROOT / f"artifacts/efilter-conformance/output-{case['case']}.txt", case["output_sha256"]))
    transport = molecular_transport()
    frozen = load_report("molecular-transport-freeze.json")
    access = load_report("molecular-transport-access.json")
    acquisition = load_report("molecular-transport-acquisition.json")
    freeze_sha = digest(PROJECT_ROOT / "artifacts/molecular-transport-freeze.json")
    record("transport: all frozen input hashes current", transport["current"])
    record("transport: freeze before expression access", not frozen["expression_holdout_read"]
        and datetime.fromisoformat(frozen["created_at"]) < datetime.fromisoformat(access["first_expression_access_at"])
        and access["freeze_sha256"] == freeze_sha == transport["freeze_sha256"])
    for previous in ["pre-http-fix", "pre-url-fix"]:
        old = load_report(f"molecular-transport-freeze-{previous}.json")
        record(f"transport: acquisition amendment {previous} did not change nominations", old["nominations"] == frozen["nominations"])
    record("transport: all prespecified strategies retained", len(frozen["nominations"]) == 4
        and all(len(v) == 50 for v in frozen["nominations"].values())
        and {r["strategy"] for r in transport["strategy_results"]} == set(frozen["nominations"]))
    record("transport: exactly 12 selected untreated samples", len(acquisition["files"]) == 12
        and {r["sample_id"] for r in acquisition["files"]} == {f"GSM{i}" for i in range(8148389,8148401)}
        and transport["sample_count"] == 12 and transport["age_weeks"] == 65)
    for receipt in acquisition["files"]:
        path = PROJECT_ROOT / receipt["path"]
        record(f"transport: {receipt['sample_id']} raw hash / size / chronology", matches(path, receipt["sha256"])
            and path.stat().st_size == receipt["bytes"]
            and datetime.fromisoformat(receipt["retrieved_at"]) > datetime.fromisoformat(frozen["created_at"]))
    record("transport: annotation coverage gate", len(transport["qc"]["mapping"]) == 12
        and all(r["mapped_fraction"] >= .99 and abs(r["total_tpm"]-1e6) < 1 for r in transport["qc"]["mapping"]))
    record("transport: exact enumeration and minimum p", transport["permutation_assignments"] == 924
        and np.isclose(transport["minimum_attainable_p"], 1/924))
    with np.load(PROJECT_ROOT / transport["expression_path"], allow_pickle=False) as data:
        record("transport: no missing or synthetic expression", np.isfinite(data["expression"]).all()
            and (data["expression"] >= 0).all() and data["expression"].shape[1] == 12
            and len(set(data["features"].tolist())) == len(data["features"]))
    genes = transport["gene_results"]
    record("transport: missing nominations retained with p=1", all(g["signed_p"] == 1
        and g["BY_q_union_signed_family"] == 1 for g in genes if g.get("missing")))
    record("transport: signed union and BY count", len(genes) == transport["signed_gene_family_size"] == 121
        and sum(g["BY_q_union_signed_family"] <= .05 for g in genes) == transport["signed_genes_BY_005"])
    reanalysis = load_report("molecular-transport-reanalysis.json")
    record("transport: independent R reconstruction current", reanalysis["passed"]
        and len(reanalysis["checks"]) == 5 and all(c["passed"] for c in reanalysis["checks"])
        and matches(PROJECT_ROOT / "artifacts/molecular-transport.json", reanalysis["report_sha256"])
        and matches(PROJECT_ROOT / "scripts/check_transport_reanalysis.py", reanalysis["python_runner_sha256"])
        and matches(PROJECT_ROOT / "scripts/transport_crosscheck.R", reanalysis["r_runner_sha256"])
        and matches(PROJECT_ROOT / "scripts/transport_gene_crosscheck.R", reanalysis["r_gene_runner_sha256"]))
    block = molecular_animal_blocks()
    record("animal blocks: current inputs and source", block["current"])
    record("animal blocks: all 96 fits converged", block["all_fits_converged"] and len(block["folds"]) == 48
        and all(len(r["convergence"]) == 2 and all(c["converged"] and c["kkt"] < 1e-4
            for c in r["convergence"].values()) for r in block["folds"]))
    profile = next(d for d in qc["datasets"] if d["accession"] == "GSE107958")
    cases = {s["unit_id"] for s in profile["samples"] if s["genotype"] == "SCA3 transgenic"}
    controls = {s["unit_id"] for s in profile["samples"] if s["genotype"] == "wildtype"}
    record("animal blocks: all 6 by 8 held-out pairs exactly once", len(cases) == 6 and len(controls) == 8
        and {(r["held_disease"],r["held_control"]) for r in block["folds"]} == set(itertools.product(cases,controls)))
    for fold in block["folds"]:
        held = {fold["held_disease"], fold["held_control"]}
        train = [s for s in profile["samples"] if s["unit_id"] not in held]
        test = [s for s in profile["samples"] if s["unit_id"] in held]
        record(f"animal blocks: {fold['held_disease']}/{fold['held_control']} whole-animal partition",
            len(train) == fold["training_samples"] and len(test) == fold["held_samples"]
            and len({s["unit_id"] for s in train}) == fold["training_animals"] == 12)
    for row in block["summaries"]:
        record(f"animal blocks: {row['method']} aggregate recomputed", np.isclose(row["mean_fold_mse"],
            np.mean([f["losses"][row["method"]] for f in block["folds"]]), atol=1e-14))


def validate_release() -> dict:
    checks = []
    def record(name, passed, detail=""):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
    overview = molecular_overview()
    audit = load_report("molecular-data-audit.json")
    qc = load_report("molecular-qc.json")
    record("registry_current", overview["registry_current"])
    record("qc_current", overview["qc_current"])
    for dataset in audit["datasets"]:
        record(f"{dataset['accession']}: acquisition", dataset["download_status"] == "complete")
        for file in dataset["files"]:
            record(f"{dataset['accession']}: {file['kind']} hash", matches(PROJECT_ROOT / file["path"], file["sha256"]))
        profile = next(row for row in qc["datasets"] if row["accession"] == dataset["accession"])
        if dataset["role"] == "sealed_candidate":
            record(f"{dataset['accession']}: expression sealed",
                profile["status"] == "sealed_not_parsed" and not profile.get("samples")
                and not (PROJECT_ROOT / f"data/processed/molecular/{dataset['accession']}-counts.npz").exists())
        if "derived_path" in profile:
            record(f"{dataset['accession']}: processed hash", matches(PROJECT_ROOT / profile["derived_path"], profile["derived_sha256"]))
    phases = {}
    for phase in ("development", "validation"):
        report = molecular_benchmark(phase)
        phases[phase] = report
        record(f"{phase}: current", report["current"])
        settings = report["effective_settings"]["settings"]
        record(f"{phase}: complete reference comparison", len(report["rows"]) == 5 * len(settings["scenarios"]))
        record(f"{phase}: synthetic provenance", report["provenance"] == "SIMULATION_NOT_PATIENT_DATA")
        record(f"{phase}: no paper-ready claim", report["paper_ready"] is False and report["novelty_status"] == "not_established")
        record(f"{phase}: nondegenerate MC bounds", all(row["fdr"]["ci95"][1] > row["fdr"]["ci95"][0] for row in report["rows"]))
    record("separate simulation seeds", phases["development"]["effective_settings"]["seed"] != phases["validation"]["effective_settings"]["seed"])
    record("same frozen simulation sources", phases["development"]["source_sha256"] == phases["validation"]["source_sha256"])
    model = molecular_shrinkage()
    record("multiregion input/output provenance current", model["current"])
    record("real model converged", model["em"]["converged"])
    record("synthetic reference fits converged", all(row.get("training_converged", True) for row in model["simulation"]["rows"]))
    record("regions not treated as independent studies", model["regions_are_not_independent_studies"])
    record("external validation not overstated", qc["summary"]["external_validation_certified"] == 0)
    extended_checks(record, qc)
    report_paths = [PROJECT_ROOT / "artifacts" / name for name in (
        "molecular-data-audit.json", "molecular-qc.json", "molecular-benchmark-development.json",
        "molecular-benchmark-validation.json", "molecular-shrinkage.json",
        "molecular-envelope-development.json", "molecular-envelope-validation.json", "molecular-envelope-freeze.json",
        "molecular-efilter-conformance.json", "molecular-transport-freeze.json", "molecular-transport.json",
        "molecular-transport-access.json", "molecular-transport-acquisition.json", "molecular-transport-reanalysis.json",
        "molecular-block-validation.json")]
    sources = sorted(Path(__file__).parent.glob("molecular*.py"))
    return {"created_at": datetime.now(UTC).isoformat(), "python": platform.python_version(),
            "passed": all(check["passed"] for check in checks), "checks": checks,
            "artifacts_sha256": {path.name: digest(path) for path in report_paths},
            "sources_sha256": {path.name: digest(path) for path in sources},
            "scientific_readiness": {"originality_established": False,
                                     "frozen_public_cohort_transport_completed": True,
                                     "independent_clinical_validation": False,
                                     "scoped_experiments_completed": True,
                                     "pre_manuscript_review_bundle": True,
                                     "publication_readiness_certified": False, "paper_drafting": "await_user"},
            "scope": "Integrity, frozen chronology, numerical parity and completed scoped experiments. Not a clinical, novelty or publication certification."}


def main() -> None:
    result = validate_release()
    write_json(PROJECT_ROOT / "artifacts/molecular-release-validation.json", result)
    print(json.dumps({"checks": len(result["checks"]), "passed": result["passed"], "scientific_readiness": result["scientific_readiness"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
