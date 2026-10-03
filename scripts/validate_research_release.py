"""Recompute local release checks and a real-public-input design experiment.

Run from the project root with .venv/Scripts/python.exe.
This is a technical validation artifact, not external scientific validation.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sca3_compass.design_assurance import AssuranceRequest, run_design_assurance
from sca3_compass.evidence_engine import evaluate_benchmark
from sca3_compass.experiment_store import ExperimentStore
from sca3_compass.fhir_service import (
    build_research_bundle,
    capability_statement,
    validate_resource,
    validation_passed,
)
from sca3_compass.learning_engine import simulate_policy_comparison
from sca3_compass.repository import (
    load_claims,
    load_evidence_benchmark,
    load_gene_expression_analysis,
    load_learning_model,
    load_progression_estimates,
    load_trials,
)

ROOT = Path(__file__).resolve().parents[1]


def check_command(arguments: list[str]) -> dict:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env={**os.environ, "SCA3_RUN_API_TESTS": "1", "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    return {
        "exit_code": completed.returncode,
        "output": completed.stdout + completed.stderr,
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    print("Checking Python tests and lint...", flush=True)
    # A fresh, workspace-owned fixture path; pytest may clear its own basetemp.
    temporary_root = Path(tempfile.mkdtemp(prefix="verification-", dir=ROOT / ".run"))
    if not temporary_root.resolve().is_relative_to((ROOT / ".run").resolve()):
        raise RuntimeError("Verification temporary path escaped the workspace")
    tests = check_command(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", str(temporary_root / "pytest")]
    )
    lint = check_command(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "src",
            "tests",
            "scripts/validate_research_release.py",
        ]
    )
    if tests["exit_code"] or lint["exit_code"]:
        print(tests["output"] + lint["output"])
        return 1
    print("Computing 100,000-draw public-input design grid...", flush=True)
    request = AssuranceRequest(simulations=100_000, calibration_simulations=500_000)
    design = run_design_assurance(load_progression_estimates(), request)
    experiment = ExperimentStore(ROOT / "data/runtime/experiments.sqlite3").save(design)
    trials = [row for row in load_trials() if row["exact_sca3_text_match"]]
    bundle = build_research_bundle(trials, load_claims()["claims"])
    bundle_outcome = validate_resource(bundle)
    capability_outcome = validate_resource(capability_statement())
    genes = load_gene_expression_analysis()
    print("Checking paired learning simulation and evidence baseline...", flush=True)
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "release": "3.0.0-research",
        "scope": "Local numerical, schema and software verification; not an external replication, clinical certification, admissions assessment or human learning evaluation.",
        "python_tests": tests,
        "python_lint": lint,
        "design_experiment": experiment,
        "fhir": {
            "bundle_valid": validation_passed(bundle_outcome),
            "capability_valid": validation_passed(capability_outcome),
            "entries": len(bundle["entry"]),
            "research_studies": len(trials),
            "outcome": bundle_outcome,
        },
        "gene_analysis": {
            "research_question": genes["research_question"],
            "metrics": genes["metrics"],
            "interpretation_boundary": genes["interpretation_boundary"],
        },
        "paired_learning_simulation": simulate_policy_comparison(
            load_learning_model(), learners=1000, steps=12, seed=202709
        ),
        "evidence_baseline": evaluate_benchmark(load_evidence_benchmark()),
        "frontend": "Run the existing build script separately; no browser-QA certification is inferred from these checks.",
    }
    output = ROOT / "artifacts/research-release-3-validation.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(output),
                "experiment_id": experiment["run_id"],
                "null_type1": design["calibration"]["null_type1"],
                "naive_type1": design["calibration"]["naive_repeated_testing_type1"],
                "quadrature_error": design["calibration"][
                    "doubled_resolution_alpha_discrepancy"
                ],
                "fhir_valid": report["fhir"]["bundle_valid"],
                "learning_paired_difference": report["paired_learning_simulation"][
                    "paired_difference_intervals"
                ]["expected_posttest_accuracy"],
            },
            indent=2,
        )
    )
    return (
        0
        if report["fhir"]["bundle_valid"] and report["fhir"]["capability_valid"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
