"""Read validated local projections of public sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .analytics import StudyEstimate

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_trials() -> list[dict[str, Any]]:
    path = PROJECT_ROOT / "data" / "processed" / "clinical_trials.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["enrollment"] = (
            int(row["enrollment"]) if row.get("enrollment", "").isdigit() else None
        )
        row["has_results"] = row.get("has_results", "").lower() == "true"
        row["exact_sca3_text_match"] = (
            row.get("exact_sca3_text_match", "").lower() == "true"
        )
    return rows


def load_progression_estimates(primary_only: bool = True) -> list[StudyEstimate]:
    path = PROJECT_ROOT / "configs" / "progression_estimates.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    estimates: list[StudyEstimate] = []
    for row in rows:
        if primary_only and row["primary_inclusion"].lower() != "true":
            continue
        estimates.append(
            StudyEstimate(
                study_id=row["study_id"],
                cohort=row["cohort"],
                effect=float(row["sara_slope"]),
                standard_error=float(row["sara_se"]),
                sample_size=int(row["n_sca3"]) if row["n_sca3"] else None,
                source_url=row["source_url"],
            )
        )
    return estimates


def load_claims() -> dict[str, Any]:
    return _read_json(PROJECT_ROOT / "configs" / "evidence_claims.json")


def load_learning_modules() -> dict[str, Any]:
    return _read_json(PROJECT_ROOT / "configs" / "learning_modules.json")


def load_learning_model() -> dict[str, Any]:
    return _read_json(PROJECT_ROOT / "configs" / "learning_model.json")


def load_evidence_benchmark() -> dict[str, Any]:
    return _read_json(PROJECT_ROOT / "configs" / "evidence_benchmark.json")


def load_evidence_evaluation() -> dict[str, Any]:
    path = PROJECT_ROOT / "artifacts" / "evidence-retrieval-evaluation.json"
    if not path.exists():
        return {"model": "not evaluated", "metrics": {}, "case_results": []}
    return _read_json(path)


def load_learning_evaluation() -> dict[str, Any]:
    path = PROJECT_ROOT / "artifacts" / "research-release-3-validation.json"
    if not path.exists():
        return {"status": "not_run"}
    report = _read_json(path)
    return {"status": "available", "generated_at": report["generated_at"], **report["paired_learning_simulation"]}


def load_source_registry() -> dict[str, Any]:
    return _read_json(PROJECT_ROOT / "configs" / "data_sources.json")


def load_audit() -> dict[str, Any]:
    path = PROJECT_ROOT / "artifacts" / "source-audit.json"
    if not path.exists():
        return {"summary": {"total": 0, "passed": 0, "failed": 0}, "results": []}
    return _read_json(path)


def load_transcriptomics_analysis() -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "processed" / "gse309548_analysis.json"
    if not path.exists():
        return {
            "analysis_status": "not_run",
            "provenance_tier": "PUBLIC_REAL_DERIVED",
            "metrics": {},
            "pca": [],
            "top_signals": [],
        }
    return _with_donor_identity_audit(_read_json(path))


def load_gene_expression_analysis() -> dict[str, Any]:
    path = PROJECT_ROOT / "data" / "processed" / "gse309548_gene_analysis.json"
    if not path.exists():
        return {
            "analysis_status": "not_run",
            "provenance_tier": "PUBLIC_REAL_DERIVED",
            "metrics": {},
            "pca": [],
            "top_signals": [],
        }
    return _with_donor_identity_audit(_read_json(path))


def _with_donor_identity_audit(report: dict[str, Any]) -> dict[str, Any]:
    """Annotate historical results without overwriting the original artifact."""
    warning = (
        "Identity audit 2026-09-15: the eight donor groups were inferred from "
        "genotype, sex, age and PMI; source donor identities are not independently "
        "verified. Results are conditional on that grouping and exploratory only."
    )
    report["identity_audit"] = {"date": "2026-09-15", "status": "unresolved", "detail": warning}
    report["limitations"] = [warning, *[
        item.replace("Eight independent donors", "Eight inferred donor groups")
        for item in report.get("limitations", [])]]
    report["interpretation_boundary"] = warning + " " + report.get("interpretation_boundary", "")
    return report
