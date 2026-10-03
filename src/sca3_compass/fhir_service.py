"""FHIR research-resource projection and focused conformance validation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fhir.resources import get_fhir_model_class
from pydantic import ValidationError

FHIR_VERSION = "5.0.0"
BASE_URL = "http://127.0.0.1:8000/api/fhir"

PUBLICATION_STATUSES = {"draft", "active", "retired", "unknown"}


def capability_statement() -> dict[str, Any]:
    return {
        "resourceType": "CapabilityStatement",
        "id": "sca3-research-workbench",
        "url": f"{BASE_URL}/metadata",
        "version": "3.0.0",
        "name": "SCA3ResearchWorkbenchCapabilityStatement",
        "status": "active",
        "experimental": True,
        "date": "2026-09-15",
        "kind": "instance",
        "fhirVersion": FHIR_VERSION,
        "format": ["json"],
        "implementation": {
            "description": "Research-only SCA3 evidence interoperability service",
            "url": BASE_URL,
        },
        "rest": [
            {
                "mode": "server",
                "security": {
                    "cors": True,
                    "description": "No patient clinical service; public research resources only.",
                },
                "resource": [
                    {
                        "type": "ResearchStudy",
                        "interaction": [{"code": "read"}, {"code": "search-type"}],
                        "searchParam": [
                            {"name": "status", "type": "token"},
                            {"name": "identifier", "type": "token"},
                        ],
                    },
                    {
                        "type": "Evidence",
                        "interaction": [{"code": "read"}, {"code": "search-type"}],
                    },
                    {"type": "Provenance", "interaction": [{"code": "read"}]},
                ],
                "operation": [
                    {
                        "name": "validate",
                        "definition": "http://hl7.org/fhir/OperationDefinition/Resource-validate",
                    }
                ],
            }
        ],
    }


def trial_to_research_study(row: dict[str, Any]) -> dict[str, Any]:
    nct_id = row["nct_id"]
    return {
        "resourceType": "ResearchStudy",
        "id": nct_id.lower(),
        "meta": {
            "source": row["record_url"],
            "tag": [
                {
                    "system": "https://sca3-compass.local/provenance-tier",
                    "code": "PUBLIC_REAL",
                }
            ],
        },
        "identifier": [
            {"system": "https://clinicaltrials.gov", "value": nct_id}
        ],
        "title": row["brief_title"],
        # R5 status describes publication of the resource, NOT trial progress.
        "status": "active",
        "progressStatus": [{
            "state": {"coding": [{
                "system": "https://clinicaltrials.gov/overall-status",
                "code": row["overall_status"],
            }]},
            "actual": True,
        }],
        "phase": {
            "coding": [
                {
                    "system": "https://clinicaltrials.gov/phase",
                    "code": phase.lower().replace(" ", "-"),
                }
                for phase in row["phases"].split(" | ")
                if phase
            ] or [{"system": "http://hl7.org/fhir/research-study-phase", "code": "n-a"}]
        },
        "condition": [
            {"text": value}
            for value in row["conditions"].split(" | ")
            if value
        ],
        "descriptionSummary": row["official_title"] or row["brief_title"],
        "note": [
            {
                "text": (
                    "PUBLIC_REAL registry projection; not a patient record. "
                    f"Registry enrollment: {row['enrollment']}. Enrollment type was not "
                    "retained in the local projection, so target/actual is not inferred."
                )
            }
        ],
    }


def provenance_for_resource(resource: dict[str, Any]) -> dict[str, Any]:
    source = resource.get("meta", {}).get("source")
    return {
        "resourceType": "Provenance",
        "id": f"prov-{resource['id']}",
        "target": [{"reference": f"{resource['resourceType']}/{resource['id']}"}],
        "recorded": datetime.now(UTC).isoformat(),
        "activity": {
            "coding": [
                {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-DataOperation",
                    "code": "TRANSFORM",
                }
            ]
        },
        "agent": [
            {
                "type": {"text": "assembler"},
                "who": {"display": "SCA3 Research Workbench deterministic mapper"},
            }
        ],
        "entity": [
            {
                "role": "source",
                "what": {"reference": source, "display": "PUBLIC_REAL source record"},
            }
        ],
    }


def claim_to_evidence(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "resourceType": "Evidence",
        "id": claim["id"],
        "meta": {
            "source": claim["source_url"],
            "tag": [
                {
                    "system": "https://sca3-compass.local/provenance-tier",
                    "code": claim["provenance_tier"],
                }
            ],
        },
        "url": f"{BASE_URL}/Evidence/{claim['id']}",
        "status": "active",
        "title": claim["title"],
        "description": claim["plain_language"],
        "variableDefinition": [{
            "variableRole": {"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/variable-role",
                "code": "population",
            }]},
            "description": "Population-level SCA3 research; see source for eligibility and unit of analysis.",
        }],
        "note": [{"text": claim["boundary"]}],
        "relatedArtifact": [
            {
                "type": "citation",
                "label": claim["source_title"],
                "document": {"url": claim["source_url"]},
            }
        ],
    }


def build_research_bundle(
    trials: list[dict[str, Any]], claims: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    resources: list[dict[str, Any]] = []
    for row in trials:
        study = trial_to_research_study(row)
        resources.extend([study, provenance_for_resource(study)])
    for claim in claims or []:
        evidence = claim_to_evidence(claim)
        resources.extend([evidence, provenance_for_resource(evidence)])
    entries = [
        {
            "fullUrl": f"{BASE_URL}/{resource['resourceType']}/{resource['id']}",
            "resource": resource,
        }
        for resource in resources
    ]
    return {
        "resourceType": "Bundle",
        "id": "sca3-public-research-evidence",
        "type": "collection",
        "timestamp": datetime.now(UTC).isoformat(),
        "meta": {
            "tag": [
                {
                    "system": "https://sca3-compass.local/provenance-tier",
                    "code": "PUBLIC_REAL",
                }
            ]
        },
        "entry": entries,
    }


def validate_resource(resource: Any) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []

    def issue(severity: str, code: str, diagnostics: str, expression: str) -> None:
        issues.append(
            {
                "severity": severity,
                "code": code,
                "diagnostics": diagnostics,
                "expression": [expression],
            }
        )

    if not isinstance(resource, dict):
        issue("error", "structure", "Resource must be a JSON object", "Resource")
    else:
        resource_type = resource.get("resourceType")
        supported = {"ResearchStudy", "Evidence", "Provenance", "Bundle", "CapabilityStatement"}
        if resource_type in supported:
            try:
                get_fhir_model_class(resource_type).model_validate(resource)
            except ValidationError as exc:
                for error in exc.errors(include_input=False, include_url=False):
                    expression = ".".join(str(part) for part in error["loc"])
                    issue("error", "structure", error["msg"], f"{resource_type}.{expression}")
        if not resource_type:
            issue("error", "required", "resourceType is required", "Resource.resourceType")
        elif resource_type == "ResearchStudy":
            for field in ("id", "title", "status", "identifier"):
                if not resource.get(field):
                    issue("error", "required", f"{field} is required", f"ResearchStudy.{field}")
            if resource.get("status") not in PUBLICATION_STATUSES:
                issue("error", "value", "R5 publication status must be draft, active, retired or unknown", "ResearchStudy.status")
        elif resource_type == "Evidence":
            for field in ("id", "status", "title"):
                if not resource.get(field):
                    issue("error", "required", f"{field} is required", f"Evidence.{field}")
            if resource.get("status") not in PUBLICATION_STATUSES:
                issue("error", "value", "Invalid publication status", "Evidence.status")
        elif resource_type == "Provenance":
            for field in ("id", "target", "recorded", "agent", "entity"):
                if not resource.get(field):
                    issue("error", "required", f"{field} is required", f"Provenance.{field}")
        elif resource_type == "Bundle":
            if resource.get("type") not in {"collection", "searchset", "batch", "transaction"}:
                issue("error", "value", "Unsupported Bundle type", "Bundle.type")
            if resource.get("type") not in {"searchset", "history"} and "total" in resource:
                issue("error", "invariant", "bdl-1: total is only allowed on searchset/history", "Bundle.total")
            entries = resource.get("entry", [])
            for index, entry in enumerate(entries if isinstance(entries, list) else []):
                if not isinstance(entry, dict):
                    issue("error", "structure", "Bundle entry must be an object", f"Bundle.entry[{index}]")
                    continue
                nested = validate_resource(entry.get("resource"))
                for nested_issue in nested.get("issue", []):
                    if nested_issue["severity"] == "information":
                        continue
                    nested_issue = dict(nested_issue)
                    nested_issue["expression"] = [
                        f"Bundle.entry[{index}].{nested_issue.get('expression', ['Resource'])[0]}"
                    ]
                    issues.append(nested_issue)
        elif resource_type != "CapabilityStatement":
            issue("error", "not-supported", f"Validation not implemented for {resource_type}", "Resource.resourceType")
    if not issues:
        issues.append(
            {
                "severity": "information",
                "code": "informational",
                "diagnostics": (
                    "Passed fhir.resources 8.1.0 R5 structural validation and local checks. "
                    "Not an HL7 certification: external references, full terminology and all FHIRPath invariants are not validated."
                ),
            }
        )
    return {
        "resourceType": "OperationOutcome",
        "issue": issues,
    }


def validation_passed(outcome: dict[str, Any]) -> bool:
    return not any(item["severity"] in {"error", "fatal"} for item in outcome["issue"])
