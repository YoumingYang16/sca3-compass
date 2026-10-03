"""Create content-addressed snapshots of the registered public real data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data_provenance import DataTier, validate_registry

USER_AGENT = "SCA3-Compass-Ingestion/0.1 (public-data research)"
CTG_ALL_STUDIES_URL = (
    "https://clinicaltrials.gov/api/v2/studies?"
    "query.cond=Spinocerebellar%20Ataxia%20Type%203&pageSize=100&countTotal=true"
)
GEO_METADATA_URLS = {
    "geo_gse309548_metadata": (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
        "acc=GSE309548&targ=all&form=text&view=brief"
    ),
    "geo_gse93713_metadata": (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
        "acc=GSE93713&targ=all&form=text&view=brief"
    ),
}


def fetch(
    url: str, timeout: int = 60, attempts: int = 5
) -> tuple[bytes, dict[str, str | None]]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=timeout) as response:
                body = response.read()
                metadata = {
                    "status": str(response.status),
                    "content_type": response.headers.get("Content-Type"),
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                }
            return body, metadata
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt == attempts - 1:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = (
                float(retry_after)
                if retry_after and retry_after.isdigit()
                else 2**attempt
            )
        except (URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            delay = 2**attempt
        time.sleep(min(delay, 10.0))
    raise RuntimeError(f"unreachable fetch failure: {last_error}")


def store_snapshot(
    raw_root: Path,
    source_id: str,
    url: str,
    body: bytes,
    response_metadata: dict[str, str | None],
    suffix: str,
) -> dict[str, Any]:
    digest = hashlib.sha256(body).hexdigest()
    directory = raw_root / source_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}{suffix}"
    if not path.exists():
        path.write_bytes(body)
    return {
        "source_id": source_id,
        "url": url,
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sha256": digest,
        "bytes": len(body),
        "path": str(path),
        "response": response_metadata,
    }


def _get(mapping: dict[str, Any], *path: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _join(values: Any, key: str | None = None) -> str:
    if not isinstance(values, list):
        return ""
    output: list[str] = []
    for value in values:
        if key and isinstance(value, dict):
            value = value.get(key)
        if value is not None:
            output.append(str(value).strip())
    return " | ".join(item for item in output if item)


SCA3_PATTERN = re.compile(
    r"\bSCA\s*-?\s*3\b|spinocerebellar\s+ataxia\s+(?:type\s+)?3|Machado[- ]Joseph",
    re.IGNORECASE,
)


def normalize_trial(study: dict[str, Any]) -> dict[str, Any]:
    protocol = study.get("protocolSection", {})
    identification = protocol.get("identificationModule", {})
    status = protocol.get("statusModule", {})
    design = protocol.get("designModule", {})
    conditions = protocol.get("conditionsModule", {}).get("conditions", [])
    interventions = protocol.get("armsInterventionsModule", {}).get("interventions", [])
    outcomes = protocol.get("outcomesModule", {})
    sponsors = protocol.get("sponsorCollaboratorsModule", {})
    locations = protocol.get("contactsLocationsModule", {}).get("locations", [])

    intervention_names = _join(interventions, "name")
    intervention_types = _join(interventions, "type")
    primary_outcomes = _join(outcomes.get("primaryOutcomes", []), "measure")
    secondary_outcomes = _join(outcomes.get("secondaryOutcomes", []), "measure")
    countries = sorted(
        {
            str(location.get("country", "")).strip()
            for location in locations
            if location.get("country")
        }
    )
    searchable = " ".join(
        [
            str(identification.get("briefTitle", "")),
            str(identification.get("officialTitle", "")),
            " ".join(str(value) for value in conditions),
        ]
    )
    enrollment = design.get("enrollmentInfo", {}) or {}
    return {
        "nct_id": identification.get("nctId", ""),
        "brief_title": identification.get("briefTitle", ""),
        "official_title": identification.get("officialTitle", ""),
        "overall_status": status.get("overallStatus", ""),
        "study_type": design.get("studyType", ""),
        "phases": _join(design.get("phases", [])),
        "enrollment": enrollment.get("count", ""),
        "enrollment_type": enrollment.get("type", ""),
        "conditions": _join(conditions),
        "intervention_types": intervention_types,
        "intervention_names": intervention_names,
        "primary_outcomes": primary_outcomes,
        "secondary_outcomes": secondary_outcomes,
        "lead_sponsor": _get(sponsors, "leadSponsor", "name", default=""),
        "countries": " | ".join(countries),
        "start_date": _get(status, "startDateStruct", "date", default=""),
        "primary_completion_date": _get(
            status, "primaryCompletionDateStruct", "date", default=""
        ),
        "completion_date": _get(status, "completionDateStruct", "date", default=""),
        "has_results": bool(study.get("hasResults", False)),
        "exact_sca3_text_match": bool(SCA3_PATTERN.search(searchable)),
        "record_url": f"https://clinicaltrials.gov/study/{identification.get('nctId', '')}",
    }


def write_trials_csv(path: Path, studies: list[dict[str, Any]]) -> dict[str, int]:
    rows = [normalize_trial(study) for study in studies]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("ClinicalTrials.gov returned no studies")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "candidate_records": len(rows),
        "exact_text_matches": sum(bool(row["exact_sca3_text_match"]) for row in rows),
        "records_with_results": sum(bool(row["has_results"]) for row in rows),
    }


def run(registry_path: Path, raw_root: Path, processed_root: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assets = validate_registry(registry["sources"])
    if any(asset.tier is not DataTier.PUBLIC_REAL for asset in assets):
        raise ValueError("The public ingestion job accepts PUBLIC_REAL assets only")

    snapshots: list[dict[str, Any]] = []
    ctg_body, ctg_metadata = fetch(CTG_ALL_STUDIES_URL)
    snapshots.append(
        store_snapshot(
            raw_root,
            "clinicaltrials_sca3_api",
            CTG_ALL_STUDIES_URL,
            ctg_body,
            ctg_metadata,
            ".json",
        )
    )
    ctg_payload = json.loads(ctg_body)
    trial_summary = write_trials_csv(
        processed_root / "clinical_trials.csv", ctg_payload.get("studies", [])
    )

    source_by_id = {item["id"]: item for item in registry["sources"]}
    for source_id in (
        "geo_gse309548_human_rnaseq",
        "pmc_china_two_year_cohort",
        "pmc_eurosca_two_year_cohort",
        "pmc_esmi_biomarker_progression",
    ):
        item = source_by_id[source_id]
        body, metadata = fetch(item["url"])
        suffix = ".json" if "BioC_json" in item["url"] else ".txt"
        snapshots.append(
            store_snapshot(raw_root, source_id, item["url"], body, metadata, suffix)
        )
        time.sleep(0.4)

    for source_id, url in GEO_METADATA_URLS.items():
        body, metadata = fetch(url)
        snapshots.append(
            store_snapshot(raw_root, source_id, url, body, metadata, ".soft.txt")
        )
        time.sleep(0.4)

    manifest = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "policy": "PUBLIC_REAL only",
        "trial_summary": trial_summary,
        "snapshots": snapshots,
    }
    manifest_path = raw_root / "ingestion-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry", type=Path, default=Path("configs/data_sources.json")
    )
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    args = parser.parse_args(argv)
    manifest = run(args.registry, args.raw_root, args.processed_root)
    print(json.dumps(manifest["trial_summary"], indent=2))
    print(f"snapshots: {len(manifest['snapshots'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
