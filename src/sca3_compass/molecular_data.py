"""Public molecular acquisition and metadata audit, with sealed-expression guards."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW = PROJECT_ROOT / "data/raw/molecular"
REPORT = PROJECT_ROOT / "artifacts/molecular-data-audit.json"
REGISTRY = PROJECT_ROOT / "configs/molecular_datasets.json"
PROTOCOL = PROJECT_ROOT / "configs/molecular_protocol.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def geo_base(accession: str) -> str:
    if not re.fullmatch(r"GSE\d+", accession):
        raise ValueError("Invalid GEO accession")
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{accession[:-3]}nnn/{accession}"


def acquire(url: str, target: Path, maximum: int = 300_000_000) -> dict:
    import httpx  # Acquisition dependency; offline matrix QC does not require it.

    if urlparse(url).hostname != "ftp.ncbi.nlm.nih.gov":
        raise ValueError("Molecular ingestion is restricted to the registered NCBI host")
    target.parent.mkdir(parents=True, exist_ok=True)
    receipt = target.with_suffix(target.suffix + ".receipt.json")
    if target.exists():
        if not receipt.exists():
            raise ValueError(f"Unreceipted existing file; refusing to overwrite {target.name}")
        previous = json.loads(receipt.read_text(encoding="utf-8"))
        if previous["sha256"] != digest(target) or previous["url"] != url:
            raise ValueError(f"Cache integrity failure: {target.name}")
        return {**previous, "cache_verified": True}
    temporary = target.with_suffix(target.suffix + ".partial")
    for attempt in range(3):
        try:
            with httpx.Client(timeout=45, follow_redirects=True) as client:  # noqa: SIM117
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    length = int(response.headers.get("content-length", 0))
                    if length > maximum:
                        raise ValueError("Source exceeds registered acquisition size limit")
                    size = 0
                    with temporary.open("wb") as handle:
                        for block in response.iter_bytes(1024 * 1024):
                            size += len(block)
                            if size > maximum:
                                raise ValueError("Source exceeds acquisition size limit")
                            handle.write(block)
                    # Content-Length describes HTTP payload bytes, whereas
                    # iter_bytes() decodes Content-Encoding (e.g. gzip). Compare
                    # the transport counter, not the decoded local file size.
                    transferred = response.num_bytes_downloaded
                    if not size or (length and transferred != length):
                        raise ValueError("Incomplete or empty transfer")
                    entry = {"url": url, "path": str(target.relative_to(PROJECT_ROOT)),
                             "bytes": size, "sha256": digest(temporary),
                             "http_transfer_bytes": transferred,
                             "http_content_encoding": response.headers.get("content-encoding"),
                             "retrieved_at": datetime.now(UTC).isoformat(),
                             "last_modified": response.headers.get("last-modified"),
                             "source_license": "NCBI public access; redistribution terms not separately certified",
                             "provenance": "PUBLIC_REAL", "cache_verified": False}
            os.replace(temporary, target)
            write_json(receipt, entry)
            return entry
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Acquisition unexpectedly exhausted")


def parse_matrix_metadata(path: Path) -> dict:
    series: dict[str, list[str]] = {}
    sample_rows: dict[str, list[list[str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as handle:
        for line in handle:
            if line.startswith("!series_matrix_table_begin"):
                break  # Do not consume the expression matrix, including sealed candidates.
            row = next(csv.reader([line], delimiter="\t"))
            if not row:
                continue
            key = row[0]
            if key.startswith("!Series_"):
                series.setdefault(key[8:], []).extend(row[1:])
            elif key.startswith("!Sample_"):
                sample_rows.setdefault(key[8:], []).append(row[1:])
    ids = sample_rows.get("geo_accession", [[]])[0]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Missing or duplicate GEO sample identifiers")
    for key, rows in sample_rows.items():
        if any(len(row) != len(ids) for row in rows):
            raise ValueError(f"Metadata column mismatch: {key}")
    samples = []
    for i, identifier in enumerate(ids):
        sample = {"sample_id": identifier}
        for key in ("title", "source_name_ch1", "organism_ch1", "characteristics_ch1", "description", "data_processing"):
            sample[key] = [row[i] for row in sample_rows.get(key, [])]
        sample["biological_unit_id"] = None
        sample["biological_unit_status"] = "not_manually_resolved"
        samples.append(sample)
    return {"series": series, "samples": samples, "sample_count": len(ids),
            "independent_units": None, "independence_status": "requires_sample_level_audit"}


def assert_expression_access(dataset: dict) -> None:
    if dataset["role"] == "sealed_candidate":
        raise PermissionError("Expression is sealed: a separate versioned release plan is required")


def inspect_archive(path: Path, dataset: dict) -> dict:
    assert_expression_access(dataset)
    if not path.name.endswith(".tar"):
        return {"status": "format_specific_qc_pending"}
    with tarfile.open(path) as archive:
        members = [m for m in archive if m.isfile()]
        if any(Path(m.name).is_absolute() or ".." in Path(m.name).parts for m in members):
            raise ValueError("Unsafe archive member")
        return {"status": "archive_index_verified", "file_count": len(members),
                "members": [m.name for m in members], "uncompressed_bytes": sum(m.size for m in members)}


def ingest_one(dataset: dict) -> dict:
    import httpx

    accession = dataset["accession"]
    result = {**dataset, "source_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}",
              "files": [], "errors": [], "scientific_eligibility": "pending", "expression_inspected": False}
    base = geo_base(accession)
    target = RAW / accession
    try:
        if dataset.get("local_expression"):
            path = PROJECT_ROOT / dataset["local_expression"]
            result["files"].append({"kind": "expression", "path": dataset["local_expression"],
                                    "sha256": digest(path), "bytes": path.stat().st_size,
                                    "provenance": "PUBLIC_REAL", "status": "existing_verified"})
        else:
            path = target / dataset["expression"]
            result["files"].append({"kind": "expression", **acquire(f"{base}/suppl/{dataset['expression']}", path)})
        if dataset["role"] == "sealed_candidate":
            result["expression_qc"] = {"status": "sealed_not_parsed"}
        else:
            result["expression_qc"] = inspect_archive(path, dataset)
            result["expression_inspected"] = result["expression_qc"]["status"] == "archive_index_verified"
    except (OSError, ValueError, httpx.HTTPError, tarfile.TarError) as exc:
        result["errors"].append({"stage": "expression", "message": str(exc)})
    try:
        metadata = target / f"{accession}_series_matrix.txt.gz"
        result["files"].append({"kind": "metadata", **acquire(f"{base}/matrix/{metadata.name}", metadata)})
        result["metadata"] = parse_matrix_metadata(metadata)
    except (OSError, ValueError, httpx.HTTPError, EOFError) as exc:
        result["errors"].append({"stage": "metadata", "message": str(exc)})
    result["download_status"] = "complete" if not result["errors"] else "partial_or_failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    rows = []
    started = datetime.now(UTC).isoformat()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(ingest_one, registry["datasets"]):
            rows.append(row)
            print(f"{row['accession']}: {row['download_status']}; errors={row['errors']}", flush=True)
            write_json(REPORT, {"schema_version": "1.0", "started_at": started,
                               "registry_sha256": digest(REGISTRY), "protocol_sha256": digest(PROTOCOL),
                               "status": "in_progress", "datasets": rows})
    report = {"schema_version": "1.0", "started_at": started,
              "completed_at": datetime.now(UTC).isoformat(),
              "registry_sha256": digest(REGISTRY), "protocol_sha256": digest(PROTOCOL),
              "status": "acquisition_finished_eligibility_pending", "datasets": rows,
              "summary": {"registered": len(rows), "download_complete": sum(not r["errors"] for r in rows),
                          "sealed_candidates": sum(r["role"] == "sealed_candidate" for r in rows),
                          "eligible_independent_validation": 0}}
    write_json(REPORT, report)
    if any(r["errors"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
