"""Audit registered public sources without downloading large archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .data_provenance import validate_registry

USER_AGENT = "SCA3-Compass-Source-Audit/0.1 (research reproducibility)"


def _request(url: str, mode: str, max_bytes: int) -> dict[str, Any]:
    method = "HEAD" if mode == "head" else "GET"
    request = Request(url, method=method, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        body = b"" if method == "HEAD" else response.read(max_bytes + 1)
        truncated = len(body) > max_bytes
        if truncated:
            body = body[:max_bytes]
        return {
            "http_status": response.status,
            "content_type": response.headers.get("Content-Type"),
            "content_length": response.headers.get("Content-Length"),
            "etag": response.headers.get("ETag"),
            "last_modified": response.headers.get("Last-Modified"),
            "bytes_inspected": len(body),
            "payload_truncated": truncated,
            "sha256_inspected_payload": hashlib.sha256(body).hexdigest()
            if body
            else None,
            "body": body,
        }


def audit_source(item: dict[str, Any]) -> dict[str, Any]:
    check = item.get("check", {})
    mode = check.get("mode", "head")
    max_bytes = int(check.get("max_bytes", 5_000_000))
    started = datetime.now(UTC)
    result: dict[str, Any] = {
        "id": item["id"],
        "tier": item["tier"],
        "url": item["url"],
        "checked_at": started.isoformat(),
        "ok": False,
    }
    try:
        response = _request(item["url"], mode, max_bytes)
        body = response.pop("body")
        result.update(response)
        if response["http_status"] != 200:
            raise ValueError(f"unexpected HTTP status {response['http_status']}")
        expected = check.get("contains")
        if expected:
            text = body.decode("utf-8", errors="replace")
            missing = [value for value in expected if value not in text]
            if missing:
                raise ValueError(f"payload missing markers: {missing}")
        if mode == "json":
            payload = json.loads(body)
            field = check.get("positive_integer_field")
            if field and (
                not isinstance(payload.get(field), int) or payload[field] <= 0
            ):
                raise ValueError(f"JSON field {field!r} is not a positive integer")
            result["reported_count"] = payload.get(field) if field else None
        result["ok"] = True
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_ms"] = round((datetime.now(UTC) - started).total_seconds() * 1000)
    return result


def run_audit(registry_path: Path) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry["sources"])
    results = [audit_source(item) for item in registry["sources"]]
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "registry": str(registry_path),
        "summary": {
            "total": len(results),
            "passed": sum(item["ok"] for item in results),
            "failed": sum(not item["ok"] for item in results),
        },
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_audit(args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
