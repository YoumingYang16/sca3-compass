"""Run software checks and verify the computed literature artifacts without inventing results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from sca3_compass.literature_corpus import digest_json, load_corpus, verify_spans
from sca3_compass.semantic_retrieval import REPORT_PATH, LiteratureIndex, file_sha256

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    env = {**os.environ, "SCA3_RUN_API_TESTS": "1", "PYTHONIOENCODING": "utf-8"}
    (ROOT / ".run").mkdir(exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix="literature-validation-", dir=ROOT / ".run")
    )
    if not temporary.resolve().is_relative_to((ROOT / ".run").resolve()):
        raise ValueError("Test directory escaped the project's .run directory")
    checks = []
    for name, command in [
        (
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "tests",
                "-q",
                "-p",
                "no:cacheprovider",
                "--basetemp",
                str(temporary / "tests"),
            ],
        ),
        (
            "ruff",
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                "src",
                "tests",
                "scripts/validate_literature_release.py",
            ],
        ),
    ]:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        checks.append(
            {
                "name": name,
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        print(completed.stdout, flush=True)
        if completed.returncode:
            return completed.returncode
    corpus = load_corpus()
    verification = verify_spans(corpus)
    index = LiteratureIndex()
    benchmark = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    if benchmark["sha256"] != digest_json(
        {k: v for k, v in benchmark.items() if k != "sha256"}
    ):
        raise ValueError("Benchmark digest mismatch")
    if (
        benchmark["corpus_sha256"] != corpus["sha256"]
        or benchmark["index_sha256"] != index.manifest["sha256"]
    ):
        raise ValueError("Benchmark inputs no longer match the live corpus and index")
    for name, expected in benchmark["reproducibility"]["source_sha256"].items():
        if file_sha256(ROOT / "src/sca3_compass" / name) != expected:
            raise ValueError(f"Source changed since the benchmark: {name}")
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "passed",
        "software_checks": checks,
        "corpus_verification": verification,
        "articles": len(corpus["articles"]),
        "corpus_sha256": corpus["sha256"],
        "index_sha256": index.manifest["sha256"],
        "benchmark_sha256": benchmark["sha256"],
        "metrics": benchmark["aggregate"],
        "gpu_build": index.manifest["device"],
        "boundary": "Software and artifact integrity checks, not clinical validation or independent user evaluation.",
    }
    output = ROOT / "artifacts/research-release-4-validation.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Verified release report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
