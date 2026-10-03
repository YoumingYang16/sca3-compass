"""Called only after verify.ps1 has checked pytest, lint and build exit codes."""
import sys
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree

from sca3_compass.molecular_data import PROJECT_ROOT, digest, write_json

path = Path(sys.argv[1]).resolve()
path.relative_to(PROJECT_ROOT)  # Only repository-local generated test receipts.
tree = ElementTree.parse(path)
suites = list(tree.getroot().iter("testsuite"))
counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ["tests", "failures", "errors", "skipped"]}
if counts["tests"] == 0 or counts["failures"] or counts["errors"]:
    raise SystemExit("Non-passing JUnit output")
files = sorted((PROJECT_ROOT / "src").rglob("*.py")) + sorted((PROJECT_ROOT / "tests").rglob("*.py"))
files += sorted((PROJECT_ROOT / "apps/web/src").rglob("*.tsx")) + sorted((PROJECT_ROOT / "apps/web/src").rglob("*.css"))
files += sorted((PROJECT_ROOT / "apps/web/dist").rglob("*"))
files += [PROJECT_ROOT / "scripts/verify.ps1", Path(__file__), PROJECT_ROOT / "pyproject.toml"]
result = {"created_at": datetime.now(UTC).isoformat(), "passed": True, "pytest": counts,
          "junit_path": str(path.relative_to(PROJECT_ROOT)), "junit_sha256": digest(path),
          "ruff": "passed", "typescript_and_vite": "passed",
          "scope": "Local execution record, not a third-party certification. Warnings are not suppressed.",
          "file_sha256": {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in files if p.is_file()}}
write_json(PROJECT_ROOT / "artifacts/software-verification.json", result)
print(f"Verification receipt: {counts['tests']} tests, zero failures/errors")
