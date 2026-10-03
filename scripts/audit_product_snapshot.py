"""Check curated copy bytes, frozen manifests and static exposure; no models run."""
import argparse
import hashlib
import json
from pathlib import Path
from prepare_github_release import scan_text


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    problems = []
    manifest = json.loads((root / "EXPORT_MANIFEST.json").read_text())
    listed = {"EXPORT_MANIFEST.json"}
    for row in manifest["files"]:
        p = root / row["path"]
        if not p.resolve().is_relative_to(root.resolve()) or p.is_symlink():
            problems.append({"path": row["path"], "issue": "unsafe path"}); continue
        listed.add(row["path"])
        if sha(p) != row["sha256"]: problems.append({"path": row["path"], "issue": "manifest mismatch"})
        if scan_text(p.read_text(encoding="utf-8-sig")):
            problems.append({"path": row["path"], "issue": "credential pattern"})
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts}
    if actual != listed: problems.append({"issue": "file list mismatch"})
    freeze_files = [*root.glob("research/*/C001/freeze.json"), *root.glob("research/r5-selection-aware-20260918/D*/freeze.json")]
    frozen_count = 0
    for f in freeze_files:
        for rel, digest in json.loads(f.read_text())["files"].items():
            p = f.parent / rel
            if not p.is_file() or sha(p) != digest:
                problems.append({"path": str(p.relative_to(root)), "issue": "frozen source missing/changed"})
            else: frozen_count += 1
    v1 = root / "releases/K-NR-1.0.0"
    for rel, digest in json.loads((v1 / "release.json").read_text())["source_sha256"].items():
        p = v1 / "source" / rel
        if not p.is_file() or sha(p) != digest:
            problems.append({"path": str(p.relative_to(root)), "issue": "V1 frozen source missing/changed"})
        else: frozen_count += 1
    # Build is separate, read-only artefact; never serve the source snapshot itself.
    source = Path(__file__).resolve().parents[1]
    assets = source / "apps/web/dist-public"
    js = list((assets / "assets").glob("*.js"))
    if not js: problems.append({"issue": "public build missing"})
    for p in js:
        if any(v in p.read_text(encoding="utf8") for v in ["/api/molecular", "/api/learning", "/api/design", "GSE320100"]):
            problems.append({"issue": "legacy/private route in public build", "path": p.name})
    if sha(assets / "research-evidence.json") != sha(root / "apps/web/public/research-evidence.json"):
        problems.append({"issue": "public build has stale evidence"})
    return {"status": "PASS" if not problems else "FAIL", "files": len(listed),
            "frozen_files_checked": frozen_count, "issues": problems,
            "limited_checks_not_absolute_security_guarantee": True, "R5_accepted": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.snapshot.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf8")
    print(json.dumps(report))
    raise SystemExit(0 if report["status"] == "PASS" else 1)
