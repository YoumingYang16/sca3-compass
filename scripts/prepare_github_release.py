"""Create a new, explicit-allowlist source snapshot. Never publish or edit originals."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXT = {".py", ".tsx", ".ts", ".css", ".html", ".json", ".yaml", ".yml", ".toml", ".md", ".txt", ".ps1", ".R"}
DENY_PARTS = {".git", ".venv", ".r4-venv", "venv", "node_modules", "__pycache__", "data", "external", ".tools", "delivery", "logs"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{35,})"),
    "service_key": re.compile(r"\b(?:sk-proj-|sk-live-|sk_live_)[A-Za-z0-9_\-]{20,}"),
    "aws_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "authenticated_url": re.compile(r"https?://[^\s/@:'\"]+:[^\s/@'\"]+@"),
    "credential_assignment": re.compile(r'''(?im)^\s*(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[:=]\s*["']?([A-Za-z0-9_\-/+=]{24,})'''),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_text(text):
    return [{"rule": k, "line": text.count("\n", 0, m.start()) + 1}
            for k, pattern in PATTERNS.items() for m in pattern.finditer(text)]


def safe_path(path, root=ROOT):
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        return False
    rel = path.relative_to(root)
    if any(p in DENY_PARTS or p.startswith(".env") for p in rel.parts):
        return False
    return path.suffix in EXT and not path.name.endswith((".tsbuildinfo", ".d.ts"))


def candidates():
    paths = set()
    for folder in ["src", "scripts", "tests", "configs", "apps/web/src"]:
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and safe_path(p))
    for name in ["README.md", "pyproject.toml", "apps/web/package.json", "apps/web/pnpm-lock.yaml",
                 "apps/web/index.html", "apps/web/vite.config.ts", "apps/web/tsconfig.json",
                 "apps/web/tsconfig.app.json", "apps/web/tsconfig.node.json", "apps/web/src/vite-env.d.ts",
                 "apps/web/public/research-evidence.json", "docs/DATA_POLICY.md", "docs/ARCHITECTURE.md",
                 "docs/PRODUCT_RELEASE_2026-10-04.md", "docs/PUBLIC_DEPLOYMENT_SECURITY.md", "docs/GITHUB_README.md"]:
        paths.add(ROOT / name)
    # Completed freezes only. Never enumerate protected data or future validation directories.
    phases = {"r2-finite-20260917": ["C001"], "r3-efficiency-20260917": ["C001"],
              "r4-target-calibration-20260918": ["C001"],
              "r5-selection-aware-20260918": ["D011", "D019", "D020"]}
    for phase, batches in phases.items():
        base = ROOT / "research" / phase
        paths.update(p for p in base.glob("*.py") if safe_path(p))
        paths.update(p for p in base.glob("*.md") if safe_path(p) and not any(w in p.name.upper() for w in ["INTAKE", "PROMPT", "TRANSCRIPT"]))
        for name in ["ACCEPTANCE_GATES.json", "FINAL_STATUS.json", "DELIVERY_MANIFEST.json", "EXPERIMENT_REGISTRY.json"]:
            if (base / name).is_file(): paths.add(base / name)
        paths.update(base.glob("requirements*.txt"))
        for batch in batches:
            frozen = base / batch
            paths.update(p for p in frozen.iterdir() if p.is_file() and safe_path(p)
                         and (p.suffix in {".py", ".md", ".txt"} or p.name in {"freeze.json", "protocol.json", "summary.json", "index.json"}))
    v1 = ROOT / "releases/K-NR-1.0.0"
    paths.update(p for p in (v1 / "source").rglob("*") if p.is_file() and safe_path(p))
    paths.update(p for p in v1.glob("*.md"))
    paths.update(v1.glob("requirements*.txt"))
    for name in ["release.json", "MANIFEST.json", "verification.json"]:
        paths.add(v1 / name)
    novelty = ROOT / "research/r3-novelty-20260918"
    for name in ["FINAL_REPORT_ZH.md", "FINAL_CLAIM_EVIDENCE.md", "R3_NOVELTY_THEOREM_MATRIX.md", "CLOSEST_WORK.md", "EFFICIENCY_ROBUSTNESS_MECHANISM.md"]:
        paths.add(novelty / name)
    # No individual records or arrays are copied; the portal JSON contains derived scalar records.
    return sorted(paths)


def prepare(out):
    out = out.resolve()
    if out.exists() or not out.is_relative_to((ROOT / "exports").resolve()):
        raise ValueError("Use a new output directory inside project exports")
    out.mkdir(parents=True)
    copied, excluded = [], []
    for source in candidates():
        relative = source.relative_to(ROOT).as_posix()
        if not source.is_file():
            excluded.append({"path": relative, "reason": "missing"}); continue
        if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
            excluded.append({"path": relative, "reason": "symlink/outside"}); continue
        if source.stat().st_size > 10_000_000:
            excluded.append({"path": relative, "reason": "over 10MB; review separately"}); continue
        try:
            text = source.read_text(encoding="utf-8-sig")
        except UnicodeError:
            excluded.append({"path": relative, "reason": "not UTF8 text"}); continue
        hits = scan_text(text)
        if hits:
            excluded.append({"path": relative, "reason": "credential-pattern review required", "hits": hits}); continue
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"path": relative, "bytes": target.stat().st_size, "sha256": sha(target), "modified": False})
    # Root README is a publication-specific index, not the historical workspace README.
    shutil.copy2(ROOT / "docs/GITHUB_README.md", out / "README.md")
    # These generated files describe the reduced snapshot, not a frozen full-research release.
    (out / ".gitignore").write_text(".env\n.env.*\n**/.venv/\n**/node_modules/\n**/__pycache__/\n**/dist*/\n*.sqlite3\n*.npz\n*.gz\n*.zip\n*.tsbuildinfo\n", encoding="utf8")
    (out / "REPOSITORY_SCOPE.md").write_text(
        "# Curated source/evidence snapshot\n\n"
        "Initial private repository; not a completed R5 method release. No open-source license selected.\n"
        "Includes local product source, selected research source/proofs, completed freeze manifests,\n"
        "summaries and derived D020 scalar records in apps/web/public/research-evidence.json.\n"
        "V1/R2/R3/R4 frozen bytes copied unchanged where included. R5 stays DEVELOPMENT.\n\n"
        "NOT INCLUDED: original observation arrays, full raw experiments, biological datasets,\n"
        "future/held-out data, environments, local runtime databases, old terminal logs and model weights.\n"
        "The original complete archives remain on the owner's machine. Historical full replay commands\n"
        "may require these omitted artifacts and original Windows-path provenance. This snapshot cannot\n"
        "rerun all confirmations or re-export the display JSON without those archives; do not claim it can.\n"
        "The static portal builds directly from the included exported JSON. No data ingestion is necessary.\n\n"
        "Build: cd apps/web; pnpm install --frozen-lockfile; pnpm build:public.\n"
        "Serve only dist-public; never publish FastAPI as-is.\n"
        "GitHub upload does not deploy a public website. PUBLIC_DEPLOYMENT_SECURITY.md gives the boundary.\n"
        "Manifest hashes identify this export, not correctness, novelty, or clinical benefit.\n", encoding="utf8")
    report = {"status": "CURATED_SNAPSHOT_NOT_RESEARCH_ACCEPTANCE", "intended_repo": "YoumingYang16/sca3-compass", "visibility": "private",
              "readme_source": "docs/GITHUB_README.md (publication index; not a frozen historical source)",
              "R5_accepted": False, "missing_or_excluded_files": excluded,
              "policy_exclusions": ["all raw/processed/heldout biological data", "all simulation input and numerical arrays", "credentials/environments/.git", "unreviewed logs", "large binaries and prior archives"],
              "security": "Explicit allowlist + path checks + credential-pattern scan, not an absolute guarantee",
              "frozen_originals_modified": False}
    (out / "EXPORT_REVIEW.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    manifest = []
    for p in sorted(out.rglob("*")):
        if p.is_file(): manifest.append({"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
    (out / "EXPORT_MANIFEST.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"out": str(out), "files": len(manifest) + 1, "bytes": sum(f["bytes"] for f in manifest),
                      "exclusions": len(excluded), "credential_hit_files": sum(bool(e.get("hits")) for e in excluded)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    prepare(parser.parse_args().out)
