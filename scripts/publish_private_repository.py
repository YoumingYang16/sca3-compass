"""Publish the explicitly reviewed new snapshot using normal Git credentials.

Credentials stay in memory, are obtained for github.com only, and are never
printed, stored in source/config, passed in a URL or written to a log.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.error
import urllib.request

OWNER = "YoumingYang16"
NAME = "sca3-compass"
URL = f"https://github.com/{OWNER}/{NAME}.git"


def credentials():
    env = dict(os.environ, GCM_INTERACTIVE="never", GIT_TERMINAL_PROMPT="0")
    p = subprocess.run(["git", "-c", "credential.interactive=false", "credential", "fill"],
        input=f"protocol=https\nhost=github.com\nusername={OWNER}\n\n", text=True,
        capture_output=True, timeout=30, env=env)
    if p.returncode: raise RuntimeError("GitHub credential helper unavailable; authenticate through normal Git login")
    result = dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)
    if not result.get("password"): raise RuntimeError("No GitHub credential returned")
    return result["password"]


def api(token, path, payload=None):
    request = urllib.request.Request("https://api.github.com" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                 "User-Agent": "SCA3-Compass-authorized-release", "X-GitHub-Api-Version": "2026-03-10"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def verify_snapshot(path):
    original = Path(__file__).resolve().parents[1]
    if not path.resolve().is_relative_to((original / "exports").resolve()):
        raise ValueError("Only isolated exports snapshot is allowed")
    if (path / ".git").exists():
        raise ValueError("Initial upload requires a fresh snapshot with no Git history; do not reuse failed push directories")
    manifest = json.loads((path / "EXPORT_MANIFEST.json").read_text())
    expected = {"EXPORT_MANIFEST.json"}
    for f in manifest["files"]:
        p = path / f["path"]
        if p.is_symlink() or not p.resolve().is_relative_to(path.resolve()): raise ValueError("unsafe member")
        if hashlib.sha256(p.read_bytes()).hexdigest() != f["sha256"]: raise ValueError("snapshot changed")
        expected.add(f["path"])
    actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file() and ".git" not in p.relative_to(path).parts}
    if actual != expected: raise ValueError("unmanifested snapshot files")
    return len(expected)


def git(path, *args):
    proc = subprocess.run(["git", "-C", str(path), "-c", f"credential.https://github.com.username={OWNER}", *args], text=True, capture_output=True, timeout=180,
                          env=dict(os.environ, GCM_INTERACTIVE="never", GIT_TERMINAL_PROMPT="0"))
    if proc.returncode: raise RuntimeError("Git operation failed: " + args[0] + " (details withheld to avoid credential leakage)")
    return proc.stdout.strip()


def verify_commit(path):
    if git(path, "rev-list", "--count", "HEAD") != "1":
        raise ValueError("Unexpected commit ancestry")
    expected = {r["path"]: r["sha256"] for r in json.loads((path / "EXPORT_MANIFEST.json").read_text())["files"]}
    expected["EXPORT_MANIFEST.json"] = hashlib.sha256((path / "EXPORT_MANIFEST.json").read_bytes()).hexdigest()
    result = subprocess.run(["git", "-C", str(path), "archive", "--format=tar", "HEAD"],
                            capture_output=True, timeout=60)
    if result.returncode: raise ValueError("Cannot verify committed snapshot")
    actual = {}
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for member in archive.getmembers():
            if member.isdir(): continue
            if not member.isfile(): raise ValueError("Unexpected committed link or special file")
            actual[member.name] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    if actual != expected: raise ValueError("Committed bytes differ from approved snapshot")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["status", "publish"])
    p.add_argument("--snapshot", type=Path)
    args = p.parse_args()
    if args.action == "publish":
        if args.snapshot is None: raise ValueError("--snapshot is required")
        root = args.snapshot.resolve()
        count = verify_snapshot(root)
    token = credentials()
    user = api(token, "/user")
    if user["login"].lower() != OWNER.lower(): raise ValueError("Authenticated GitHub account differs from authorized owner")
    if args.action == "status":
        print(json.dumps({"authenticated_account": user["login"], "id": user["id"], "credentials_disclosed": False})); return
    try:
        repo = api(token, f"/repos/{OWNER}/{NAME}")
    except urllib.error.HTTPError as error:
        if error.code != 404: raise
        repo = api(token, "/user/repos", {"name": NAME, "private": True, "auto_init": False,
              "description": "SCA3 Compass: transparent research evidence and family-facing literacy; R5 remains experimental.",
              "has_wiki": False, "has_projects": False})
    if not repo["private"] or repo["owner"]["login"].lower() != OWNER.lower():
        raise ValueError("Repository scope/visibility mismatch; not changing it")
    if not (root / ".git").exists():
        # Size alone is not sufficient: remote must have no refs at all.
        if repo["size"] != 0 or git(root, "ls-remote", URL):
            raise ValueError("Existing repository has content or refs; review before writing")
        git(root, "init", "-b", "main")
        git(root, "config", "user.name", OWNER)
        git(root, "config", "user.email", f"{user['id']}+{OWNER}@users.noreply.github.com")
        git(root, "config", "core.autocrlf", "false")
        git(root, "config", "credential.https://github.com.username", OWNER)
        git(root, "remote", "add", "origin", URL)
        git(root, "add", "--all")
        staged = git(root, "diff", "--cached", "--name-only").splitlines()
        if len(staged) != count: raise ValueError("Not every approved snapshot file is staged; inspect ignore rules")
        git(root, "commit", "-m", "Add reviewed evidence portal and scoped research source; R5 experimental")
    verify_commit(root)
    if git(root, "remote", "get-url", "origin") != URL: raise ValueError("Unexpected remote")
    git(root, "push", "-u", "origin", "main")
    local = git(root, "rev-parse", "HEAD")
    remote = git(root, "ls-remote", "origin", "refs/heads/main").split()[0]
    if remote != local: raise ValueError("Remote commit not verified")
    checked = api(token, f"/repos/{OWNER}/{NAME}")
    if not checked["private"]: raise ValueError("Private visibility not confirmed")
    print(json.dumps({"status": "UPLOADED_VERIFIED", "url": checked["html_url"], "private": True,
                      "commit": local, "files": count, "R5_accepted": False, "website_deployed": False}))


if __name__ == "__main__":
    try:
        main()
    except urllib.error.HTTPError as exc:
        print(json.dumps({"status": "BLOCKED", "http_status": exc.code})); raise SystemExit(1)
