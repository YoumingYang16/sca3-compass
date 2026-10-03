"""No network operations: release path and secret scanner regression checks."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_safety", ROOT / "scripts/prepare_github_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_detects_credential_not_value_in_output():
    fake = "ghp_" + "x" * 40
    hits = module.scan_text("first line\n" + fake)
    assert hits == [{"rule": "github_token", "line": 2}]
    assert fake not in repr(hits)


def test_detects_authenticated_url():
    assert module.scan_text("https://" + "name:credential" + "@example.org/repo")


@pytest.mark.parametrize("p", ["data/raw/example.py", ".venv/lib/file.py", "scripts/.env.secret", "apps/web/node_modules/file.ts"])
def test_denied_paths(p):
    assert not module.safe_path(ROOT / p)


def test_safe_research_paths():
    assert module.safe_path(ROOT / "src/sca3_compass/app.py")
    assert module.safe_path(ROOT / "research/r5-selection-aware-20260918/D020/action_borrowing.py")


def test_public_build_has_no_legacy_api_code():
    assets = ROOT / "apps/web/dist-public/assets"
    if not assets.exists(): pytest.skip("Run the public web build first")
    scripts = list(assets.glob("*.js"))
    assert scripts
    for p in scripts:
        text = p.read_text(encoding="utf-8")
        assert "/api/learning" not in text
        assert "/api/molecular" not in text
        assert "/api/design" not in text
        assert "GSE320100" not in text


def test_public_build_only_exports_intended_files():
    built = ROOT / "apps/web/dist-public"
    if not built.exists(): pytest.skip("Run the public web build first")
    for p in built.rglob("*"):
        if p.is_file():
            assert p.suffix in {".html", ".js", ".css", ".json"}
            assert not p.is_symlink()
