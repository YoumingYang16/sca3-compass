"""Product contract checks; not a new statistical confirmation."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("product_evidence", ROOT / "scripts/export_product_evidence.py")
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.fixture(scope="module")
def payload():
    return json.loads((ROOT / "apps/web/public/research-evidence.json").read_text(encoding="utf-8"))


def test_all_paired_records_and_stages(payload):
    assert len(payload["r5"]["records"]) == 20
    assert len(payload["r5"]["cases"]) == 10
    assert payload["r5"]["confirmation_count"] == 0
    assert not payload["decision"]["R5_promoted"]
    assert sum(r["n"] for r in payload["r4"]["rows"]) == 5120
    assert all(r["n"] == 512 for r in payload["r4"]["rows"])


def test_export_matches_saved_public_payload(payload):
    if not (ROOT / "research/r5-selection-aware-20260918/D020/raw").exists():
        pytest.skip("Full owner-held raw archive intentionally not in curated repository")
    assert payload == export.extract(ROOT)


def test_null_power_is_not_zero(payload):
    for c in [4, 5, 6]:
        assert all(m["power"] is None for m in payload["r5"]["cases"][c]["methods"])


def test_negative_and_outside_results_visible(payload):
    row = {m["id"]: m for m in payload["r5"]["cases"][7]["methods"]}
    assert row["R5_A014"]["power"] < row["R4_target"]["power"]
    assert row["R5_A014"]["power"] < row["matched_mix"]["power"]
    outside = payload["r5"]["cases"][9]
    assert outside["outside"]
    assert outside["methods"][0]["fdp"] > .05


@pytest.mark.parametrize("bad", ["../secret", "C:/private", "raw/../../secret"])
def test_no_path_escape(bad):
    with pytest.raises(ValueError):
        export.safe_member(ROOT, bad)


def test_scalar_projection_has_no_raw_patient_or_simulation_inputs(payload):
    for row in payload["r5"]["records"]:
        assert not {"z", "truth", "calibration_source", "calibration_target", "traceback"}.intersection(row)
    assert payload["kind"] == "SIMULATION_NOT_PATIENT_DATA"
