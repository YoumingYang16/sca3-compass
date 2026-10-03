"""TRAIN extraction, provenance, deterministic seeds and failure semantics."""
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import robustness_cf_repeated_diagnostic as repeated
from sca3_compass.robustness_methods import evaluate_candidates, contrasts
from sca3_compass.robustness_prior import energy_prior_candidates

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / repeated.CONFIG).read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("index", repeated.INDICES)
def test_extraction_exactly_equals_actual_reference(index):
    with threadpool_limits(limits=1):
        z, calibration = repeated.generate_observations(CASES[index], index, 0)
        actual, receipt = repeated.extract_train(z, calibration)
        _, diagnostics = evaluate_candidates(z, calibration, "frontier")
        energy_prior_candidates(z, calibration, diagnostics)
        info = diagnostics["energy_prior"]["folds"][0]
        shape = np.asarray(info["study_shape"])
        means = z.mean(-1)
        residual = (z - means[..., None]) @ contrasts(z.shape[-1])
        q = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual) / (1 - diagnostics["fit"]["rho"])
        train = np.arange(len(z)) % 2 != 0
        dimension, v = info["residual_dimension"], info["projection_variance"]
        prior = info["target_only"]
        df = np.inf if prior["gaussian_bic_selected"] else prior["df"] + dimension
        variance = (np.full(train.sum(), v * prior["scatter"]) if np.isinf(df)
                    else v * (prior["df"] * prior["scatter"] + q[train]) / df)
        for key, expected in (("means", means[train]), ("q", q[train]),
                              ("shape", shape), ("base_variance", variance)):
            np.testing.assert_array_equal(actual[key], expected)
        assert actual["df"] == (None if np.isinf(df) else df)
        assert actual["projection_variance"] == v
        assert actual["residual_dimension"] == dimension
        assert receipt["uses_held_rows"] is False


def test_held_rows_can_be_poisoned_without_changing_inputs_or_receipts():
    with threadpool_limits(limits=1):
        z, calibration = repeated.generate_observations(CASES[2], 2, 1)
        first = repeated.extract_train(z, calibration)
        z[::2] = np.nan
        second = repeated.extract_train(z, calibration)
    assert repeated.json_value(first) == repeated.json_value(second)


def test_generator_truth_return_is_never_read(monkeypatch):
    z = np.zeros((32, 4, 6))
    calibration = np.ones((4, 64, 6))

    class TruthTrap:
        def __getitem__(self, index):
            if index not in (0, 1):
                raise AssertionError("Truth was read")
            return (z, calibration)[index]

        def __iter__(self):
            raise AssertionError("Generator returns were unpacked")

    monkeypatch.setattr(repeated, "data", lambda *args: TruthTrap())
    actual_z, actual_calibration = repeated.generate_observations(CASES[0], 0, 0)
    assert actual_z is z and actual_calibration is calibration


def test_seeds_observations_and_snapshot_bytes_are_deterministic(tmp_path):
    assert repeated.seed_receipt(6, 7) == repeated.seed_receipt(6, 7)
    assert repeated.seed_receipt(6, 7) != repeated.seed_receipt(6, 8)
    assert repeated.seed_receipt(6, 7) != repeated.seed_receipt(2, 7)
    with threadpool_limits(limits=1):
        a = repeated.generate_observations(CASES[6], 6, 7)
        b = repeated.generate_observations(CASES[6], 6, 7)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left, right)
    first = repeated.freeze_sources(ROOT, tmp_path / "first")
    second = repeated.freeze_sources(ROOT, tmp_path / "second")
    assert first == second
    repeated.verify_sources(tmp_path / "first", first)
    repeated.verify_sources(tmp_path / "second", second)
    assert "src/sca3_compass/robustness_prior.py" in first
    assert "src/sca3_compass/robustness_patterns.py" in first
    repeated.write_new(tmp_path / "a.json.gz", {"z": a[0]}, True)
    repeated.write_new(tmp_path / "b.json.gz", {"z": b[0]}, True)
    assert repeated.sha256(tmp_path / "a.json.gz") == repeated.sha256(tmp_path / "b.json.gz")
    with pytest.raises(FileExistsError):
        repeated.write_new(tmp_path / "a.json.gz", {})


def inputs():
    return {"means": np.ones((32, 4)), "base_variance": np.ones(32), "shape": np.eye(4),
            "df": None, "q": np.ones(32) * 20, "residual_dimension": 20, "projection_variance": 1.}


@pytest.mark.parametrize("mode", ["cf_base", "cf_observed_scale"])
def test_gaussian_constant_variance_never_becomes_success(mode):
    result = repeated.fit_one(mode, inputs())
    assert result["status"] == "NOT_IDENTIFIED"
    assert result["tau"] is None
    assert not result["eligible_for_estimate_summary"]


def test_exception_is_failure_without_fallback(monkeypatch):
    def explode(*args, **kwargs):
        raise FloatingPointError("intentional numerical failure")
    monkeypatch.setattr(repeated, "cf_fit", explode)
    result = repeated.fit_one("cf_observed_scale", inputs())
    assert result["status"] == "FAILED_EXCEPTION"
    assert result["tau"] is None
    assert "intentional" in result["error_message"]


def test_failed_cf_local_fit_cannot_be_counted_as_favorable_estimate(monkeypatch):
    monkeypatch.setattr(repeated, "cf_fit", lambda *a, **k: {
        "tau": 1.2, "status": "DEVELOPMENT_PROFILE_OPTIMUM_NOT_IDENTIFICATION",
        "grid_objective": [.9], "objective": .9, "active_boundary": False,
        "local_fits": [{"success": False}]})
    result = repeated.fit_one("cf_base", inputs())
    assert result["status"] == "FAILED_NUMERICAL"
    assert result["raw"]["tau"] == 1.2
    assert not result["eligible_for_estimate_summary"]


def test_joint_numerical_event_cannot_be_hidden_by_selected_convergence():
    result = repeated.classify("point_joint", {"tau": 1., "diagnostics": {
        "converged": True, "numerical_failure_count": 1,
        "central_initialization_failure": None, "active_bound": False}})
    assert result["status"] == "FAILED_NUMERICAL"


def test_snapshot_tampering_is_detected(tmp_path):
    repeated.write_new(tmp_path / "source/receipt.json", {"a": 1})
    with pytest.raises(RuntimeError, match="hash mismatch"):
        repeated.verify_sources(tmp_path, {"receipt.json": "incorrect"})


def test_full_schedule_prespecified_unique_and_complete():
    plan = repeated.protocol(ROOT, 32, True, {})
    assert len(plan["schedule"]) == 224
    assert len({tuple(item["seed"]["entropy"]) for item in plan["schedule"]}) == 224
    assert all(item["seed"]["entropy"][0] == 7105211 for item in plan["schedule"])
    assert plan["outer_fold"] == 0 and plan["held_rows_used"] is False
