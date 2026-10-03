"""Numerical checks use labeled mathematical fixtures, not patient observations."""

import sqlite3

import numpy as np
import pytest
from pydantic import ValidationError
from scipy.integrate import quad
from scipy.stats import multivariate_normal

from sca3_compass.advanced_analytics import _posterior_grid
from sca3_compass.analytics import StudyEstimate
from sca3_compass.design_assurance import (
    AssuranceRequest,
    calibrate_sequential_boundary,
    run_design_assurance,
    sequential_null_rejection,
    subject_slope_variance,
    wilson_interval,
)
from sca3_compass.experiment_store import ExperimentStore


def mathematical_fixture():
    return [
        StudyEstimate(
            str(i),
            "MATHEMATICAL_TEST_FIXTURE",
            effect,
            0.15,
            None,
            "https://example.org/test-fixture",
        )
        for i, effect in enumerate([1.2, 1.4, 1.5, 0.8])
    ]


def test_quadrature_matches_independent_multivariate_normal_integration():
    effects = np.array([1.2, 1.4, 1.5, 0.8])
    variances = np.full(4, 0.15**2)
    grid = _posterior_grid(
        effects,
        variances,
        mu_prior_mean=1.5,
        mu_prior_sd=2,
        tau_prior_scale=0.5,
        grid_points=8001,
    )

    def density(tau):
        covariance = np.diag(variances + tau * tau) + 4 * np.ones((4, 4))
        return np.exp(
            multivariate_normal.logpdf(effects, mean=np.full(4, 1.5), cov=covariance)
            - 0.5 * (tau / 0.5) ** 2
        )

    norm = quad(density, 0, np.inf, epsabs=1e-10)[0]
    expected_tau = (
        quad(lambda tau: tau * density(tau), 0, np.inf, epsabs=1e-10)[0] / norm
    )
    assert abs(grid["probabilities"] @ grid["tau"] - expected_tau) < 1e-5


def test_analytic_slope_variance_matches_independent_subject_simulation():
    rng = np.random.default_rng(142)
    times = np.arange(5) * 0.5
    errors = rng.multivariate_normal(
        np.zeros(5),
        0.8**2 * 0.45 ** np.abs(np.subtract.outer(np.arange(5), np.arange(5))),
        size=80_000,
    )
    outcomes = (
        rng.normal(15, 6, (80_000, 1))
        + rng.normal(1.5, 1.5, (80_000, 1)) * times
        + errors
    )
    beta = np.linalg.lstsq(
        np.column_stack([np.ones(5), times]), outcomes.T, rcond=None
    )[0][1]
    assert abs(np.var(beta) / subject_slope_variance(24, 1.5, 0.8, 0.45) - 1) < 0.02


def test_single_look_recovers_z_boundary():
    result = calibrate_sequential_boundary([1.0], 80_000, 415)
    assert abs(result["constant"] - 1.959964) < 0.025
    assert abs(result["null_type1"]["estimate"] - 0.025) < 0.003


def test_sequential_null_validation_is_not_naive_repeated_testing():
    result = calibrate_sequential_boundary([0.5, 0.75, 1], 100_000, 14)
    assert abs(result["null_type1"]["estimate"] - 0.025) < 0.003
    assert result["naive_repeated_testing_type1"]["estimate"] > 0.035
    assert result["z_boundaries"][0] > result["z_boundaries"][-1]
    assert result["doubled_resolution_alpha_discrepancy"] < 1e-8


def test_sequential_recursion_matches_independent_multivariate_normal_cdf():
    times = np.array([.5, .75, 1.0])
    constant = 2.02
    covariance = np.minimum.outer(times, times)
    reference = 1 - multivariate_normal.cdf(np.full(3, constant), mean=np.zeros(3), cov=covariance, maxpts=2_000_000, abseps=1e-7, rng=np.random.default_rng(617))
    assert abs(sequential_null_rejection(constant, times.tolist()) - reference) < 2e-6


@pytest.mark.parametrize(
    "payload",
    [
        {"sample_sizes": [80, 40]},
        {"sample_sizes": [0]},
        {"durations": [13]},
        {"information_fractions": [0.5, 0.4, 1]},
        {"information_fractions": [0, 1]},
        {"information_fractions": [0.5, 0.9]},
        {"slope_sd": float("nan")},
        {"unknown": 1},
    ],
)
def test_invalid_designs_are_rejected(payload):
    with pytest.raises(ValidationError):
        AssuranceRequest(**payload)


def small_request(**kwargs):
    return AssuranceRequest(
        sample_sizes=[40, 100],
        durations=[24],
        simulations=2000,
        calibration_simulations=10_000,
        seed=47,
        **kwargs,
    )


def test_assurance_is_reproducible_monotone_and_explicitly_simulated():
    result = run_design_assurance(mathematical_fixture(), small_request())
    assert result == run_design_assurance(mathematical_fixture(), small_request())
    assert result["cells"][0]["fixed_assurance"] < result["cells"][1]["fixed_assurance"]
    assert result["provenance_tier"].startswith("SIMULATION_OUTPUT")
    assert "assumed" in result["input_contract"]
    assert len(result["cells"][0]["sensitivity"]) == 12


def test_zero_effect_reduces_to_null_fixed_probability():
    result = run_design_assurance(
        mathematical_fixture(), small_request(treatment_reduction=0)
    )
    for cell in result["cells"]:
        assert abs(cell["fixed_assurance"] - 0.025) < 1e-12
    assert result["candidate"] is None


def test_wilson_endpoints():
    assert wilson_interval(0, 100)["low"] == 0
    assert wilson_interval(100, 100)["high"] == 1


def test_experiment_content_addressing_and_tamper_detection(tmp_path):
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    result = run_design_assurance(mathematical_fixture(), small_request())
    first, second = store.save(result), store.save(result)
    assert first == second
    assert len(store.recent()) == 1
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE experiments SET payload='{}'")
    with pytest.raises(ValueError, match="digest"):
        store.get(first["run_id"])
