import numpy as np
import pytest

from sca3_compass.molecular_qc import match_columns
from sca3_compass.molecular_shrinkage import fit_mixture, posterior


def test_gaussian_conjugacy_matches_closed_form():
    b = np.array([[2.0], [-1.0]])
    v = np.ones((2, 1, 1)) * 0.5
    priors = np.ones((1, 1, 1)) * 2
    result = posterior(b, v, priors, np.array([1.0]))
    np.testing.assert_allclose(result["mean"], b * 0.8)
    np.testing.assert_allclose(result["sd"]**2, 0.4)


def test_point_null_sign_probability_is_not_false_certainty():
    result = posterior(np.ones((3, 2)), np.tile(np.eye(2), (3, 1, 1)), np.zeros((1, 2, 2)), np.array([1.0]))
    np.testing.assert_allclose(result["mean"], 0)
    np.testing.assert_allclose(result["lfsr"], 1)


def test_optimizer_preserves_monotonicity_and_simplex():
    rng = np.random.default_rng(10)
    b = rng.normal(size=(150, 2))
    v = np.tile(np.eye(2) * 0.5, (150, 1, 1))
    priors = np.array([np.zeros((2, 2)), np.eye(2), np.ones((2, 2))])
    result = fit_mixture(b, v, priors, max_iter=50)
    assert np.min(np.diff(result["objective"])) >= -1e-6
    assert np.isclose(result["weights"].sum(), 1)
    assert result["converged"]
    assert result["simplex_kkt_residual"] < 1e-4


def test_invalid_covariance_rejected():
    with pytest.raises(ValueError):
        posterior(np.ones((2, 2)), np.zeros((2, 2, 2)), np.zeros((1, 2, 2)), np.array([1.0]))


def test_sample_mapping_uses_recorded_ids_and_not_position():
    samples = [{"sample_id": "GSM1", "title": ["mouse 111 LUM0332"]},
               {"sample_id": "GSM2", "title": ["222 [LUM0413]"]}]
    assert [s["sample_id"] for s in match_columns(["LUM0413", "LUM0332"], samples)] == ["GSM2", "GSM1"]
    with pytest.raises(ValueError):
        match_columns(["unknown", "LUM0332"], samples)
