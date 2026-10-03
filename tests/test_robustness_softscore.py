from itertools import combinations

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import betabinom
from threadpoolctl import threadpool_limits

import sca3_compass.robustness_softscore as softscore_module
from sca3_compass.molecular_envelope import rank_tail
from sca3_compass.molecular_methods import covariance_root
from sca3_compass.robustness_mc import projection_weights
from sca3_compass.robustness_softscore import (
    _rank_tail_sorted,
    _soft_log_scores,
    softscore_pc,
)


@pytest.fixture(autouse=True)
def cap_blas_in_test_caller():
    with threadpool_limits(limits=1, user_api="blas"):
        yield


def test_score_is_prespecified_sparse_dense_likelihood_mixture():
    shape = .35 * np.ones((4, 4)) + .65 * np.eye(4)
    w = projection_weights(shape, (0, 2, 3))
    assert w.shape == (4, 7)
    assert np.all(w >= 0) and np.all(w[1] == 0)
    np.testing.assert_allclose(np.diag(w.T @ shape @ w), 1, atol=1e-14)
    x = np.array([[0., 2., -1., 3.], [2., -4., 1., 0.]])
    actual = np.exp(_soft_log_scores(x @ w))
    prior = np.array([.15] * 3 + [1 / 30] * 3 + [.45])
    expected2 = np.exp(2 * (x @ w) - 2) @ prior
    expected4 = np.exp(4 * (x @ w) - 8) @ prior
    np.testing.assert_allclose(actual[:, 0], expected2, rtol=1e-14)
    np.testing.assert_allclose(actual[:, 1], expected4, rtol=1e-14)
    np.testing.assert_allclose(actual[:, 2], (expected2 + expected4) / 2, rtol=1e-14)
    assert np.isfinite(_soft_log_scores(np.full((2, 7), 10000.))).all()
    assert np.isfinite(_soft_log_scores(np.full((2, 7), -10000.))).all()


@pytest.mark.parametrize("df", [np.inf, 5.])
def test_coordinatewise_monotonicity_in_scores_and_pc_pvalues(df):
    rng = np.random.default_rng(382)
    shapes = [1.2 * np.eye(4) - .2 * np.ones((4, 4)),
              .45 * np.ones((4, 4)) + .55 * np.eye(4)]
    mean = rng.normal(size=(17, 2, 4))
    variance = np.exp(rng.normal(size=17))
    increased = mean + rng.uniform(0, 2, size=mean.shape)
    for shape in shapes:
        for triple in combinations(range(4), 3):
            w = projection_weights(shape, triple)
            a = _soft_log_scores((mean / np.sqrt(variance[:, None, None])) @ w)
            b = _soft_log_scores((increased / np.sqrt(variance[:, None, None])) @ w)
            assert np.all(b >= a)
    a, _ = softscore_pc(mean, variance, shapes, df, np.random.default_rng(27), 4095)
    b, _ = softscore_pc(increased, variance, shapes, df, np.random.default_rng(27), 4095)
    for name in a:
        assert np.all(b[name] <= a[name])


def test_rank_ties_are_included_and_plus_one_prevents_zero():
    reference = np.array([1., 2., 2., 4.])
    observed = np.array([[0., 1., 2.], [3., 4., 5.]])
    expected = np.array([[1., 1., .8], [.4, .4, .2]])
    np.testing.assert_array_equal(_rank_tail_sorted(reference, observed), expected)
    np.testing.assert_array_equal(_rank_tail_sorted(reference, observed), rank_tail(reference, observed))
    np.testing.assert_array_equal(_rank_tail_sorted(np.zeros(10), np.zeros((3, 2))), 1.)


@pytest.mark.parametrize("df", [np.inf, 7.])
def test_output_matches_direct_fold_triple_ranks_and_reuses_bank(df):
    data_rng = np.random.default_rng(901)
    mean = data_rng.normal(size=(11, 2, 4))
    variance = np.exp(data_rng.normal(size=11))
    correlations = [np.eye(4), .4 * np.ones((4, 4)) + .6 * np.eye(4)]
    scales = [np.array([.6, 1., 1.3, 2.]), np.array([2., .8, 1.2, .5])]
    shapes = [c * np.outer(s, s) for c, s in zip(correlations, scales)]
    saved_mean, saved_variance = mean.copy(), variance.copy()
    saved_shapes = np.array(shapes)
    draws = 511
    rng = np.random.default_rng(99)
    actual, diagnostics = softscore_pc(mean, variance, shapes, df, rng, draws)
    expected = {name: np.zeros((11, 2)) for name in ("strength2", "strength4", "mixture")}
    reference_rng = np.random.default_rng(99)
    log_prior = np.log([.15] * 3 + [1 / 30] * 3 + [.45])

    def direct_scores(x, shape, triple):
        directions = []
        for size in (1, 2, 3):
            for selected in combinations(triple, size):
                indicator = np.zeros(4)
                indicator[list(selected)] = 1
                directions.append(indicator / np.sqrt(indicator @ shape @ indicator))
        projected = x @ np.array(directions).T
        terms2 = 2 * projected - 2 + log_prior
        terms4 = 4 * projected - 8 + log_prior
        return (logsumexp(terms2, axis=-1), logsumexp(terms4, axis=-1),
                logsumexp(np.concatenate((terms2, terms4), axis=-1), axis=-1) - np.log(2))

    for fold, shape in enumerate(shapes):
        held = np.arange(len(mean)) % 2 == fold
        scale = np.sqrt(np.diag(shape))
        root = scale[:, None] * covariance_root(shape / np.outer(scale, scale))
        bank = reference_rng.normal(size=(draws, 4)) @ root.T
        if np.isfinite(df):
            bank /= np.sqrt(reference_rng.chisquare(df, size=(draws, 1)) / df)
        x = mean[held] / np.sqrt(variance[held, None, None])
        for triple in combinations(range(4), 3):
            refs = direct_scores(bank, shape, triple)
            obs = direct_scores(x, shape, triple)
            for name, ref, score in zip(expected, refs, obs):
                expected[name][held] = np.maximum(expected[name][held], rank_tail(ref, score))
    assert set(actual) == set(expected)
    for name, expected_p in expected.items():
        np.testing.assert_array_equal(actual[name], expected_p)
        assert actual[name].shape == (11, 2)
        assert np.all((actual[name] >= 1 / (draws + 1)) & (actual[name] <= 1))
    # Exactly one independent bank per fold, including Student radial draws;
    # strengths, signs, and triples must not consume additional randomness.
    np.testing.assert_array_equal(rng.normal(size=8), reference_rng.normal(size=8))
    np.testing.assert_array_equal(mean, saved_mean)
    np.testing.assert_array_equal(variance, saved_variance)
    np.testing.assert_array_equal(shapes, saved_shapes)
    assert diagnostics["draws"] == draws and diagnostics["bank_count"] == 2
    assert diagnostics["min_p"] == 1 / (draws + 1)
    assert diagnostics["elapsed_seconds"] >= 0
    assert diagnostics["calibration"] == ("gaussian" if np.isinf(df) else "student")


def test_known_gaussian_partial_null_mc_bound_2000_samples():
    n, draws = 2000, 40000
    data_rng = np.random.default_rng(8317)
    shapes = [.35 * np.ones((4, 4)) + .65 * np.eye(4),
              1.2 * np.eye(4) - .2 * np.ones((4, 4))]
    variance = np.exp(data_rng.normal(scale=.6, size=n))
    x = np.empty((n, 4))
    for fold, shape in enumerate(shapes):
        x[fold::2] = data_rng.normal(size=(n // 2, 4)) @ np.linalg.cholesky(shape).T
    # At most one positive study is still in the r=2 partial-conjunction null.
    # The remaining triple is exactly on its Gaussian component-null boundary.
    x[:, 0] += 30
    mean = np.stack((x, -x), axis=1) * np.sqrt(variance[:, None, None])
    pvalues, _ = softscore_pc(mean, variance, shapes, np.inf, np.random.default_rng(614), draws)
    for p in pvalues.values():
        for fold in (0, 1):
            for alpha in (.01, .05, .1):
                accepted_ranks = int(np.floor(alpha * (draws + 1)))
                # Under a continuous boundary null, sharing the reference bank
                # makes the rejection count beta-binomial, not binomial. PC's
                # maximum is bounded by the count from the all-null triple.
                upper = betabinom.ppf(.9999, n // 2, accepted_ranks,
                                     draws + 1 - accepted_ranks)
                assert np.count_nonzero(p[fold::2, 0] <= alpha) <= upper
        assert np.count_nonzero(p[:, 0] <= .05) >= 40


def test_default_draws_empty_input_and_single_row():
    empty, diagnostics = softscore_pc(np.empty((0, 2, 4)), np.empty(0),
                                    [np.eye(4)] * 2, np.inf, np.random.default_rng(2))
    assert all(p.shape == (0, 2) for p in empty.values())
    assert diagnostics["draws"] == 131071 and diagnostics["bank_count"] == 0
    one, diagnostics = softscore_pc(np.zeros((1, 2, 4)), np.ones(1),
                                  [np.eye(4)] * 2, 6., np.random.default_rng(2), 1)
    assert all(p.shape == (1, 2) for p in one.values())
    assert diagnostics["bank_count"] == 1


def test_chunk_boundaries_do_not_change_ranks_or_rng_consumption(monkeypatch):
    data_rng = np.random.default_rng(35)
    mean = data_rng.normal(size=(31, 2, 4))
    variance = np.exp(data_rng.normal(size=31))
    shapes = [.4 * np.ones((4, 4)) + .6 * np.eye(4)] * 2
    rng1, rng2 = np.random.default_rng(50), np.random.default_rng(50)
    expected, _ = softscore_pc(mean, variance, shapes, 8., rng1, 127)
    monkeypatch.setattr(softscore_module, "_CHUNK_ROWS", 7)
    actual, _ = softscore_pc(mean, variance, shapes, 8., rng2, 127)
    for name, expected_p in expected.items():
        np.testing.assert_array_equal(actual[name], expected_p)
    np.testing.assert_array_equal(rng1.normal(size=8), rng2.normal(size=8))


@pytest.mark.parametrize("field,value", [
    ("mean", np.zeros((2, 4))), ("mean", np.full((2, 2, 4), np.nan)),
    ("variance", np.ones((2, 1))), ("variance", [1., 0.]),
    ("variance", [1., np.inf]), ("shapes", [np.eye(4)]),
    ("shapes", [np.eye(4), np.eye(4) - 2 * np.ones((4, 4))]),
    ("df", 0), ("df", -np.inf), ("df", np.nan), ("df", [5., 6.]),
    ("draws", 0), ("draws", 1.5), ("draws", True), ("rng", 32),
])
def test_invalid_inputs_fail_before_consuming_rng(field, value):
    rng = np.random.default_rng(48)
    arguments = {"mean": np.zeros((2, 2, 4)), "variance": np.ones(2),
                 "shapes": [np.eye(4)] * 2, "df": np.inf, "rng": rng, "draws": 31}
    arguments[field] = value
    with pytest.raises((ValueError, TypeError)):
        softscore_pc(**arguments)
    assert rng.normal() == np.random.default_rng(48).normal()


def test_zero_subset_variance_is_rejected():
    shape = np.eye(4)
    shape[0, 1] = shape[1, 0] = -1
    with pytest.raises(ValueError, match="subset sum"):
        softscore_pc(np.zeros((2, 2, 4)), np.ones(2), [shape] * 2,
                     np.inf, np.random.default_rng(3), 31)
