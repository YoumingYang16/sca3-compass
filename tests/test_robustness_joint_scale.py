"""Small numerical diagnostics, not simulation evidence for a research claim."""

from inspect import signature

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_joint_scale as joint
from sca3_compass.robustness_patterns import pattern_library, pattern_logpdf


@pytest.fixture(autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1):
        yield


def _training(seed=734, rows=96, df=5., tau=2.5, base=.25):
    rng = np.random.default_rng(seed)
    shape = .55 * np.eye(4) + .45
    variance = base * np.exp(rng.normal(0, .4, rows))
    noise = rng.normal(size=(rows, 4)) @ np.linalg.cholesky(shape).T
    if np.isfinite(df):
        noise /= np.sqrt(rng.chisquare(df, rows) / df)[:, None]
    means = noise * np.sqrt(tau * variance[:, None])
    means[:rows // 3] += [3.5, 3.5, 0., 0.]
    means[rows // 3:rows // 2] -= 4.5
    return means, variance, shape, df


@pytest.mark.parametrize("df", [.2, 1.5, 5., 100., np.inf])
@pytest.mark.parametrize("tau", [.007, 1., 117.])
def test_normalized_densities_match_scipy_and_existing_fitter(df, tau):
    means, variance, shape, _ = _training(rows=5)
    density = joint.joint_pattern_logpdf(means, variance, shape, df, tau)
    patterns, amplitudes = pattern_library()
    for c in (0, 1, 74, 160, 240):
        location = patterns[c] * amplitudes[c]
        for i in range(len(means)):
            expected = (multivariate_normal.logpdf(means[i], mean=location, cov=tau * variance[i] * shape)
                        if np.isinf(df) else multivariate_t.logpdf(means[i], loc=location, shape=tau * variance[i] * shape, df=df))
            assert density[i, c] == pytest.approx(expected, rel=5e-13, abs=2e-10)
    np.testing.assert_allclose(density, pattern_logpdf(means, tau * variance, shape, df), rtol=4e-13, atol=2e-10)


@pytest.mark.parametrize("df", [.2, 5., 1e12, np.inf])
def test_component_log_tau_analytic_gradient_by_central_differences(df):
    log_q = np.array([[-np.inf, -15., -1., 0., 2., 10., 30.]])
    log_tau, step = .73, 1e-5
    kernel, radial = joint._kernel(log_q, log_tau, df)
    above = -2 * (log_tau + step) - joint._kernel(log_q, log_tau + step, df)[0]
    below = -2 * (log_tau - step) - joint._kernel(log_q, log_tau - step, df)[0]
    np.testing.assert_allclose(-2 + radial, (above - below) / (2 * step), rtol=2e-9, atol=2e-8)
    assert np.isfinite(kernel).all()


@pytest.mark.parametrize("df", [1.5, 5., np.inf])
def test_full_map_weight_and_log_tau_gradients(df):
    means, variance, root, df = joint._inputs(*_training(rows=19, df=df))
    problem = joint._Problem(means, variance, root, df, 8)
    weights = np.random.default_rng(637).dirichlet(np.full(241, 4.))
    log_tau, step = .41, 1e-6
    state = problem.state(weights, log_tau)
    numerical_tau = (problem.state(weights, log_tau + step)["objective"]
                     - problem.state(weights, log_tau - step)["objective"]) / (2 * step)
    assert state["log_tau_gradient"] == pytest.approx(numerical_tau, rel=2e-8, abs=2e-7)
    gradient = (state["counts"] + .02) / weights
    for c in (0, 4, 75, 160, 239):
        direction = np.zeros(241)
        direction[c], direction[240] = 1., -1.
        numerical = (problem.state(weights + step * direction, log_tau)["objective"]
                     - problem.state(weights - step * direction, log_tau)["objective"]) / (2 * step)
        assert gradient[c] - gradient[240] == pytest.approx(numerical, rel=2e-6, abs=5e-6)


@pytest.mark.parametrize("df", [1.5, 5., np.inf])
def test_converged_map_objective_and_both_stationarity_checks(df):
    data = _training(df=df)
    before = [a.copy() for a in data[:3]]
    fit = joint.fit_joint_pattern_scale(*data)
    diag, weights = fit["diagnostics"], fit["weights"]
    assert diag["converged"]
    assert np.all(weights > 0) and weights.sum() == pytest.approx(1., abs=2e-15)
    density = joint.joint_pattern_logpdf(*data, tau=fit["tau"])
    row_normalizer = logsumexp(density + np.log(weights), axis=1)
    objective = row_normalizer.sum() + .02 * np.log(weights).sum()
    assert diag["objective"] == pytest.approx(objective, abs=3e-10)
    assert diag["weight_kkt_residual"] <= diag["tolerance"]
    assert diag["projected_log_tau_residual"] <= diag["tolerance"]
    counts = np.exp(density + np.log(weights) - row_normalizer[:, None]).sum(axis=0)
    expected_kkt = np.max(np.abs((counts + .02) / weights / (len(data[0]) + 241 * .02) - 1))
    assert diag["weight_kkt_residual"] == pytest.approx(expected_kkt, abs=1e-12)
    assert diag["pseudocount"] == .02 and diag["dirichlet_concentration"] == 1.02
    assert len(diag["starts"]) == 4
    for start in diag["starts"]:
        assert start["failure"] is None
        assert start["status"] in ("converged", "iteration_limit")
        assert 1e-3 <= start["tau"] <= 1e3
        assert start["objective_history"][-1] == start["objective"]
        assert start["tau_history"][-1] == start["tau"]
        assert len(start["objective_history"]) == start["iterations"] + 1
        assert np.diff(start["objective_history"]).min(initial=0) >= -1e-9
        assert len(start["weights"]) == 241
        assert start["weight_summary"]["null_mass"] == start["weights"][0]
    assert diag["objective"] == max(s["objective"] for s in diag["starts"])
    for original, current in zip(before, data[:3]):
        np.testing.assert_array_equal(original, current)


@pytest.mark.parametrize("df", [5., np.inf])
def test_small_known_synthetic_mixture_recovers_scale_and_support_mass(df):
    # Fixed locations and separated noise make this a unit diagnostic only.
    fit = joint.fit_joint_pattern_scale(*_training(seed=605, rows=480, df=df, tau=3., base=.03))
    assert fit["diagnostics"]["converged"]
    assert fit["tau"] == pytest.approx(3., rel=.13)
    weights, patterns, amplitudes = fit["weights"], fit["patterns"], fit["amplitudes"]
    positive = ((patterns == [1, 1, 0, 0]).all(axis=1)) & (amplitudes == 3.5)
    negative = ((patterns == [-1, -1, -1, -1]).all(axis=1)) & (amplitudes == 4.5)
    assert weights[0] == pytest.approx(.5, abs=.04)
    assert weights[positive].sum() == pytest.approx(1 / 3, abs=.04)
    assert weights[negative].sum() == pytest.approx(1 / 6, abs=.04)


@pytest.mark.parametrize("df,tau", [(5., .006), (np.inf, 160.)])
def test_training_null_allows_large_scale_mismatch(df, tau):
    means, variance, shape, _ = _training(seed=732, rows=256, df=df, tau=tau, base=.01 / tau)
    means[:256 // 3] -= [3.5, 3.5, 0., 0.]
    means[256 // 3:256 // 2] += 4.5
    fit = joint.fit_joint_pattern_scale(means, variance, shape, df)
    assert fit["diagnostics"]["converged"]
    assert fit["tau"] == pytest.approx(tau, rel=.15)
    assert fit["weights"][0] > .95


def test_base_scale_equivariance_and_shape_factorization():
    means, variance, shape, df = _training(rows=64)
    first = joint.fit_joint_pattern_scale(means, variance, shape, df)
    # The absolute location grid stays fixed. Changing means' physical units
    # alone is NOT an equivariance of this fixed-location model.
    scaled = joint.fit_joint_pattern_scale(means, 7 * variance, shape, df, tau_bounds=(1e-3 / 7, 1e3 / 7))
    refactored = joint.fit_joint_pattern_scale(means, variance / 3, shape * 3, df)
    assert first["diagnostics"]["converged"] and scaled["diagnostics"]["converged"]
    assert scaled["tau"] * 7 == pytest.approx(first["tau"], rel=1e-5)
    assert refactored["tau"] == pytest.approx(first["tau"], rel=1e-5)
    np.testing.assert_allclose(first["weights"], scaled["weights"], rtol=2e-4, atol=2e-6)
    np.testing.assert_allclose(first["weights"], refactored["weights"], rtol=2e-4, atol=2e-6)


@pytest.mark.parametrize("max_iter", [1, 250])
def test_training_row_permutation_is_bitwise_deterministic(max_iter):
    means, variance, shape, df = _training(rows=64)
    order = np.random.default_rng(35).permutation(len(means))
    first = joint.fit_joint_pattern_scale(means, variance, shape, df, max_iter=max_iter)
    second = joint.fit_joint_pattern_scale(means[order], variance[order], shape, df, max_iter=max_iter)
    assert first["tau"] == second["tau"]
    np.testing.assert_array_equal(first["weights"], second["weights"])
    assert first["diagnostics"] == second["diagnostics"]


def test_iteration_limit_is_not_convergence_and_weight_check_is_required():
    fit = joint.fit_joint_pattern_scale(*_training(), max_iter=1, accelerate=False)
    assert not fit["diagnostics"]["converged"]
    assert all(s["iterations"] == 1 and s["em_updates"] == 1 for s in fit["diagnostics"]["starts"])
    state = {"log_tau_gradient": 0., "weight_kkt_residual": .1, "conditional_weight_gap": 1.}
    assert not joint._certificate(state, 0., (-2., 2.), 10, 1e-6)["converged"]
    state.update(log_tau_gradient=1., weight_kkt_residual=0.)
    assert not joint._certificate(state, 0., (-2., 2.), 10, 1e-6)["converged"]


@pytest.mark.parametrize("upper", [False, True])
def test_bound_optima_use_correct_one_sided_scale_stationarity(upper):
    means = np.full((12, 4), 1e4) if upper else np.zeros((12, 4))
    fit = joint.fit_joint_pattern_scale(means, np.ones(12), np.eye(4), np.inf, tau_bounds=(.01, 2.))
    diag = fit["diagnostics"]
    assert diag["converged"]
    assert diag["active_bound"] == ("upper" if upper else "lower")
    assert fit["tau"] == pytest.approx(2. if upper else .01)
    assert diag["projected_log_tau_residual"] == 0
    assert diag["log_tau_gradient"] > 0 if upper else diag["log_tau_gradient"] < 0


def test_extreme_student_tails_and_gaussian_all_start_failure_are_explicit():
    means = np.full((3, 4), 1e200)
    variance = np.full(3, 1e-100)
    density = joint.joint_pattern_logpdf(means, variance, np.eye(4), 1.5, .1)
    assert np.isfinite(density).all()
    student = joint.fit_joint_pattern_scale(means, variance, np.eye(4), 1.5)
    assert student["diagnostics"]["converged"] and student["diagnostics"]["active_bound"] == "upper"
    gaussian = joint.fit_joint_pattern_scale(means, variance, np.eye(4), np.inf)
    assert gaussian["tau"] is None and gaussian["weights"] is None
    assert gaussian["diagnostics"]["status"] == "all_starts_failed"
    assert all("no representable component" in s["failure"] for s in gaussian["diagnostics"]["starts"])


def test_one_start_failure_does_not_hide_successful_starts(monkeypatch):
    actual = joint._Problem.state
    calls = 0

    def fail_first(self, weights, log_tau):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("synthetic initial failure")
        return actual(self, weights, log_tau)

    monkeypatch.setattr(joint._Problem, "state", fail_first)
    fit = joint.fit_joint_pattern_scale(*_training(rows=32))
    assert fit["diagnostics"]["starts"][0]["status"] == "failed"
    assert fit["diagnostics"]["selected_start"] in (1, 2, 3)
    assert fit["tau"] is not None
    assert fit["diagnostics"]["numerical_failure_count"] == 1
    assert fit["diagnostics"]["fatal_solver_failure_count"] == 1
    assert fit["diagnostics"]["rejected_acceleration_numeric_count"] == 0


def test_objective_constant_cancellation_does_not_change_stationarity(monkeypatch):
    data=_training(rows=128)
    baseline=joint.fit_joint_pattern_scale(*data)
    actual=joint._Problem.__init__
    def cancel_constant(self,*args,**kwargs):
        actual(self,*args,**kwargs)
        self.constant-=baseline['diagnostics']['objective']/self.n
    monkeypatch.setattr(joint._Problem,'__init__',cancel_constant)
    shifted=joint.fit_joint_pattern_scale(*data)
    assert shifted['diagnostics']['numerical_failure_count']==0
    assert shifted['diagnostics']['converged']
    assert abs(shifted['diagnostics']['objective'])<1e-7
    np.testing.assert_allclose(shifted['tau'],baseline['tau'],rtol=2e-6)
    assert joint._objective_roundoff({'objective_absolute_sum':1000.},{'objective_absolute_sum':1100.})>1e-12


def test_recovered_acceleration_failure_is_still_counted(monkeypatch):
    actual = joint._Problem.state
    calls = 0

    def fail_first_extrapolation(self, weights, log_tau):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise FloatingPointError("synthetic rejected acceleration")
        return actual(self, weights, log_tau)

    monkeypatch.setattr(joint._Problem, "state", fail_first_extrapolation)
    fit = joint.fit_joint_pattern_scale(*_training(rows=32))
    diag, start = fit["diagnostics"], fit["diagnostics"]["starts"][0]
    assert start["converged"] and start["failure"] is None
    assert start["numerical_failure_count"] == 1
    assert start["rejected_acceleration_numeric_count"] == 1
    assert start["fatal_solver_failure_count"] == 0
    assert start["numerical_failure_events"][0]["stage"] == "acceleration"
    assert diag["numerical_failure_count"] == diag["rejected_acceleration_numeric_count"] == 1
    assert diag["fatal_solver_failure_count"] == 0
    assert "numerical_failures" not in diag and "numerical_failures" not in start


def test_fatal_failure_after_partial_em_retains_last_finite_iterate(monkeypatch):
    actual = joint._Problem.state
    calls = 0

    def fail_second_em(self, weights, log_tau):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise FloatingPointError("synthetic second EM failure")
        return actual(self, weights, log_tau)

    monkeypatch.setattr(joint._Problem, "state", fail_second_em)
    start = joint.fit_joint_pattern_scale(*_training(rows=32))["diagnostics"]["starts"][0]
    assert start["status"] == "failed" and not start["converged"]
    assert start["objective"] == start["objective_history"][-1]
    assert start["tau"] == start["tau_history"][-1]
    assert start["numerical_failure_events"][0]["stage"] == "em"
    assert start["fatal_solver_failure_count"] == start["numerical_failure_count"] == 1


def test_returned_tau_and_all_start_histories_respect_exact_decimal_bounds():
    fit = joint.fit_joint_pattern_scale(np.full((8, 4), 100.), np.ones(8), np.eye(4), np.inf, tau_bounds=(.03, .1))
    assert fit["tau"] <= .1
    for start in fit["diagnostics"]["starts"]:
        assert start["tau_bounds"] == [.03, .1]
        assert .03 <= start["initial_tau"] <= .1
        assert all(.03 <= tau <= .1 for tau in start["tau_history"])


def test_uncached_batches_and_cached_likelihood_agree(monkeypatch):
    data = _training(rows=43)
    cached = joint.fit_joint_pattern_scale(*data, max_iter=3, batch_size=8)
    monkeypatch.setattr(joint, "_CACHE_ELEMENTS", 0)
    streamed = joint.fit_joint_pattern_scale(*data, max_iter=3, batch_size=8)
    assert streamed["diagnostics"]["distance_cache_bytes"] == 0
    assert cached["diagnostics"]["distance_cache_bytes"] <= 8 * 1024**2
    assert streamed["tau"] == pytest.approx(cached["tau"], rel=2e-12)
    np.testing.assert_allclose(streamed["weights"], cached["weights"], rtol=1e-11, atol=1e-13)


def test_api_has_only_training_inputs_and_numerical_controls():
    assert list(signature(joint.fit_joint_pattern_scale).parameters) == [
        "means", "base_variance", "shape", "df", "tau_bounds", "max_iter", "tol", "batch_size", "accelerate",
    ]
    data = _training(rows=8)
    fit = joint.fit_joint_pattern_scale(*data, max_iter=0)
    assert not fit["diagnostics"]["uses_truth_labels"]
    assert not fit["diagnostics"]["held_gene_access"]
    assert fit["diagnostics"]["requires_caller_training_only_inputs"]
    for name in ("labels", "held_means", "calibration_params", "oracle_tail"):
        with pytest.raises(TypeError):
            joint.fit_joint_pattern_scale(*data, **{name: None})


@pytest.mark.parametrize("invalid", ["empty", "columns", "nan_mean", "inf_mean", "scalar_base", "short_base", "zero_base", "negative_base", "nan_base", "inf_base", "wrong_shape", "asymmetric", "singular", "indefinite", "nan_shape"])
def test_invalid_training_data(invalid):
    means, variance, shape, df = _training(rows=8)
    if invalid == "empty":
        means, variance = means[:0], variance[:0]
    elif invalid == "columns":
        means = means[:, :3]
    elif invalid.endswith("_mean"):
        means[0, 0] = np.nan if invalid == "nan_mean" else np.inf
    elif invalid == "scalar_base":
        variance = 1.
    elif invalid == "short_base":
        variance = variance[:-1]
    elif invalid.endswith("_base"):
        variance[0] = {"zero_base": 0., "negative_base": -1., "nan_base": np.nan, "inf_base": np.inf}[invalid]
    elif invalid == "wrong_shape":
        shape = np.eye(3)
    elif invalid == "asymmetric":
        shape[0, 1] += .1
    elif invalid == "singular":
        shape = np.ones((4, 4))
    elif invalid == "indefinite":
        shape[0, 0] = -1.
    else:
        shape[0, 0] = np.nan
    with pytest.raises(ValueError):
        joint.fit_joint_pattern_scale(means, variance, shape, df)


@pytest.mark.parametrize("df", [0., -1., np.nan, -np.inf, [5.], True, "bad"])
def test_invalid_df(df):
    means, variance, shape, _ = _training(rows=4)
    with pytest.raises(ValueError):
        joint.fit_joint_pattern_scale(means, variance, shape, df)


@pytest.mark.parametrize("kwargs", [
    {"tau_bounds": (0, 1)}, {"tau_bounds": (-1, 1)}, {"tau_bounds": (1, np.inf)},
    {"tau_bounds": (np.nan, 1)}, {"tau_bounds": (1, 1)}, {"tau_bounds": (2, 1)},
    {"tau_bounds": (1,)}, {"tau_bounds": 1}, {"tau_bounds": (True, 2)},
    {"tol": 0}, {"tol": np.inf}, {"tol": np.nan}, {"tol": [1]},
    {"max_iter": -1}, {"max_iter": 1.5}, {"max_iter": True},
    {"batch_size": 0}, {"batch_size": 2.5}, {"batch_size": True}, {"accelerate": 1},
])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        joint.fit_joint_pattern_scale(*_training(rows=4), **kwargs)


@pytest.mark.parametrize("tau", [0., -1., np.inf, np.nan, [1.], True])
def test_invalid_density_tau(tau):
    with pytest.raises(ValueError):
        joint.joint_pattern_logpdf(*_training(rows=4), tau=tau)
