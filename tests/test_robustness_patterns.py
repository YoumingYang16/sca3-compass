from inspect import signature

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, check_grad
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, multivariate_t
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_patterns as pattern_module
from sca3_compass.robustness_patterns import (
    fit_pattern_mixture,
    pattern_library,
    pattern_logpdf,
)


@pytest.fixture(autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1):
        yield


def _training(seed=743, rows=128, df=5.):
    rng = np.random.default_rng(seed)
    shape = .35 * np.eye(4) + .65
    variance = np.exp(rng.normal(scale=.6, size=rows))
    noise = rng.normal(size=(rows, 4)) @ np.linalg.cholesky(shape).T
    if np.isfinite(df):
        noise /= np.sqrt(rng.chisquare(df, size=rows) / df)[:, None]
    means = noise * np.sqrt(variance[:, None])
    means[:rows // 3] += [3.5, 3.5, 0., 0.]
    means[rows // 3:rows // 2] -= 3.5
    return means, variance, shape, df


def test_library_has_all_signed_supports_and_no_duplicate_null():
    patterns, amplitudes = pattern_library()
    assert patterns.shape == (241, 4) and amplitudes.shape == (241,)
    np.testing.assert_array_equal(patterns[0], 0)
    assert amplitudes[0] == 0
    assert np.count_nonzero((patterns == 0).all(axis=1)) == 1
    assert len(np.unique(patterns, axis=0)) == 81
    assert len(np.unique(patterns * amplitudes[:, None], axis=0)) == 241
    for support in np.unique(patterns[1:], axis=0):
        np.testing.assert_array_equal(amplitudes[(patterns == support).all(axis=1)], [2.5, 3.5, 4.5])
    patterns[:] = 9
    amplitudes[:] = 8
    fresh_patterns, fresh_amplitudes = pattern_library()
    assert fresh_patterns.max() == 1 and fresh_amplitudes.max() == 4.5


@pytest.mark.parametrize("df", [.2, 1.5, 5., 25., 200., np.inf])
def test_density_matches_scipy_multivariate_t_and_gaussian(df):
    rng = np.random.default_rng(46)
    raw = rng.normal(size=(4, 4))
    shape = raw @ raw.T + np.diag([.3, 1., 2., 4.])
    means = rng.normal(size=(7, 4)) * 5
    variance = np.geomspace(.01, 100., len(means))
    patterns, amplitudes = pattern_library()
    density = pattern_logpdf(means, variance, shape, df)
    reference = np.empty_like(density)
    for c, location in enumerate(patterns * amplitudes[:, None]):
        for i in range(len(means)):
            reference[i, c] = (
                multivariate_normal.logpdf(means[i], mean=location, cov=variance[i] * shape)
                if np.isinf(df) else
                multivariate_t.logpdf(means[i], loc=location, shape=variance[i] * shape, df=df)
            )
    np.testing.assert_allclose(density, reference, rtol=3e-13, atol=3e-12)


def test_exact_component_location_and_large_df_limit():
    patterns, amplitudes = pattern_library()
    means = patterns * amplitudes[:, None]
    variance = np.ones(len(means))
    gaussian = pattern_logpdf(means, variance, np.eye(4), np.inf)
    student = pattern_logpdf(means, variance, np.eye(4), 1e12)
    np.testing.assert_allclose(student, gaussian, rtol=1e-9, atol=2e-11)
    np.testing.assert_allclose(np.diag(gaussian), -2 * np.log(2 * np.pi), atol=1e-14)


def test_extreme_student_rows_are_not_trimmed_or_overflowed():
    means = np.array([[1e200, -1e200, 1e200, -1e200], [1e-200, 0., 0., 0.]])
    variance = np.array([1e-100, 1e100])
    density = pattern_logpdf(means, variance, np.eye(4), 1.5)
    assert np.isfinite(density).all()
    fit = fit_pattern_mixture(means, variance, np.eye(4), 1.5)
    assert fit['diagnostics']['training_rows'] == 2
    assert fit['diagnostics']['converged']
    np.testing.assert_allclose(fit['weights'], 1 / 241, atol=1e-12)
    with pytest.raises(FloatingPointError, match="no representable component"):
        fit_pattern_mixture(means, variance, np.eye(4), np.inf)


@pytest.mark.parametrize("df", [1.5, 5., 25., np.inf])
def test_map_objective_monotonicity_weight_simplex_and_gap(df):
    means, variance, shape, _ = _training(df=df)
    before = [x.copy() for x in (means, variance, shape)]
    fit = fit_pattern_mixture(means, variance, shape, df)
    diag, weights = fit['diagnostics'], fit['weights']
    assert diag['converged'] and diag['em_monotone']
    assert 0 <= diag['em_iterations'] <= 500
    assert diag['relative_duality_gap'] <= diag['tolerance']
    assert np.isfinite(weights).all() and np.all(weights > 0)
    assert weights.sum() == pytest.approx(1., abs=3e-15)
    assert np.diff(diag['objective_history']).min(initial=0) >= -2e-11
    density = pattern_logpdf(means, variance, shape, df)
    joint = density + np.log(weights)
    normalizer = logsumexp(joint, axis=1)
    expected_objective = normalizer.sum() + .02 * np.log(weights).sum()
    expected_gradient = (np.exp(joint - normalizer[:, None]).sum(axis=0) + .02) / weights
    expected_gap = expected_gradient.max() - np.dot(weights, expected_gradient)
    assert diag['objective'] == pytest.approx(expected_objective, abs=1e-10)
    assert diag['objective_history'][-1] == diag['objective']
    assert diag['duality_gap'] == pytest.approx(expected_gap, abs=2e-10)
    assert diag['pseudocount'] == .02 and diag['dirichlet_concentration'] == 1.02
    for original, current in zip(before, (means, variance, shape)):
        np.testing.assert_array_equal(original, current)


def test_low_iteration_budget_is_reported_as_unconverged():
    fit = fit_pattern_mixture(*_training(), max_iter=1, slsqp_max_iter=0,newton_max_iter=0)
    diag = fit['diagnostics']
    assert not diag['converged'] and diag['em_iterations'] == 1
    assert diag['relative_duality_gap'] > diag['tolerance']
    assert not diag['slsqp']['attempted']
    assert len(diag['objective_history']) == 2


def test_slsqp_polishing_is_monotone_and_not_just_optimizer_success():
    data = _training(rows=96)
    fit = fit_pattern_mixture(*data, max_iter=1, slsqp_max_iter=200, tol=1e-6)
    diag = fit['diagnostics']
    assert diag['converged'] and diag['slsqp']['attempted'] and diag['slsqp']['accepted']
    assert diag['em_iterations'] == 1 and diag['slsqp']['iterations'] <= 200
    assert diag['em_relative_duality_gap'] > diag['relative_duality_gap']
    assert np.diff(diag['objective_history']).min(initial=0) >= -2e-11
    em = fit_pattern_mixture(*data, max_iter=500, slsqp_max_iter=0, tol=1e-6)
    np.testing.assert_allclose(fit['weights'], em['weights'], rtol=2e-3, atol=3e-6)


def test_optimizer_success_without_stationarity_is_not_convergence(monkeypatch):
    def no_progress(fun, x0, **kwargs):
        return OptimizeResult(x=x0.copy(), success=True, nit=1, message="simulated success")
    monkeypatch.setattr(pattern_module, 'minimize', no_progress)
    diag = fit_pattern_mixture(*_training(), max_iter=0,newton_max_iter=0)['diagnostics']
    assert diag['slsqp']['success']
    assert not diag['slsqp']['accepted'] and not diag['converged']


def test_default_500_em_budget_can_require_a_certified_bounded_polish():
    # This observed development case exhausts 500 EM updates. Keep it as a
    # numerical regression, not an independent inferential validation result.
    rng = np.random.default_rng(6283)
    n, df = 512, 1.5
    shape = .35 * np.eye(4) + .65
    variance = np.exp(rng.normal(0, 1, n))
    noise = rng.normal(size=(n, 4)) @ np.linalg.cholesky(shape).T
    noise /= np.sqrt(rng.chisquare(df, size=n) / df)[:, None]
    means = noise * np.sqrt(variance[:, None])
    means[:n//5] += [3.5, 3.5, 0, 0]
    means[n//5:2*n//5] -= 4.5
    fit = fit_pattern_mixture(means, variance, shape, df)
    diag = fit['diagnostics']
    assert diag['em_iterations'] == 500
    assert diag['em_relative_duality_gap'] > diag['tolerance']
    assert diag['slsqp']['attempted'] and diag['slsqp']['iterations'] <= 200
    assert diag['converged'] and diag['relative_duality_gap'] <= diag['tolerance']
    assert np.diff(diag['objective_history']).min(initial=0) >= -2e-11


def test_slsqp_analytic_gradient_and_feasible_improvement_tracking(monkeypatch):
    actual_minimize = pattern_module.minimize

    def checked_minimize(fun, x0, **kwargs):
        rng = np.random.default_rng(142)
        w = rng.dirichlet(np.full(len(x0), 3.))
        assert check_grad(lambda x: fun(x)[0], lambda x: fun(x)[1], w) < 3e-4
        assert kwargs['method'] == 'SLSQP' and kwargs['jac']
        return actual_minimize(fun, x0, **kwargs)

    monkeypatch.setattr(pattern_module, 'minimize', checked_minimize)
    diag = fit_pattern_mixture(*_training(rows=64), max_iter=1, tol=1e-6)['diagnostics']
    assert diag['slsqp']['attempted'] and diag['converged']


@pytest.mark.parametrize('max_iter', [1, 500])
def test_gene_row_permutation_is_bitwise_deterministic(max_iter):
    means, variance, shape, df = _training(rows=96)
    permutation = np.random.default_rng(946).permutation(len(means))
    first = fit_pattern_mixture(means, variance, shape, df, max_iter=max_iter, tol=1e-6)
    second = fit_pattern_mixture(means[permutation], variance[permutation], shape, df, max_iter=max_iter, tol=1e-6)
    np.testing.assert_array_equal(first['weights'], second['weights'])
    assert first['diagnostics'] == second['diagnostics']
    for sign in ('positive', 'negative'):
        for field in first['replicated'][sign]:
            np.testing.assert_array_equal(first['replicated'][sign][field], second['replicated'][sign][field])


@pytest.mark.parametrize('support', [[1, 0, 0, 0], [1, 1, 0, 0], [-1, -1, -1, -1]])
@pytest.mark.parametrize('df', [5., np.inf])
def test_synthetic_singleton_is_not_learned_as_replication(support, df):
    rng = np.random.default_rng(5097)
    n = 240
    variance = np.full(n, .04)
    noise = rng.normal(scale=.2, size=(n, 4))
    if np.isfinite(df):
        noise /= np.sqrt(rng.chisquare(df, size=n) / df)[:, None]
    means = 3.5 * np.asarray(support) + noise
    fit = fit_pattern_mixture(means, variance, np.eye(4), df)
    assert fit['diagnostics']['converged']
    support_mask = (fit['patterns'] == support).all(axis=1)
    assert fit['weights'][support_mask].sum() > .94
    for name, sign in [('positive', 1), ('negative', -1)]:
        summary = fit['replicated'][name]
        is_replicated = np.count_nonzero(np.asarray(support) == sign) >= 2
        if is_replicated:
            assert summary['mass'] > .94 and summary['dominant_mass_fraction'] > .94
            np.testing.assert_array_equal(summary['dominant_pattern'], support)
            assert summary['dominant_amplitude'] == 3.5
        else:
            assert summary['mass'] < .04


def test_dominant_pattern_aggregates_amplitude_uncertainty():
    patterns, amplitudes = pattern_library()
    weights = np.full(len(patterns), 1e-8)
    support_a, support_b = [1, 1, 0, 0], [1, 1, 1, 0]
    a = np.flatnonzero((patterns == support_a).all(axis=1))
    b = np.flatnonzero((patterns == support_b).all(axis=1))
    weights[a] = [.2, .2, .2]
    weights[b] = [.39, .005, .005]
    weights /= weights.sum()
    summary = pattern_module._replicated_summary(weights, patterns, amplitudes, 1)
    np.testing.assert_array_equal(summary['dominant_pattern'], support_a)
    np.testing.assert_array_equal(summary['component_dominant_pattern'], support_b)
    assert summary['dominant_mass'] == pytest.approx(weights[a].sum())
    assert summary['dominant_mass_fraction'] > .599
    assert summary['component_dominant_mass_fraction'] < .391
    np.testing.assert_allclose(summary['dominant_amplitude_weights'], 1/3)
    assert summary['dominant_amplitude'] == 2.5  # deterministic ascending-amplitude tie


def test_sign_summaries_contain_only_replicated_alternatives_and_overlap_is_explicit():
    fit = fit_pattern_mixture(*_training())
    patterns, weights = fit['patterns'], fit['weights']
    for name, sign in [('positive', 1), ('negative', -1)]:
        summary = fit['replicated'][name]
        selected = (patterns == sign).sum(axis=1) >= 2
        assert len(summary['component_indices']) == 99
        assert len(summary['support_patterns']) == 33
        np.testing.assert_array_equal(summary['component_indices'], np.flatnonzero(selected))
        assert summary['mass'] == pytest.approx(weights[selected].sum())
        assert summary['conditional_weights'].sum() == pytest.approx(1.)
        assert summary['support_conditional_weights'].sum() == pytest.approx(1.)
        assert summary['support_masses'].sum() == pytest.approx(summary['mass'])
        for prefix, count in [('', 33), ('component_', 99)]:
            assert 0 <= summary[prefix + 'normalized_entropy'] <= 1 + 1e-14
            assert 1 <= summary[prefix + 'effective_count'] <= count + 1e-12
            assert summary[prefix + 'effective_count'] == pytest.approx(np.exp(summary[prefix + 'entropy']))
    positive = set(fit['replicated']['positive']['component_indices'])
    negative = set(fit['replicated']['negative']['component_indices'])
    assert len(positive & negative) == 18


def test_training_only_call_has_no_held_gene_or_label_dependencies():
    means, variance, shape, df = _training(rows=100)
    training = np.arange(len(means)) % 2 == 0
    original = fit_pattern_mixture(means[training], variance[training], shape, df)
    means[~training] += 1e6
    variance[~training] *= 1e12
    repeated = fit_pattern_mixture(means[training], variance[training], shape, df)
    np.testing.assert_array_equal(original['weights'], repeated['weights'])
    assert original['diagnostics'] == repeated['diagnostics']
    parameters = list(signature(fit_pattern_mixture).parameters)
    assert parameters == ['means', 'variance', 'shape', 'df', 'pseudocount', 'max_iter', 'tol', 'slsqp_max_iter','newton_max_iter']
    assert not original['diagnostics']['uses_truth_labels']
    assert original['diagnostics']['requires_caller_training_only_inputs']
    with pytest.raises(TypeError):
        fit_pattern_mixture(means[training], variance[training], shape, df, labels=np.ones(training.sum()))


@pytest.mark.parametrize('invalid', [
    'empty', 'wrong_dimension', 'nan_mean', 'infinite_mean', 'scalar_variance',
    'short_variance', 'zero_variance', 'negative_variance', 'nan_variance',
    'infinite_variance', 'wrong_shape', 'asymmetric', 'singular', 'indefinite',
    'nan_shape', 'zero_df', 'negative_df', 'nan_df', 'array_df',
])
def test_invalid_data_or_known_model_are_rejected(invalid):
    means, variance, shape, df = _training(rows=12)
    if invalid == 'empty':
        means, variance = np.empty((0, 4)), np.empty(0)
    elif invalid == 'wrong_dimension':
        means = means[:, :3]
    elif invalid == 'nan_mean':
        means[0, 0] = np.nan
    elif invalid == 'infinite_mean':
        means[0, 0] = np.inf
    elif invalid == 'scalar_variance':
        variance = 1.
    elif invalid == 'short_variance':
        variance = variance[:-1]
    elif invalid.endswith('_variance'):
        variance[0] = {'zero_variance': 0., 'negative_variance': -1., 'nan_variance': np.nan,
                       'infinite_variance': np.inf}[invalid]
    elif invalid == 'wrong_shape':
        shape = np.eye(3)
    elif invalid == 'asymmetric':
        shape[0, 1] += .2
    elif invalid == 'singular':
        shape = np.ones((4, 4))
    elif invalid == 'indefinite':
        shape[0, 0] = -1
    elif invalid == 'nan_shape':
        shape[0, 0] = np.nan
    else:
        df = {'zero_df': 0., 'negative_df': -1., 'nan_df': np.nan, 'array_df': np.array([5.])}[invalid]
    for function in (fit_pattern_mixture, pattern_logpdf):
        with pytest.raises(ValueError):
            function(means, variance, shape, df)


@pytest.mark.parametrize('kwargs', [
    {'pseudocount': 0}, {'pseudocount': -1}, {'pseudocount': np.inf}, {'pseudocount': np.nan},
    {'pseudocount': [1]}, {'tol': 0}, {'tol': np.nan}, {'tol': np.inf},
    {'max_iter': -1}, {'max_iter': 1.5}, {'max_iter': True},
    {'slsqp_max_iter': -1}, {'slsqp_max_iter': .5}, {'slsqp_max_iter': False},
])
def test_invalid_optimization_controls_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_pattern_mixture(*_training(rows=4), **kwargs)
