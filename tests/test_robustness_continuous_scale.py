"""Isolated numerical/unit tests, not power/FDR/independent research evidence."""

import inspect
import json

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp, roots_genlaguerre
from scipy.stats import multivariate_normal, multivariate_t
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_continuous_scale as cs


@pytest.fixture(autouse=True)
def one_blas_thread():
    with threadpool_limits(limits=1):
        yield


def _data(seed=723, n=48, df=5., tau=1.7):
    rng = np.random.default_rng(seed)
    shape = .6*np.eye(4)+.4
    v = np.exp(rng.normal(-.4, .6, n))
    residual = rng.normal(size=(n, 4)) @ np.linalg.cholesky(shape).T
    if np.isfinite(df):
        residual /= np.sqrt(rng.chisquare(df, n)/df)[:, None]
    x = residual*np.sqrt(tau*v[:, None])
    signal = rng.random(n) < .35
    x[signal] += rng.normal(size=(int(signal.sum()), 4))*3
    return x, v, shape, df


SMALL_LIBRARY = np.array([np.zeros((4, 4)), 9*np.eye(4), 9*np.ones((4, 4))])


def test_library_count_psd_uniqueness_and_sign_quotient():
    u, metadata = cs.covariance_library()
    assert u.shape == (154, 4, 4)
    assert len(metadata) == len(u)
    assert len({a.tobytes() for a in u}) == len(u)
    assert np.linalg.eigvalsh(u).min() > -1e-12
    assert sum(m['kind'] == 'rank_one' for m in metadata) == 120
    assert sum(m['kind'] == 'diagonal' for m in metadata) == 33
    for m in metadata[1:]:
        if m['kind'] == 'rank_one':
            assert next(value for value in m['support'] if value) == 1
    assert cs.covariance_library((3., 1.), diagonal=False)[0].shape == (81, 4, 4)


@pytest.mark.parametrize('shape', [.05, .75, 2.5, 100., 5e7])
@pytest.mark.parametrize('order', [16, 64])
def test_normalized_laguerre_moments_without_gamma_overflow(shape, order):
    x, w = cs._gamma_rule(order, shape)
    assert w.sum() == pytest.approx(1., abs=3e-15)
    assert w @ x == pytest.approx(shape, rel=2e-12)
    assert w @ ((x-shape)/np.sqrt(shape))**2 == pytest.approx(1., rel=2e-8)
    if shape < 101:
        xp, wp = roots_genlaguerre(order, shape-1)
        np.testing.assert_allclose(x, xp, rtol=2e-11, atol=2e-12)
        np.testing.assert_allclose(w, wp/wp.sum(), atol=2e-13)


@pytest.mark.parametrize('tau', [.003, 1.3, 100.])
def test_gaussian_convolution_matches_full_covariance_formula(tau):
    x, v, shape, _ = _data(n=4)
    u, _ = cs.covariance_library((1., 3.))
    observed = cs.continuous_component_logpdf(x, v, shape, np.inf, tau, covariances=u)
    for i in range(4):
        for c in [0, 1, 5, 28, 80, 81, 90, 102]:
            expected = multivariate_normal.logpdf(x[i], cov=tau*v[i]*shape+u[c])
            assert observed[i, c] == pytest.approx(expected, abs=1e-7, rel=1e-10)


@pytest.mark.parametrize('df', [.1, 1.5, 5., 100., 1e8])
@pytest.mark.parametrize('tau', [.003, 1., 700.])
def test_null_analytic_student_including_extreme_rows(df, tau):
    x = np.array([[0., 0., 0., 0.], [1., -2., 3., -4.], [1e120, 1e100, -1e99, 0.]])
    v = np.array([.003, 2., 1.])
    shape = .55*np.eye(4)+.45
    observed = cs.continuous_component_logpdf(x, v, shape, df, tau, covariances=np.zeros((1, 4, 4)))[:, 0]
    for i in range(3):
        expected = multivariate_t.logpdf(x[i], shape=tau*v[i]*shape, df=df)
        assert observed[i] == pytest.approx(expected, abs=1e-6, rel=2e-13)


@pytest.mark.parametrize('df', [1.5, 5., 25.])
@pytest.mark.parametrize('tau', [.08, 1., 14.])
def test_null_quadrature_against_analytic_and_increasing_order(df, tau):
    x = np.array([[.1, -.2, 0., .2], [4., -3., 1., 2.], [50., 20., -30., 2.]])
    v = np.array([.5, 1., 2.])
    analytic = cs.continuous_component_logpdf(x, v, np.eye(4), df, tau, covariances=np.zeros((1, 4, 4)))
    errors = []
    for order in [16, 64, 128]:
        problem = cs._Convolution(x, v, np.eye(4), df, np.zeros((1, 4, 4)), order, 32, 4)
        raw = problem.densities(np.log(tau), analytic_null=False)[0]
        errors.append(np.max(np.abs(raw-analytic)))
    assert errors[-1] < .02  # Numerical check across challenging scale shifts.
    assert errors[-1] < errors[0]+1e-10


def _independent_loglambda_integral(x, v, shape, df, tau, u, center):
    inverse = np.linalg.inv(np.linalg.cholesky(shape))
    values, vectors = np.linalg.eigh(inverse @ u @ inverse.T)
    values[np.abs(values) < 1e-12] = 0
    y = vectors.T @ inverse @ x
    constant = -2*np.log(2*np.pi)-.5*np.linalg.slogdet(shape)[1]
    a = df/2

    def integrand(log_lambda):
        lam = np.exp(log_lambda)
        diagonal = tau*v/lam+values
        log_normal = constant-.5*np.sum(np.log(diagonal)+y*y/diagonal)
        log_gamma_with_jacobian = a*np.log(a)-gammaln(a)+a*log_lambda-a*lam
        return np.exp(log_normal+log_gamma_with_jacobian-center)

    pieces = [quad(integrand, low, high, epsabs=2e-11, epsrel=2e-11, limit=200)[0]
              for low, high in [(-50., -15.), (-15., -4.), (-4., 0.), (0., 4.), (4., 9.)]]
    return np.log(sum(pieces))+center


@pytest.mark.parametrize('df', [5., 9., 25.])
@pytest.mark.parametrize('tau', [.2, 1., 6.])
def test_true_nonnull_convolution_matches_independent_adaptive_integral(df, tau):
    x = np.array([[1., -.3, 2., 0.], [12., -4., 1., 7.]])
    v = np.array([.7, 1.3])
    shape = .6*np.eye(4)+.4
    density = cs.continuous_component_logpdf(x, v, shape, df, tau, covariances=SMALL_LIBRARY, quadrature_order=512)
    for i in range(len(x)):
        for c in [1, 2]:
            reference = _independent_loglambda_integral(x[i], v[i], shape, df, tau, SMALL_LIBRARY[c], density[i, c])
            assert density[i, c] == pytest.approx(reference, abs=3e-4)


def test_very_low_df_unresolved_quadrature_is_detected_by_independent_audit():
    # Retained counterexample: the density helper is an approximation, not an
    # accuracy certificate. This low-df, large-observation case remains hard.
    x = np.array([[1., -.3, 2., 0.], [12., -4., 1., 7.]])
    shape = .6*np.eye(4)+.4
    kernel = cs._Convolution(x, np.array([.7, 1.3]), np.linalg.cholesky(shape), 1.5,
                             SMALL_LIBRARY, 128, 64, 16)
    audit = cs._reference_check(kernel, 1., np.array([.05, .45, .5]), 2e-4)
    assert not audit['passed']
    assert audit['max_mixture_log_difference'] > .005
    assert audit['failures'] == []
    assert all(row['relative_density_tail_bound'] <= 1e-10 and row['relative_score_tail_bound'] <= 1e-10
               for row in audit['rows'])


def test_convolution_is_not_student_with_added_covariance():
    x = np.array([[0., 0., 0., 0.]])
    density = cs.continuous_component_logpdf(x, [1.], np.eye(4), 1.5, covariances=SMALL_LIBRARY, quadrature_order=128)
    incorrect = multivariate_t.logpdf(x[0], shape=10*np.eye(4), df=1.5)
    assert abs(density[0, 1]-incorrect) > .2


@pytest.mark.parametrize('df', [1.5, 25., np.inf])
@pytest.mark.parametrize('tau', [.03, 1.4, 30.])
def test_component_derivative_is_same_objective_finite_difference(df, tau):
    x, v, shape, _ = _data(n=7)
    args = (x, v, shape, df)
    step = 2e-5
    _, score = cs.continuous_component_logpdf(*args, tau, covariances=SMALL_LIBRARY, return_score=True)
    above = cs.continuous_component_logpdf(*args, tau*np.exp(step), covariances=SMALL_LIBRARY)
    below = cs.continuous_component_logpdf(*args, tau*np.exp(-step), covariances=SMALL_LIBRARY)
    np.testing.assert_allclose(score, (above-below)/(2*step), atol=2e-7, rtol=3e-8)


@pytest.mark.parametrize('df', [5., np.inf])
def test_fit_stationarity_objective_json_and_input_immutability(df):
    data = _data(df=df)
    before = [value.copy() for value in data[:3]]
    fit = cs.fit_continuous_scale(*data, covariances=SMALL_LIBRARY)
    assert fit['converged'], fit['diagnostics']
    assert fit['diagnostics']['numerical_failure_count'] == 0
    assert len(fit['diagnostics']['attempts'][-1]['starts']) == 3
    attempt = fit['diagnostics']['attempts'][-1]
    selected = attempt['starts'][attempt['selected_start']]
    assert selected['weight_kkt_residual'] <= 1e-6
    assert selected['projected_log_tau_residual'] <= 1e-6
    order = attempt['quadrature_order'] or 32
    logdensity, score = cs.continuous_component_logpdf(*data, fit['variance_multiplier'], covariances=SMALL_LIBRARY,
                                                    quadrature_order=order, return_score=True)
    w = np.asarray(fit['weights'])
    mixtures = logsumexp(logdensity+np.log(w), axis=1)
    expected = mixtures.sum()+np.dot(np.r_[1., np.full(len(w)-1, .02/(len(w)-1))], np.log(w))
    assert fit['objective'] == pytest.approx(expected, abs=2e-10)
    responsibilities = np.exp(logdensity+np.log(w)-mixtures[:, None])
    assert np.sum(responsibilities*score)/len(data[0]) == pytest.approx(selected['log_tau_gradient_per_row'], abs=2e-11)
    json.dumps(fit, allow_nan=False)
    for original, after in zip(before, data[:3]):
        np.testing.assert_array_equal(original, after)


def test_default_dictionary_fit_and_row_permutation():
    data = _data(n=32, df=np.inf)
    fit = cs.fit_continuous_scale(*data, initial_taus=(1.,))
    rng = np.random.default_rng(811)
    permutation = rng.permutation(len(data[0]))
    other = cs.fit_continuous_scale(data[0][permutation], data[1][permutation], *data[2:], initial_taus=(1.,))
    assert fit['converged'] and other['converged']
    assert fit['objective'] == other['objective']
    assert fit['variance_multiplier'] == other['variance_multiplier']
    assert fit['weights'] == other['weights']


@pytest.mark.parametrize('df', [1.5, np.inf])
def test_batching_preserves_densities(df):
    x, v, shape, _ = _data(n=13)
    a = cs.continuous_component_logpdf(x, v, shape, df, amplitudes=(1.,), batch_size=3, component_batch=5)
    b = cs.continuous_component_logpdf(x, v, shape, df, amplitudes=(1.,), batch_size=64, component_batch=64)
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-13)


@pytest.mark.parametrize('df', [5., np.inf])
def test_small_monte_carlo_continuous_effect_scale_recovery(df):
    estimates = []
    # Fixed seeds and broad recovery tolerance are unit-level model sanity,
    # not validation of downstream FDR, power or competition with other fits.
    for seed in [7183, 9137, 11551]:
        fit = cs.fit_continuous_scale(*_data(seed=seed, n=240, df=df, tau=2.5),
                                     covariances=SMALL_LIBRARY, initial_taus=(.5, 3.),
                                     quadrature_order=32, max_quadrature_order=512)
        assert fit['converged'], fit['diagnostics']
        estimates.append(fit['variance_multiplier'])
    assert abs(np.mean(estimates)-2.5) < .5
    assert all(1.2 < value < 4 for value in estimates)


@pytest.mark.parametrize('df', [5., np.inf])
def test_null_large_scale_recovery_analytic_only(df):
    rng = np.random.default_rng(7723)
    x = rng.normal(size=(256, 4))*np.sqrt(20.)
    if np.isfinite(df):
        x /= np.sqrt(rng.chisquare(df, len(x))/df)[:, None]
    fit = cs.fit_continuous_scale(x, np.ones(len(x)), np.eye(4), df, covariances=np.zeros((1, 4, 4)))
    assert fit['converged']
    assert 16 < fit['variance_multiplier'] < 25
    assert fit['diagnostics']['attempts'][0]['quadrature_check']['passed']


def test_failed_starts_retained_without_fake_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError('injected density failure')
    monkeypatch.setattr(cs._Convolution, 'densities', fail)
    fit = cs.fit_continuous_scale(*_data(n=12), covariances=SMALL_LIBRARY)
    assert fit['status'] == 'all_starts_failed'
    assert fit['variance_multiplier'] is None and not fit['converged']
    assert fit['diagnostics']['numerical_failure_count'] == 3
    assert len(fit['diagnostics']['attempts'][0]['starts']) == 3
    json.dumps(fit, allow_nan=False)


def test_weight_failure_is_not_success(monkeypatch):
    def fail(*args, **kwargs):
        raise FloatingPointError('injected weight failure')
    monkeypatch.setattr(cs, '_fit_weights', fail)
    fit = cs.fit_continuous_scale(*_data(n=12), covariances=SMALL_LIBRARY, initial_taus=(1.,))
    assert not fit['converged']
    assert fit['diagnostics']['numerical_failure_count'] == 1


def test_quadrature_failure_refines_and_preserves_all_attempts(monkeypatch):
    calls = []

    def fail_check(kernel, refined, *args):
        calls.append((kernel.order, refined.order))
        return {'passed': False, 'failure': 'injected nonagreement'}
    monkeypatch.setattr(cs, '_quadrature_check', fail_check)
    fit = cs.fit_continuous_scale(*_data(n=12), covariances=SMALL_LIBRARY, initial_taus=(1.,),
                                 quadrature_order=8, max_quadrature_order=32)
    assert not fit['converged']
    assert fit['variance_multiplier'] is not None
    assert calls == [(8, 16), (16, 32)]
    assert len(fit['diagnostics']['attempts']) == 2


def test_iteration_limit_and_boundary_certificate():
    fit = cs.fit_continuous_scale(*_data(n=12), covariances=SMALL_LIBRARY, initial_taus=(.01,), max_iter=0)
    assert not fit['converged']
    boundary = cs.fit_continuous_scale(np.zeros((12, 4)), np.ones(12), np.eye(4), np.inf,
                                     covariances=np.zeros((1, 4, 4)), initial_taus=(1.,), tau_bounds=(.02, 10.))
    assert boundary['converged']
    assert boundary['variance_multiplier'] == pytest.approx(.02)
    selected = boundary['diagnostics']['attempts'][0]['starts'][0]
    assert selected['active_bound'] == 'lower'
    assert selected['log_tau_gradient_per_row'] < 0


def test_reference_rejection_cannot_be_overruled_by_laguerre_agreement(monkeypatch):
    monkeypatch.setattr(cs, '_quadrature_check', lambda *args: {'passed': True})
    monkeypatch.setattr(cs, '_reference_check', lambda *args: {'passed': False, 'failures': [{'message': 'injected adaptive error'}]})
    fit = cs.fit_continuous_scale(*_data(n=12), covariances=SMALL_LIBRARY, initial_taus=(1.,),
                                 quadrature_order=8, max_quadrature_order=32)
    assert not fit['converged']
    assert fit['diagnostics']['quadrature_rejection_count'] == 0
    assert fit['diagnostics']['reference_rejection_count'] == 2
    assert fit['diagnostics']['reference_integration_failure_count'] == 2


def test_custom_penalty_objective_and_base_scale_equivariance():
    data = _data(n=32, df=np.inf)
    options = {'covariances': SMALL_LIBRARY, 'null_pseudocount': 2., 'alternative_pseudocount': .3,
               'initial_taus': (1.,)}
    fit = cs.fit_continuous_scale(*data, **options)
    assert fit['converged']
    assert fit['diagnostics']['weight_log_penalty'] == [2., .15, .15]
    density = cs.continuous_component_logpdf(*data, fit['variance_multiplier'], covariances=SMALL_LIBRARY)
    weights = np.asarray(fit['weights'])
    expected = logsumexp(density+np.log(weights), axis=1).sum()+np.dot([2., .15, .15], np.log(weights))
    assert fit['objective'] == pytest.approx(expected, abs=1e-10)
    shifted = cs.fit_continuous_scale(data[0], 3*data[1], *data[2:],
                                     **(options | {'initial_taus': (1/3,), 'tau_bounds': (1e-3/3, 1e3/3)}))
    assert shifted['converged']
    assert shifted['objective'] == pytest.approx(fit['objective'], abs=1e-9)
    assert 3*shifted['variance_multiplier'] == pytest.approx(fit['variance_multiplier'], rel=1e-5)


def test_exact_scale_nonidentifiability_counterexample():
    # N(0, 2I): all-null tau=2 and all-effect U=I,tau=1 are observationally
    # identical. A prior preference is not data-based resolution of this fact.
    x, _, _, _ = _data(n=5)
    u = np.array([np.zeros((4, 4)), np.eye(4)])
    a = cs.continuous_component_logpdf(x, np.ones(5), np.eye(4), np.inf, 2., covariances=u)[:, 0]
    b = cs.continuous_component_logpdf(x, np.ones(5), np.eye(4), np.inf, 1., covariances=u)[:, 1]
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_large_finite_df_is_computed_and_matches_gaussian_limit():
    data = _data(n=4, df=np.inf)
    large, score = cs.continuous_component_logpdf(*data[:3], 1e8, covariances=SMALL_LIBRARY, return_score=True)
    gaussian, gaussian_score = cs.continuous_component_logpdf(*data, covariances=SMALL_LIBRARY, return_score=True)
    # Finite Student differs genuinely by O(q^2/df); these observations have
    # q close to 100. Analytic null equality to Student is tested separately.
    np.testing.assert_allclose(large, gaussian, atol=3e-5, rtol=2e-7)
    np.testing.assert_allclose(score, gaussian_score, atol=6e-5, rtol=2e-7)
    kernel = cs._Convolution(data[0], data[1], np.linalg.cholesky(data[2]), 1e8, SMALL_LIBRARY, 32, 64, 16)
    check = cs._reference_check(kernel, 1., np.array([.5, .25, .25]), 2e-4)
    assert check['passed'], check


@pytest.mark.parametrize('df', [1.5, 5., 25.])
def test_nonnull_extreme_student_tail_not_gaussianized_or_trimmed(df):
    x = np.array([[1e120, -2e120, 3e119, 0.]])
    result, score = cs.continuous_component_logpdf(x, [1.], np.eye(4), df,
                                                 covariances=SMALL_LIBRARY, return_score=True)
    assert np.isfinite(result).all() and np.isfinite(score).all()
    # Fixed bounded Gaussian effects vanish relative to an enormous Student
    # tail observation. This is a limiting numeric diagnostic, not truncation.
    np.testing.assert_allclose(result, np.broadcast_to(result[:, :1], result.shape), atol=2e-4, rtol=0.)


@pytest.mark.parametrize('df', [np.inf, 25.])
def test_retained_isotropic_null_failure_not_a_numerical_failure(df):
    # Main-supplied falsification fixture. This regression records the actual
    # failure rather than labelling numerical stationarity as scale recovery.
    x = np.random.default_rng(80317).normal(0., 3., (128, 4))
    fit = cs.fit_continuous_scale(x, np.ones(128), np.eye(4), df)
    assert fit['converged'], fit['diagnostics']
    assert fit['statistical_validation'] == 'not_validated'
    assert fit['variance_multiplier'] < .2  # Known negative, not a desired target.
    selected = fit['diagnostics']['attempts'][-1]
    assert all(start['optimizer_converged'] for start in selected['starts'])
    assert max(start['tau'] for start in selected['starts']) > 6.


def test_extreme_proposal_transport_derivative_near_smooth_transition():
    x = np.array([[3000., -2000., 1200., 900.]])
    step = 1e-5
    args = (x, [1.], np.eye(4), 9.)
    _, score = cs.continuous_component_logpdf(*args, covariances=SMALL_LIBRARY, return_score=True)
    above = cs.continuous_component_logpdf(*args, np.exp(step), covariances=SMALL_LIBRARY)
    below = cs.continuous_component_logpdf(*args, np.exp(-step), covariances=SMALL_LIBRARY)
    np.testing.assert_allclose(score, (above-below)/(2*step), atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize('kwargs', [
    {'max_iter': -1}, {'max_iter': True}, {'quadrature_order': 3}, {'batch_size': 0},
    {'component_batch': .5}, {'initial_taus': ()}, {'tau_bounds': (1, 1)},
    {'tau_bounds': (-1, 1)}, {'tol': 0}, {'quadrature_tol': np.nan},
    {'quadrature_order': 64, 'max_quadrature_order': 64}, {'amplitudes': (1, 1)},
    {'amplitudes': ()}, {'weight_max_iter': 0}, {'diagonal': 'yes'},
])
def test_invalid_controls(kwargs):
    with pytest.raises(ValueError):
        cs.fit_continuous_scale(*_data(n=4), **kwargs)


@pytest.mark.parametrize('change', [
    (0, np.zeros((4, 3))), (0, np.zeros((0, 4))), (0, np.full((4, 4), np.nan)),
    (1, np.zeros(4)), (1, np.ones(3)), (2, np.ones((4, 4))),
    (2, np.diag([1., 1., 1., -1.])), (3, 0.), (3, .01), (3, 1e10), (3, np.nan),
])
def test_invalid_data(change):
    data = list(_data(n=4))
    data[change[0]] = change[1]
    with pytest.raises(ValueError):
        cs.fit_continuous_scale(*data)


def test_negative_effect_covariance_and_nonnull_first_rejected():
    for covariances in [np.ones((1, 4, 4)), np.array([np.zeros((4, 4)), -np.eye(4)])]:
        with pytest.raises(ValueError):
            cs.fit_continuous_scale(*_data(n=4), covariances=covariances)


def test_api_does_not_accept_test_labels_or_test_means():
    parameters = inspect.signature(cs.fit_continuous_scale).parameters
    assert not any('test' in name or 'label' in name for name in parameters)
