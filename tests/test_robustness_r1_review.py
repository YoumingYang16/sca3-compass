"""Isolated deterministic checks for the independent R1 mathematical review.

No random draws, production method, source mutation, or confirmation experiment.
Floating-point identities here are checks of the analytic arguments, not certified
special-function intervals or approval of main's production implementation.
"""

from itertools import combinations
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import helmert
from scipy.special import betaln
from scipy.stats import beta, chi2, ncx2, norm, t


def _projection_ratio(z, a):
    d = z.shape[-1] - 1
    h = helmert(d + 1, full=False).T
    y = (z - z.mean(axis=-1, keepdims=True)) @ h
    return float(a @ z.mean(axis=-1) / np.sqrt(np.sum((a @ y) ** 2) / d))


def _endpoint_values(z, u, d):
    """Diagnostic values only: no assertion of directed floating-point bounds."""
    if z == 0:
        return np.full_like(u, 0.5)
    q = t.isf((1 - u) / 2, d)
    result = t.sf(z * q / t.ppf(0.75, d), d)
    result[0] = 0.5
    result[-1] = 0.0 if z > 0 else 1.0
    return result


def _cost_constant(n, k):
    return 0.5 * math.exp(
        2 * betaln((k + 1) / 2, (n + 2 - k) / 2)
        - betaln(k, n + 1 - k)
    )


@pytest.mark.parametrize("rho", [-0.1, 0.0, 0.8, 0.95])
@pytest.mark.parametrize("row_scales", [[1.0, 1.0, 1.0], [0.2, 3.0, 1.7]])
def test_projected_gaussian_covariance_has_student_factorization(rho, row_scales):
    k = 6
    h = helmert(k, full=False).T
    root = np.array([[1.0, 0, 0], [-0.3, 1.2, 0], [0.4, 0.1, 0.8]])
    rs = np.diag(row_scales) @ root @ root.T @ np.diag(row_scales)
    rp = (1 - rho) * np.eye(k) + rho * np.ones((k, k))
    a = np.array([0.0, 0.4, 1.7])
    mean_map = np.kron(a, np.ones(k) / k)
    contrast_map = np.kron(a[:, None], h)
    covariance = np.kron(rs, rp)
    variance_a = a @ rs @ a
    np.testing.assert_allclose(
        mean_map @ covariance @ mean_map,
        (1 + (k - 1) * rho) / k * variance_a,
        rtol=1e-13,
    )
    np.testing.assert_allclose(mean_map @ covariance @ contrast_map, 0, atol=2e-14)
    np.testing.assert_allclose(
        contrast_map.T @ covariance @ contrast_map,
        (1 - rho) * variance_a * np.eye(k - 1),
        atol=2e-14,
    )


def test_pointwise_radius_and_direction_scale_cancellation():
    z = np.array([[0.2, 1.7, -0.3, 0.9], [1.1, -0.7, 0.4, 0.1], [-1, 2, 0.5, 0.6]])
    a = np.array([0.2, 0.7, 1.3])
    original = _projection_ratio(z, a)
    for radius in [1e-12, 0.1, 3.0, 1e12]:
        for direction_scale in [0.01, 1.0, 17.0]:
            assert _projection_ratio(radius * z, direction_scale * a) == pytest.approx(original)


@pytest.mark.parametrize("mu", [[0, 0, 0, 0], [100, 0, -2, 0], [0, -100, 2, -3]])
def test_composite_singleton_null_has_a_dominated_triple(mu):
    mu = np.asarray(mu)
    noise = np.array([[0.2, -0.1, 1, 0.5], [1, 0.3, -0.7, 0.4],
                      [-0.4, 0.7, 0.2, 1.1], [0.6, -0.3, 0.4, 0.9]])
    a = np.array([0.2, 0.7, 1.3])
    covered = 0
    triples = list(combinations(range(4), 3))
    assert len(triples) == 4
    for triple in triples:
        index = list(triple)
        if np.all(mu[index] <= 0):
            covered += 1
            assert _projection_ratio(noise[index] + mu[index, None], a) <= (
                _projection_ratio(noise[index], a) + 1e-12
            )
    assert covered >= 1


@pytest.mark.parametrize("d", [1, 2, 5, 11])
def test_exact_anchor_integrand_is_linear_for_every_student_df(d):
    u = np.linspace(0, 1, 1001)
    values = _endpoint_values(t.ppf(0.75, d), u, d)
    np.testing.assert_allclose(values, (1 - u) / 2, atol=2e-11, rtol=0)


@pytest.mark.parametrize("n,k", [(1, 1), (16, 8), (32, 16), (512, 256)])
def test_arbitrary_partition_gap_cannot_beat_analytic_cost_bound(n, k):
    intervals = 257
    exact_anchor = (n + 1 - k) / (2 * (n + 1))
    for exponent in [0.25, 1.0, 4.0]:
        u = np.linspace(0, 1, intervals + 1) ** exponent
        cdf = beta.cdf(u, k, n + 1 - k)
        mass = np.diff(cdf)
        f = (1 - u) / 2
        lower, upper = mass @ f[1:], mass @ f[:-1]
        assert lower <= exact_anchor <= upper
        gap = upper - lower
        assert gap + 2e-15 >= _cost_constant(n, k) / intervals
        assert gap == pytest.approx(0.5 * (np.diff(u) @ mass), abs=2e-15)
        if n == 1 and exponent == 1:
            assert gap == pytest.approx(0.5 / intervals, abs=2e-15)


@pytest.mark.parametrize("z", [0.03, 1.0, 10.0])
def test_equal_beta_probability_partition_has_half_over_J_gap(z):
    intervals = 100
    w = np.linspace(0, 1, intervals + 1)
    u = beta.ppf(w, 16, 17)
    f = _endpoint_values(z, u, 5)
    upper = np.sum(f[:-1]) / intervals
    lower = np.sum(f[1:]) / intervals
    assert upper - lower == pytest.approx(0.5 / intervals, abs=2e-15)


def test_reported_necessary_interval_counts():
    assert math.ceil(_cost_constant(32, 16) / 1e-7) == 2_099_663
    assert math.ceil(_cost_constant(512, 256) / 1e-7) == 552_004


def test_validation_selected_e_gate_has_expectation_one_point_two_five():
    # N=k=1, kappa=1.  B<1 iff U<1/2.  At p<=1/4,
    # the conditional rejection probability is sf(abs-t-quantile(U)).
    probability, error = quad(lambda u: t.sf(t.isf((1 - u) / 2, 5), 5), 0, 0.5)
    assert error < 1e-10  # A numerical cross-check only, not a proof certificate.
    assert probability == pytest.approx(3 / 16, abs=1e-11)
    expectation = 4 * probability + 0.5
    assert expectation == pytest.approx(1.25, abs=1e-10)
    assert expectation > 1


def test_duplicated_calibration_rows_are_anticonservative_for_upper_order():
    alleged_n, alleged_k = 2, 2
    alleged_level = (alleged_n + 1 - alleged_k) / (2 * (alleged_n + 1))
    actual_n, actual_k = 1, 1
    actual_probability = (actual_n + 1 - actual_k) / (2 * (actual_n + 1))
    assert alleged_level == pytest.approx(1 / 6)
    assert actual_probability == 1 / 4
    assert actual_probability > alleged_level


def test_negative_tail_needs_reversed_endpoints_and_zero_needs_no_quantile():
    u = np.linspace(0, 1, 101)
    positive = _endpoint_values(1.7, u, 5)
    negative = _endpoint_values(-1.7, u, 5)
    np.testing.assert_allclose(negative, 1 - positive, atol=1e-15)
    assert np.all(np.diff(positive) <= 0)
    assert np.all(np.diff(negative) >= 0)
    np.testing.assert_array_equal(_endpoint_values(0, u, 5), np.full_like(u, 0.5))


def test_ordinary_bonferroni_pc_max_triples_equals_three_times_second_p():
    # Deterministic p vectors, including ties, boundary values and mixed orders.
    p = np.array([[0.001, 0.08, 0.07, 0.6], [1, 1, 1, 1], [0, 0, 0.2, 0.9],
                  [0.2, 0.1, 0.1, 0.03], [0.6, 0.5, 0.4, 0.01]])
    values = [np.minimum(1, 3 * np.min(p[:, list(i)], axis=1))
              for i in combinations(range(4), 3)]
    np.testing.assert_array_equal(np.max(values, axis=0),
                                  np.minimum(1, 3 * np.sort(p, axis=1)[:, 1]))


@pytest.mark.parametrize("d", [1, 5, 11])
def test_noncentral_contrast_norm_is_stochastically_larger(d):
    squared_radii = np.geomspace(1e-6, 300, 501)
    central = chi2.cdf(squared_radii, d)
    for noncentrality in [0.1, 3, 50]:
        shifted = ncx2.cdf(squared_radii, d, noncentrality)
        assert np.all(shifted <= central + 2e-14)
        assert np.any(shifted < central - 1e-8)


@pytest.mark.parametrize("delta,noncentrality", [(0, 0), (-0.7, 0), (0, 3), (-0.7, 3)])
@pytest.mark.parametrize("positive_threshold", [0.5, 3.0])
def test_noncentral_denominator_and_nonpositive_numerator_give_tail_bound(
    delta, noncentrality, positive_threshold
):
    # Conditional on V and TRAIN, the normalized ratio is (N+delta)/R,
    # where d*R^2 is noncentral chi-square and N is independent standard normal.
    d = 5

    def integrand(radius):
        density = 2 * d * radius * ncx2.pdf(d * radius**2, d, noncentrality)
        return norm.sf(positive_threshold * radius - delta) * density

    probability, error = quad(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-11)
    bound = t.sf(positive_threshold, d)
    assert error < 1e-9  # Diagnostic quadrature only; the proof is analytic.
    assert probability <= bound + 1e-10
    if delta == 0 and noncentrality == 0:
        assert probability == pytest.approx(bound, abs=1e-10)
    else:
        assert probability < bound


def test_heterogeneous_negative_shift_is_not_pointwise_conservative():
    # The distributional extension must NOT be tested by requiring monotonicity
    # under a heterogeneous shift for each fixed Gaussian realization.
    noise = np.array([[1.0, 3.0]])
    nonpositive_shift = np.array([[0.0, -1.5]])
    a = np.ones(1)
    before = _projection_ratio(noise, a)
    after = _projection_ratio(noise + nonpositive_shift, a)
    assert before == pytest.approx(math.sqrt(2))
    assert after == pytest.approx(2.5 * math.sqrt(2))
    assert after > before


def test_untruncated_predictive_survival_can_fail_heterogeneous_null():
    # K=6, rho=.8, every study has pipeline means (-L,0,0,0,0,0).
    # As L increases, X/sqrt(kappa) -> -.2.  This algebraic limit, not
    # simulated data, gives a counterexample to applying full Q at negative X.
    pipelines, rho, n, order = 6, 0.8, 32, 16
    kappa = (1 + (pipelines - 1) * rho) / (pipelines * (1 - rho))
    limit_magnitude = 1 / math.sqrt(pipelines * kappa)
    alpha = 1 - (n + 1 - order) / (2 * (n + 1))
    absolute_t_cdf = 2 * t.cdf(limit_magnitude, pipelines - 1) - 1
    limiting_rejection_probability = beta.sf(absolute_t_cdf, order, n + 1 - order)
    assert alpha == pytest.approx(49 / 66)
    assert limit_magnitude == pytest.approx(0.2)
    assert limiting_rejection_probability == pytest.approx(0.9999962942266428)
    assert limiting_rejection_probability > alpha + 0.25


def test_current_production_tail_retains_nonpositive_statistic_safeguard():
    # No generated observations or learner execution: inspect the actual scalar
    # p-value function on a fixed set of nonpositive values.
    from sca3_compass.robustness_pivotal import predictive_tail

    np.testing.assert_array_equal(predictive_tail(np.array([-10.0, -0.2, 0.0]), 32, 5),
                                  np.ones(3))
