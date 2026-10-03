"""Standalone analytical sanity checks; NOT a new production PC implementation.

Only known-nuisance distribution identities and illustrative counterexamples.
Run directly with python -B to print the small, seeded diagnostic to stdout.
No project data, registry, benchmark modules, network, or output files are used.
"""
import json

import numpy as np
from scipy.stats import beta, multivariate_t, norm, t
from threadpoolctl import threadpool_limits


def equicorrelation(dimension, rho):
    return (1 - rho) * np.eye(dimension) + rho * np.ones((dimension, dimension))


def conditional_t_parameters(mu, shape, df, index, observed):
    """Law before the extra argmax truncation, in the location/shape convention."""
    mu = np.asarray(mu, float)
    shape = np.asarray(shape, float)
    retained = np.delete(np.arange(len(mu)), index)
    cross = shape[retained, index]
    residual = observed - mu[index]
    conditional_mean = mu[retained] + cross * residual / shape[index, index]
    schur = shape[np.ix_(retained, retained)] - np.outer(cross, cross) / shape[index, index]
    conditional_shape = schur * (df + residual**2 / shape[index, index]) / (df + 1)
    return conditional_mean, conditional_shape, df + 1


def student_log_density_lipschitz(df, dimension=4):
    """Global constant in the Mahalanobis location metric, finite Student df."""
    if not np.isfinite(df) or df <= 0:
        raise ValueError("This bound requires positive FINITE Student df")
    return (df + dimension) / (2 * np.sqrt(df))


def directional_bin_probabilities(mu, alpha=0.05):
    """Positive-axis equal-normal-mass bins; 2D independent Gaussian ONLY."""
    count = int(round(0.5 / alpha))
    if not np.isclose(count * alpha, 0.5):
        raise ValueError("This sanity construction requires .5/alpha integer")
    edges = norm.ppf(np.linspace(0.5, 1, count + 1))
    return np.diff(norm.cdf(edges - mu))


def independence_conditioning_counterexample(n=160000, seed=771903):
    """NOT published cPCH: exact iid global-null order-statistic formula misused.

    For iid continuous zero-location observations, conditional on their maximum
    c, P(second maximum >= s | maximum=c) = 1 - (F(s)/F(c))**3.
    Applying this factorization to a correlated/common-radial model is invalid.
    All component means here are zero; even the nuisance means are KNOWN.
    """
    rng = np.random.default_rng(seed)
    rows = []
    with threadpool_limits(limits=1):
        for df in (np.inf, 25.0, 5.0):
            for rho in (0.0, 0.65):
                shape = equicorrelation(4, rho)
                x = rng.normal(size=(n, 4)) @ np.linalg.cholesky(shape).T
                if np.isfinite(df):
                    x /= np.sqrt(rng.chisquare(df, size=(n, 1)) / df)
                top = np.partition(x, 2, axis=1)[:, 2:]
                cdf = norm.cdf(top) if np.isinf(df) else t.cdf(top, df)
                p = -np.expm1(3 * (np.log(cdf[:, 0]) - np.log(cdf[:, 1])))
                for alpha in (0.05, 0.001):
                    k = int(np.sum(p <= alpha))
                    low = float(beta.ppf(0.005, k, n - k + 1)) if k else 0.0
                    high = float(beta.ppf(0.995, k + 1, n - k)) if k < n else 1.0
                    rows.append({"df": "Gaussian" if np.isinf(df) else df,
                                 "rho": rho, "alpha": alpha, "n": n,
                                 "rejections": k, "rate": k / n,
                                 "pointwise_99pct_CP": [low, high]})
    return rows


def test_conditional_student_density_factorization():
    rng = np.random.default_rng(82071)
    for df in (1.5, 5.0, 25.0):
        for rho in (0.0, 0.65):
            shape = equicorrelation(4, rho)
            mu = np.array([2.0, -0.4, 0.0, -2.0])
            for j in range(4):
                x = rng.normal(size=4)
                cm, cs, cdf = conditional_t_parameters(mu, shape, df, j, x[j])
                keep = np.delete(np.arange(4), j)
                joint = multivariate_t.logpdf(x, loc=mu, shape=shape, df=df)
                marginal = t.logpdf((x[j] - mu[j]) / np.sqrt(shape[j, j]), df) - 0.5 * np.log(shape[j, j])
                conditional = multivariate_t.logpdf(x[keep], loc=cm, shape=cs, df=cdf)
                np.testing.assert_allclose(joint - marginal, conditional, rtol=0, atol=2e-12)


def test_conditioning_does_not_remove_the_free_mean():
    shape = equicorrelation(4, 0.65)
    first = conditional_t_parameters([0, 0, 0, 0], shape, 25, 0, 2)
    second = conditional_t_parameters([2, 0, 0, 0], shape, 25, 0, 2)
    third = conditional_t_parameters([8, 0, 0, 0], shape, 25, 0, 2)
    np.testing.assert_allclose(first[0], 1.3)
    np.testing.assert_allclose(second[0], 0)
    np.testing.assert_allclose(third[0], -3.9)
    assert not np.allclose(first[1], second[1])


def test_zero_correlation_student_still_has_conditional_scale_dependence():
    at_zero = conditional_t_parameters([0] * 4, np.eye(4), 5, 0, 0)
    at_four = conditional_t_parameters([0] * 4, np.eye(4), 5, 0, 4)
    np.testing.assert_allclose(at_zero[0], at_four[0])
    np.testing.assert_allclose(at_four[1], 4.2 * at_zero[1])


def test_whitening_does_not_preserve_the_directional_pc_null():
    shape = equicorrelation(4, 0.65)
    original = np.array([-3.0, 0, 0, 0])
    transformed = np.linalg.solve(np.linalg.cholesky(shape), original)
    assert np.count_nonzero(original > 0) == 0
    assert np.count_nonzero(transformed > 0) == 3


def test_student_location_density_ratio_bound():
    rng = np.random.default_rng(711820)
    for df in (1.5, 5, 25, 100):
        shape = equicorrelation(4, 0.65)
        inv = np.linalg.inv(shape)
        for _ in range(100):
            x = rng.normal(size=4) * rng.choice([0.01, 1.0, 100.0])
            mu0 = rng.normal(size=4) * 3
            delta = rng.normal(size=4) * 0.2
            change = abs(multivariate_t.logpdf(x, mu0 + delta, shape, df)
                         - multivariate_t.logpdf(x, mu0, shape, df))
            bound = student_log_density_lipschitz(df) * np.sqrt(delta @ inv @ delta)
            assert change <= bound + 1e-12


def test_student_location_score_sign_against_finite_differences():
    shape = equicorrelation(4, 0.65)
    inv = np.linalg.inv(shape)
    x = np.array([2.0, -0.1, 1.4, 0.2])
    mu = np.array([0.4, -0.5, 0.0, -1.0])
    residual = x - mu
    for df in (1.5, 5.0, 25.0):
        analytic = (df + 4) * (inv @ residual) / (df + residual @ inv @ residual)
        numerical = []
        for j in range(4):
            delta = np.eye(4)[j] * 1e-5
            numerical.append((multivariate_t.logpdf(x, mu + delta, shape, df)
                              - multivariate_t.logpdf(x, mu - delta, shape, df)) / 2e-5)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-8, atol=2e-9)


def test_directional_positive_bin_null_and_power_sanity():
    alpha = 0.05
    for mu_null in (-100, -6, -2, -0.3, 0):
        masses = directional_bin_probabilities(mu_null)
        assert np.all(masses <= alpha + 2e-15)
        for other in (-6, 0, 1, 2, 6, 100):
            probability = masses @ directional_bin_probabilities(other)
            assert probability <= alpha + 2e-15
    for effect in (1, 2, 3):
        masses = directional_bin_probabilities(effect)
        max_p_power = norm.sf(norm.isf(alpha) - effect) ** 2
        assert masses @ masses > max_p_power


def test_unbounded_null_complement_has_a_tail_certificate():
    for df in (1.5, 5, 25):
        tau, baseline, b, sigma = 0.001, 0.00099, 4.0, 1.7
        outer = b + sigma * t.isf(tau - baseline, df)
        assert outer > b
        np.testing.assert_allclose(t.sf((outer - b) / sigma, df), tau - baseline,
                                   rtol=1e-8, atol=1e-13)


def test_small_seeded_wrong_independence_counterexample():
    rows = independence_conditioning_counterexample(n=40000)
    matched = next(row for row in rows if row['df'] == 'Gaussian' and row['rho'] == 0 and row['alpha'] == 0.05)
    wrong = next(row for row in rows if row['df'] == 25 and row['rho'] == 0.65 and row['alpha'] == 0.05)
    assert 0.045 < matched['rate'] < 0.055
    assert wrong['pointwise_99pct_CP'][0] > 0.08


if __name__ == '__main__':
    print(json.dumps({"status": "ANALYTICAL_SANITY_NOT_PROJECT_CONFIRMATION",
                      "seed": 771903,
                      "wrong_independence_diagnostic": independence_conditioning_counterexample(),
                      "independent_2D_directional_bins": [
                          {"means": [effect, effect],
                           "bin_power": float(directional_bin_probabilities(effect) @ directional_bin_probabilities(effect)),
                           "max_p_power": float(norm.sf(norm.isf(0.05) - effect) ** 2)}
                          for effect in (1, 2, 3)]}, indent=2, allow_nan=False))
