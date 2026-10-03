"""Exact finite-sample checks and numerical regressions; no confirmation data."""
import json
import math
import random
from collections import Counter
from decimal import Decimal, localcontext
from fractions import Fraction
from functools import cache
from itertools import pairwise, product

import pytest

from sca3_compass.robustness_betting import (
    BETTING_GRID,
    _log_mixture,
    betting_interval,
)

EXPECTED_GRID = (.0001, .001, .005, .01, .02, .05, .1, .2, .4, .7, .9, .99)
EXACT_GRID = tuple(Fraction(str(f)) for f in EXPECTED_GRID)


def _counts(data):
    return tuple(sorted(Counter(data).items()))


def _direct_capital(data, m):
    return math.fsum(math.prod((1 - f) + f * x / m for x in data) for f in EXPECTED_GRID) / 12


def _exact_capital(data, m):
    return sum((math.prod(1 - f + f * x / m for x in data) for f in EXACT_GRID), Fraction()) / 12


def _decimal_log_capital(counts, m):
    """Independent 90-digit oracle using direct Decimal factors."""
    with localcontext() as context:
        context.prec = 90
        candidate = Decimal.from_float(m)
        capitals = []
        for fraction in EXPECTED_GRID:
            f = Decimal.from_float(fraction)
            capitals.append(sum(
                count * (1 - f + f * Decimal.from_float(x) / candidate).ln()
                for x, count in counts
            ))
        maximum = max(capitals)
        return float(maximum + sum((v - maximum).exp() for v in capitals).ln() - Decimal(12).ln())


@cache
def _bernoulli_intervals(n, alpha):
    return tuple(betting_interval([0.0] * (n - k) + [1.0] * k, alpha) for k in range(n + 1))


def test_api_metadata_moments_and_frozen_grid():
    data = [-.2, .4, .1, -.1] * 40
    saved = data.copy()
    result = betting_interval(data, .02, (-1, 1))
    mean = math.fsum(data) / len(data)
    variance = math.fsum((x - mean) ** 2 for x in data) / (len(data) - 1)
    assert set(result) == {
        "method", "n", "mean", "sample_variance", "alpha", "lower", "upper",
        "support", "status", "grid",
    }
    assert result["method"] == "two_sided_fixed_bet_mixture"
    assert result["n"] == len(data)
    assert result["mean"] == pytest.approx(mean)
    assert result["sample_variance"] == pytest.approx(variance)
    assert result["alpha"] == .02
    assert result["support"] == [-1., 1.]
    assert result["status"] == "computed"
    assert tuple(result["grid"]) == BETTING_GRID == EXPECTED_GRID
    assert data == saved
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    # Neither a returned grid nor a support list is live configuration.
    result["grid"][0] = .5
    result["support"][0] = -100
    assert betting_interval(data, .02, (-1, 1))["grid"] == list(EXPECTED_GRID)
    assert betting_interval(iter(data), .02, (-1, 1)) == betting_interval(data, .02, (-1, 1))


@pytest.mark.parametrize("n", [1, 2, 4, 8, 12, 20])
@pytest.mark.parametrize("alpha", [.01, .05, .2, .5, .9])
def test_exact_bernoulli_enumeration_coverage_and_each_tail(n, alpha):
    # Enumerate all outcomes by their sufficient count, with exact binomial
    # masses (not Monte Carlo). Each count represents comb(n,k) binary paths.
    intervals = _bernoulli_intervals(n, alpha)
    for probability in ["0", ".001", ".01", ".05", ".1", ".25", ".5",
                        ".75", ".9", ".95", ".99", ".999", "1"]:
        p = Fraction(probability)
        lower_error = upper_error = Fraction()
        for k, interval in enumerate(intervals):
            mass = math.comb(n, k) * p ** k * (1 - p) ** (n - k)
            # Fraction(float) checks the actual returned endpoints exactly.
            if p < Fraction(interval["lower"]):
                lower_error += mass
            if p > Fraction(interval["upper"]):
                upper_error += mass
        budget = Fraction(str(alpha))
        assert lower_error <= budget / 2, (n, alpha, p, lower_error)
        assert upper_error <= budget / 2, (n, alpha, p, upper_error)
        assert lower_error + upper_error <= budget


@pytest.mark.parametrize("alpha", [.05, .2, .5])
def test_exact_binary_path_enumeration_simultaneous_prefix_coverage(alpha):
    n = 6
    for p in map(Fraction, [".05", ".2", ".5", ".8", ".95"]):
        error = Fraction()
        for path in product([0, 1], repeat=n):
            k = sum(path)
            mass = p ** k * (1 - p) ** (n - k)
            for t in range(1, n + 1):
                interval = _bernoulli_intervals(t, alpha)[sum(path[:t])]
                if not Fraction(interval["lower"]) <= p <= Fraction(interval["upper"]):
                    error += mass
                    break
        assert error <= Fraction(str(alpha))


@pytest.mark.parametrize("n", [1, 3, 5])
@pytest.mark.parametrize("candidate", [".1", ".5", ".9"])
def test_exact_null_expectation_and_implemented_capital(n, candidate):
    m = Fraction(candidate)
    for p in [Fraction(), m / 2, m]:
        expectation = Fraction()
        implemented = []
        for path in product([0, 1], repeat=n):
            mass = p ** sum(path) * (1 - p) ** (n - sum(path))
            exact = _exact_capital(path, m)
            capital = math.exp(_log_mixture(_counts(path), float(m)))
            assert capital == pytest.approx(float(exact), rel=2e-13, abs=1e-14)
            expectation += mass * exact
            implemented.append(float(mass) * capital)
        assert expectation <= 1
        assert math.fsum(implemented) == pytest.approx(float(expectation), abs=2e-13)
        if p == m:
            assert expectation == 1


def test_conditional_null_expectation_with_bounded_nonbernoulli_observations():
    # Each possible past is fixed before the next draw; distributions may vary
    # with that past if their conditional mean stays <= m.
    m = Fraction(1, 2)
    for prefix in product([Fraction(0), Fraction(1, 3), Fraction(1)], repeat=3):
        current = math.exp(_log_mixture(_counts(prefix), float(m)))
        distributions = [
            [(Fraction(0), Fraction(1, 4)), (Fraction(1, 2), Fraction(1, 2)),
             (Fraction(1), Fraction(1, 4))],
            [(Fraction(0), Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2))],
        ]
        for distribution in distributions:
            expectation = math.fsum(
                float(weight) * math.exp(_log_mixture(_counts((*prefix, float(x))), float(m)))
                for x, weight in distribution
            )
            assert expectation <= current + 1e-12
            exact = sum(weight * _exact_capital((*prefix, x), m) for x, weight in distribution)
            assert exact <= _exact_capital(prefix, m)


@pytest.mark.parametrize("data", [[.2], [0., 1.], [.1, .3, .5, .9], [0.] * 12, [1.] * 12])
def test_log_capital_matches_direct_mixture_and_is_decreasing(data):
    candidates = [.00001, .01, .1, .4, .7, .99999, 1.]
    logs = [_log_mixture(_counts(data), m) for m in candidates]
    for m, value in zip(candidates, logs):
        assert math.exp(value) == pytest.approx(_direct_capital(data, m), rel=3e-13)
    assert all(a >= b for a, b in pairwise(logs))


def test_inversion_matches_both_direct_tail_thresholds_and_is_asymmetric():
    data = [.01, .01, .1, .2, .8] * 10
    result = betting_interval(data, alpha=.1)
    assert _direct_capital(data, result["lower"]) == pytest.approx(20., rel=2e-12)
    assert _direct_capital([1 - x for x in data], 1 - result["upper"]) == pytest.approx(20., rel=2e-12)
    assert _direct_capital(data, result["lower"] * .99) > 20
    assert _direct_capital(data, result["lower"] * 1.01) < 20
    assert not math.isclose(result["mean"] - result["lower"], result["upper"] - result["mean"])


@pytest.mark.parametrize("value", [0., .2, .5, 1.])
@pytest.mark.parametrize("n", [1, 2, 100, 10000])
def test_constant_data_keeps_positive_uncertainty(value, n):
    result = betting_interval([value] * n)
    assert 0 <= result["lower"] <= value <= result["upper"] <= 1
    assert result["upper"] > result["lower"]
    assert result["sample_variance"] == (None if n == 1 else 0.)
    assert result["mean"] == value
    if value == 0:
        assert result["lower"] == 0
    if value == 1:
        assert result["upper"] == 1


@pytest.mark.parametrize("alpha", [.05, .8, 1e-300])
@pytest.mark.parametrize("value", [0., .2, 1., 1e-250])
def test_singleton_matches_analytic_root(alpha, value):
    # For n=1, e(m)=1-fbar+fbar*x/m. Decimal avoids overflow in 2/alpha.
    with localcontext() as context:
        context.prec = 90
        average = sum(Decimal.from_float(f) for f in EXPECTED_GRID) / 12
        a, x = Decimal.from_float(alpha), Decimal.from_float(value)
        denominator = 2 - a * (1 - average)
        expected_lower = float(a * average * x / denominator)
        expected_upper = float(1 - a * average * (1 - x) / denominator)
    result = betting_interval([value], alpha)
    assert result["lower"] == pytest.approx(expected_lower, rel=2e-12, abs=math.ulp(0.0) * 2)
    assert result["upper"] == pytest.approx(expected_upper, rel=2e-15, abs=1e-15)


@pytest.mark.parametrize("scale,offset", [(2., -1.), (17., 8.), (.001, -3.), (-1., 0.), (-7., 4.)])
def test_affine_and_sign_equivariance(scale, offset):
    data = [0., .1, .2, .3, .7, 1.] * 30
    result = betting_interval(data, .03)
    transformed = betting_interval([offset + scale * x for x in data], .03,
                                   tuple(sorted((offset, offset + scale))))
    expected = sorted((offset + scale * result["lower"], offset + scale * result["upper"]))
    assert transformed["lower"] == pytest.approx(expected[0], abs=2e-14)
    assert transformed["upper"] == pytest.approx(expected[1], abs=2e-14)
    assert transformed["mean"] == pytest.approx(offset + scale * result["mean"], abs=2e-14)
    assert transformed["sample_variance"] == pytest.approx(scale ** 2 * result["sample_variance"])


def test_permutation_invariance_and_generator_single_pass():
    rng = random.Random(7139)
    data = [rng.random() for _ in range(80)] + [0., 1., .5] * 4
    expected = betting_interval(data)
    rng.shuffle(data)
    assert betting_interval(data) == expected
    assert betting_interval(reversed(data)) == expected
    consumed = []

    def observations():
        for value in data:
            consumed.append(value)
            yield value

    assert betting_interval(observations()) == expected
    assert consumed == data


@pytest.mark.parametrize("data", [[0.] * 10, [1.] * 10, [.2] * 12, [0., .1, .8, 1.] * 30])
def test_smaller_alpha_never_narrows_interval(data):
    results = [betting_interval(data, a) for a in [math.ulp(0.0), 1e-300, .0001, .01, .05, .2, .8,
                                                 math.nextafter(1., 0.)]]
    for wider, narrower in pairwise(results):
        assert wider["lower"] <= narrower["lower"]
        assert wider["upper"] >= narrower["upper"]


@pytest.mark.parametrize("data,error", [
    ([], ValueError), ([float("nan")], ValueError), ([float("inf")], ValueError),
    ([float("-inf")], ValueError), ([math.nextafter(0., -1.)], ValueError),
    ([math.nextafter(1., 2.)], ValueError), ([None], TypeError), ([True], TypeError),
    ([False], TypeError), ([".5"], TypeError), ([complex(.5)], TypeError),
    ([[.5]], TypeError), (None, TypeError), (.5, TypeError), ("01", TypeError),
    ({0: "a", 1: "b"}, TypeError), ({0., 1.}, TypeError), ([10 ** 1000], ValueError),
])
def test_invalid_observations_are_rejected_without_dropping_or_clipping(data, error):
    with pytest.raises(error):
        betting_interval(data)


@pytest.mark.parametrize("alpha,error", [
    (0, ValueError), (1, ValueError), (-.1, ValueError), (1.01, ValueError),
    (float("nan"), ValueError), (float("inf"), ValueError), (True, TypeError),
    (".05", TypeError), (None, TypeError), (complex(.05), TypeError),
])
def test_invalid_alpha(alpha, error):
    with pytest.raises(error):
        betting_interval([.5], alpha)


@pytest.mark.parametrize("support,error", [
    ((1, 0), ValueError), ((0, 0), ValueError), ((0, float("inf")), ValueError),
    ((float("nan"), 1), ValueError), ((-1e308, 1e308), ValueError),
    ((False, 1), TypeError), (("0", 1), TypeError), ((None, 1), TypeError),
    ((0,), ValueError), ((0, 1, 2), ValueError), (None, TypeError),
    ("01", TypeError), ({0, 1}, TypeError), ({0: 0, 1: 1}, TypeError),
])
def test_invalid_support(support, error):
    with pytest.raises(error):
        betting_interval([.5], support=support)


@pytest.mark.parametrize("kwargs", [{"grid": [.1]}, {"weights": [1]}, {"fractions": [.2]}])
def test_bets_cannot_be_supplied_or_tuned_through_api(kwargs):
    with pytest.raises(TypeError):
        betting_interval([.5], **kwargs)


@pytest.mark.parametrize("counts,m", [
    (((0., 1000000), (1., 1000000)), 1e-300),
    (((0., 10000000),), .9),
    (((1., 1000000000000),), 1 - 1e-12),
    (((.5, 1000000000000),), math.nextafter(.5, 0.)),
    (((math.ulp(0.0), 1), (1., 1)), math.ulp(0.0)),
])
def test_log_capital_extremes_against_high_precision_oracle(counts, m):
    actual = _log_mixture(counts, m)
    assert math.isfinite(actual)
    assert actual == pytest.approx(_decimal_log_capital(counts, m), rel=3e-13, abs=2e-10)


def test_large_sample_avoids_overflow_and_underflow_of_products():
    result = betting_interval([0.] * 100000 + [1.] * 100000, alpha=1e-300)
    assert result["n"] == 200000
    assert .4 < result["lower"] < .5 < result["upper"] < .6
    assert result["lower"] == pytest.approx(1 - result["upper"], abs=2e-15)
    assert result["sample_variance"] == pytest.approx(200000 / 199999 * .25)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("support,data", [
    ((0., 1e308), [5e307, 5e307, 5e307]),
    ((0., 1e308), [0., 1e308]),
    ((-1e308, 0.), [-1e308, 0.]),
    ((-1e-200, 1e-200), [-1e-200, 0., 1e-200]),
    ((0., math.ulp(0.0)), [0., math.ulp(0.0)]),
    ((1., math.nextafter(1., 2.)), [1., math.nextafter(1., 2.)]),
])
def test_extreme_finite_supports_keep_finite_bounds_and_serializable_metadata(support, data):
    result = betting_interval(data, support=support)
    assert support[0] <= result["lower"] <= result["mean"] <= result["upper"] <= support[1]
    assert all(math.isfinite(result[key]) for key in ["mean", "lower", "upper"])
    if max(data) - min(data) == 1e308:
        assert result["sample_variance"] is None
        assert result["status"] == "computed_variance_overflow"
    else:
        assert result["status"] == "computed"
    json.dumps(result, allow_nan=False)


def test_small_normalized_root_and_endpoint_reflection():
    value = 1e-300
    small = betting_interval([value] * 10)
    assert 0 < small["lower"] < value
    # Direct reflection retains the tiny distance from U=0 even though 1-X
    # would round to zero after normalizing values near the upper endpoint.
    reflected = betting_interval([-value] * 10, support=(-1., 0.))
    assert reflected["upper"] < 0
    assert reflected["upper"] == pytest.approx(-small["lower"], rel=2e-13, abs=0.)


def test_subnormal_alpha_and_observation_do_not_raise_or_loop():
    tiny = math.ulp(0.0)
    for data in [[tiny], [0., tiny], [1.]]:
        result = betting_interval(data, alpha=tiny)
        assert 0 <= result["lower"] <= result["mean"] <= result["upper"] <= 1
        json.dumps(result, allow_nan=False)
