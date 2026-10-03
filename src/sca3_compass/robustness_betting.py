"""Classical fixed-bet-mixture intervals for means across experiment repetitions.

This is an analysis primitive, not a molecular testing algorithm. The fixed
grid and equal weights must be frozen before independent confirmation data.
See docs/robustness_betting.md for the proof, assumptions, and limitations.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from numbers import Real
from typing import Any

BETTING_GRID = (.0001, .001, .005, .01, .02, .05, .1, .2, .4, .7, .9, .99)
BETTING_REFERENCE = "https://arxiv.org/html/2010.09686v5"
_LOG_BETS = tuple((f, math.log(f), math.log1p(-f)) for f in BETTING_GRID)
_LOG_GRID_SIZE = math.log(len(BETTING_GRID))


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context}: expected a finite real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{context}: expected a finite real number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite real number")
    return number


def _logaddexp(a: float, b: float) -> float:
    """Stable log(exp(a) + exp(b)); callers supply finite a and b."""
    return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def _log_mixture(counts: tuple[tuple[float, int], ...], m: float) -> float:
    """Log mean capital for validated normalized observations and 0 < m <= 1.

    Counts only compress identical factors; they do not select bets or weights.
    log1p preserves tiny gains near m. logaddexp handles large X/m without ever
    constructing the ratio, product capital, or 2/alpha.
    """
    log_m = math.log(m)
    log_capitals = []
    for fraction, log_fraction, log_cash in _LOG_BETS:
        contributions = []
        for x, count in counts:
            if x == 0.0:
                log_factor = log_cash
            elif x <= 2.0 * m:
                log_factor = math.log1p(fraction * ((x - m) / m))
            else:
                log_factor = _logaddexp(log_cash, log_fraction + math.log(x) - log_m)
            contributions.append(count * log_factor)
        log_capitals.append(math.fsum(contributions))
    largest = max(log_capitals)
    return largest + math.log(math.fsum(math.exp(v - largest) for v in log_capitals)) - _LOG_GRID_SIZE


def _lower_bound(counts: tuple[tuple[float, int], ...], log_threshold: float) -> float:
    """Invert decreasing capital, retaining the outward side of the bracket."""
    right = max(x for x, _ in counts)
    if right == 0.0:
        return 0.0
    # e(0+) = infinity if any X > 0; e(max X) <= 1. No endpoint division.
    left = 0.0
    while True:
        middle = left + (right - left) / 2.0
        if middle == left or middle == right:
            # Adjacent floats; no absolute tolerance that erases small roots.
            return math.nextafter(left, 0.0)
        if _log_mixture(counts, middle) >= log_threshold:
            left = middle
        else:
            right = middle


def _moments(counts: tuple[tuple[float, int], ...], n: int) -> tuple[float, float | None]:
    # Weights before summation avoid overflow in sum(data) for large supports.
    mean = math.fsum(value * (count / n) for value, count in counts)
    mean = min(counts[-1][0], max(counts[0][0], mean))
    if n == 1:
        return mean, None
    deviation = max(abs(value - mean) for value, _ in counts)
    if deviation == 0.0:
        return mean, 0.0
    scaled_variance = math.fsum(
        (count / (n - 1)) * ((value - mean) / deviation) ** 2
        for value, count in counts
    )
    standard_deviation = deviation * math.sqrt(scaled_variance)
    return mean, standard_deviation * standard_deviation


def betting_interval(
    values: Iterable[float], alpha: float = .05, support: tuple[float, float] = (0, 1),
) -> dict[str, Any]:
    """Two-sided bounded-mean CI from an equal mixture of fixed fractions.

    For X = (value-L)/(U-L), the lower tail inverts
    mean_f prod_i ((1-f) + f*X_i/m) at 2/alpha; the upper tail reflects X.
    Independent repetitions with a common mean suffice. More generally, the
    same conditional mean given past observations suffices. Support, fractions,
    mixture weights, and the analysis protocol must be specified in advance.

    Empty, nonfinite, non-real, boolean, and out-of-support data are rejected.
    A singleton has sample_variance=None but still a valid betting interval.
    Variance overflow yields None and status='computed_variance_overflow';
    the bounds and mean remain available. Bounds are not a symmetric radius.
    """
    alpha = _number(alpha, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    if isinstance(support, (str, bytes, Mapping, AbstractSet)):
        raise TypeError("support must be an ordered pair of finite real numbers")
    lower, upper = support
    lower, upper = _number(lower, "support lower"), _number(upper, "support upper")
    width = upper - lower
    if lower >= upper or not math.isfinite(width):
        raise ValueError("A finite positive support width is required")
    if isinstance(values, (str, bytes, Mapping, AbstractSet)):
        raise TypeError("values must be an iterable of repetition observations")
    histogram: Counter[float] = Counter()
    for value in values:
        value = _number(value, "bounded observation")
        if value < lower or value > upper:
            raise ValueError(f"Observations must lie in [{lower}, {upper}]")
        histogram[value] += 1
    if not histogram:
        raise ValueError("At least one repetition is required")
    counts = tuple(sorted(histogram.items()))
    n = sum(histogram.values())
    mean, variance = _moments(counts, n)
    normalized = tuple(((value - lower) / width, count) for value, count in counts)
    # Compute the reflection from the original support to retain small U-value
    # differences that would disappear in floating-point subtraction 1-X.
    reflected = tuple(((upper - value) / width, count) for value, count in reversed(counts))
    log_threshold = math.log(2.0) - math.log(alpha)
    lo = lower + width * _lower_bound(normalized, log_threshold)
    hi = upper - width * _lower_bound(reflected, log_threshold)
    status = "computed"
    if variance is not None and not math.isfinite(variance):
        variance, status = None, "computed_variance_overflow"
    return {
        "method": "two_sided_fixed_bet_mixture", "n": n, "mean": mean,
        "sample_variance": variance, "alpha": alpha,
        "lower": max(lower, min(mean, math.nextafter(lo, -math.inf))),
        "upper": min(upper, max(mean, math.nextafter(hi, math.inf))),
        "support": [lower, upper], "status": status, "grid": list(BETTING_GRID),
    }
