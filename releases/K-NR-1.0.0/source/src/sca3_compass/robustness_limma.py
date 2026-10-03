# SPDX-License-Identifier: GPL-2.0-only
"""Classic limma variance moderation for a complete vector with common df.

Python adaptation (2026-09-15) of limma 3.66.0, RELEASE_3_22:
fitFDist.R, fitFDistRobustly.R and squeezeVar.R, by Gordon Smyth and
Belinda Phipson. The log-minus-digamma approximation is from Gordon Smyth's
statmod; the root solver is adapted from R Core's GPL R_zeroin2.
Sources, attribution, checksums, GPL text and executable R parity evidence
are in external/limma-reference. This combined adaptation uses GPL v2,
WITHOUT ANY WARRANTY; see that directory's GPL-2.

Scope: >3 positive finite residual variances, scalar common residual df,
no covariate/trend, classic (legacy=True) hyperparameter estimation. This
is an established empirical-Bayes comparator, not a novel Student method.
Fit on the full variance vector, without labels or effect-size filtering.
For residual energy q with known d=20, pass q/d, NOT q. A downstream
moderated t uses variance_factor * var_post and df + df_prior. limma's
eBayes additionally caps that t df at the pooled residual df; squeezeVar
itself does not calculate t statistics or p-values. Fitted residual shapes
remain a plug-in assumption; this module does not establish FDR control.

R is used only by the independent parity script, never by this module.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import betainc, betaln, hyp2f1, polygamma, roots_legendre
from scipy.stats import chi2, f, rankdata

__all__ = ["squeeze_var"]

LIMMA_RELEASE = "RELEASE_3_22"
LIMMA_VERSION = "3.66.0"


def _logmdigamma(x: float) -> float:
    """statmod::logmdigamma, retaining its large-x cancellation avoidance."""
    if x < 5:
        return (math.log(x / (x + 5)) + _logmdigamma(x + 5)
                + 1/x + 1/(x+1) + 1/(x+2) + 1/(x+3) + 1/(x+4))
    inverse_square = (1 / x) ** 2
    coefficients = (-1/12, 1/120, -1/252, 1/240, -1/132,
                    691/32760, -1/12, 3617/8160)
    tail = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        tail = coefficient + inverse_square * tail
    return 1 / (2 * x) - inverse_square * tail


def _trigamma_inverse(x: float) -> float:
    """limma::trigammaInverse's special cases and monotone Newton steps."""
    if x > 1e7:
        return 1 / math.sqrt(x)
    if x < 1e-6:
        return 1 / x if x else math.inf
    y = 0.5 + 1 / x
    for _ in range(51):
        tri = float(polygamma(1, y))
        change = tri * (1 - tri / x) / float(polygamma(2, y))
        y += change
        if -change / y < 1e-8:
            return y
    raise ArithmeticError("limma trigamma inversion did not converge")


def _fit_f_dist(x: np.ndarray, df: float) -> tuple[float, float, dict]:
    floor = 1e-5 * float(np.median(x))
    working = np.maximum(x, floor)
    e = np.log(working) + _logmdigamma(df / 2)
    emean = float(np.mean(e))
    excess = float(np.sum((e - emean) ** 2) / (x.size - 1) - polygamma(1, df / 2))
    if excess > 0:
        prior_df = 2 * _trigamma_inverse(excess)
        scale = math.exp(emean - _logmdigamma(prior_df / 2))
    else:
        prior_df = math.inf
        # The official infinite-df branch uses the arithmetic pooled mean.
        scale = float(np.mean(working))
    return scale, prior_df, {"log_variance_excess": excess,
                             "standard_floored_count": int(np.sum(x < floor))}


def _r_zeroin(fun, a: float, b: float, fa: float, fb: float) -> float:
    """R's R_zeroin2 algorithm, with limma's uniroot tolerance of 1e-8.

    Adapted 2026-09-15 from R Core (1999-2016), Peter Dalgaard and NETLIB.
    Keeping R's stopping rule matters when df2/(1+df2) is close to one.
    """
    if fa == 0:
        return a
    if fb == 0:
        return b
    c, fc = a, fa
    for _ in range(1001):
        previous_step = b - a
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb
        tolerance = 2 * np.finfo(float).eps * abs(b) + 1e-8 / 2
        new_step = (c - b) / 2
        if abs(new_step) <= tolerance or fb == 0:
            return b
        if abs(previous_step) >= tolerance and abs(fa) > abs(fb):
            cb = c - b
            if a == c:
                t1 = fb / fa
                p, q = cb * t1, 1 - t1
            else:
                q, t1, t2 = fa / fc, fb / fc, fb / fa
                p = t2 * (cb * q * (q - t1) - (b - a) * (t1 - 1))
                q = (q - 1) * (t1 - 1) * (t2 - 1)
            if p > 0:
                q = -q
            else:
                p = -p
            if p < 0.75 * cb * q - abs(tolerance * q) / 2 and p < abs(previous_step * q / 2):
                new_step = p / q
        if abs(new_step) < tolerance:
            new_step = tolerance if new_step > 0 else -tolerance
        a, fa = b, fb
        b += new_step
        fb = float(fun(b))
        if not math.isfinite(fb):
            raise ArithmeticError("Non-finite robust limma root objective")
        if (fb > 0 and fc > 0) or (fb < 0 and fc < 0):
            c, fc = a, fa
    raise ArithmeticError("Robust limma root solver did not converge")


@lru_cache(maxsize=1)
def _uniform_quadrature() -> tuple[np.ndarray, np.ndarray]:
    # Same 128-node Gaussian rule as statmod::gauss.quad.prob(dist='uniform').
    nodes, weights = roots_legendre(128)
    nodes, weights = (nodes + 1) / 2, weights / 2
    nodes.flags.writeable = weights.flags.writeable = False
    return nodes, weights


@lru_cache(maxsize=64)
def _infinite_moments(df: float, tails: tuple[float, float]) -> tuple[float, float]:
    return _winsorized_moments(df, math.inf, tails)


def _winsorized_moments(df: float, prior_df: float, tails: tuple[float, float]) -> tuple[float, float]:
    probabilities = np.array([tails[0], 1 - tails[1]])
    fq = chi2.ppf(probabilities, df) / df if math.isinf(prior_df) else f.ppf(probabilities, df, prior_df)
    zq = np.log(fq)
    q = fq / (1 + fq)
    rule_nodes, weights = _uniform_quadrature()
    width = q[1] - q[0]
    nodes = q[0] + width * rule_nodes
    fnodes = nodes / (1 - nodes)
    znodes = np.log(fnodes)
    density = chi2.pdf(fnodes * df, df) * df if math.isinf(prior_df) else f.pdf(fnodes, df, prior_df)
    density = density / (1 - nodes) ** 2
    mean = float(width * np.sum(weights * density * znodes) + np.sum(zq * tails))
    variance = float(width * np.sum(weights * density * (znodes - mean) ** 2)
                     + np.sum((zq - mean) ** 2 * tails))
    return mean, variance


def _f_logsf(statistic: np.ndarray | float, df: float, prior_df: float) -> np.ndarray:
    """F upper tail in log space, including tails below float underflow.

    I_z(a,b) = z**a/a/B(a,b) * 2F1(a,1-b;a+1;z). The identity is used
    only where betainc underflows, preserving R's log.p=TRUE tail behavior.
    """
    statistic = np.asarray(statistic, dtype=float)
    log_z = -np.logaddexp(0, math.log(df) - math.log(prior_df) + np.log(statistic))
    z = np.exp(log_z)
    a, b = prior_df / 2, df / 2
    probability = betainc(a, b, z)
    with np.errstate(divide="ignore"):
        result = np.log(probability)
    underflow = (probability == 0) & np.isfinite(log_z)
    if np.any(underflow):
        result = np.array(result, copy=True)
        result[underflow] = (a * log_z[underflow] - math.log(a) - betaln(a, b)
                             + np.log(hyp2f1(a, 1 - b, a + 1, z[underflow])))
    return result


def _fit_f_dist_robustly(x: np.ndarray, df: float, tails: tuple[float, float]):
    n = x.size
    floor = float(np.median(x)) * 1e-12
    x = np.maximum(x, floor)
    regular_scale, regular_df, info = _fit_f_dist(x, df)
    info.update(nonrobust_prior_df=regular_df, nonrobust_prior_variance=regular_scale)
    if max(tails) < 1 / n:
        return regular_scale, np.full(n, regular_df), regular_df, {**info, "branch": "no_winsorization"}

    z = np.log(x)
    trimmed = int(n * tails[1])
    ztrend = float(np.mean(np.sort(z)[trimmed:n-trimmed]))
    zresid = z - ztrend
    zrq = np.quantile(zresid, [tails[0], 1 - tails[1]], method="linear")
    zwins = np.clip(zresid, zrq[0], zrq[1])
    zwmean = float(np.mean(zwins))
    zwvar = float(np.mean((zwins - zwmean) ** 2) * n / (n - 1))
    moment_mean, moment_var = _infinite_moments(df, tails)
    info["winsorized_log_variance"] = zwvar
    if zwvar <= moment_var:
        corrected = ztrend + zwmean - moment_mean
        scale = math.exp(corrected)
        statistic = np.exp(z - corrected)
        tail_p = chi2.sf(statistic * df, df)
        empirical = (n - rankdata(statistic, method="average") + 0.5) / n
        prob_not_outlier = np.minimum(tail_p / empirical, 1)
        shrunk = np.full(n, math.inf)
        outlier = prob_not_outlier < 1
        if np.any(outlier):
            shrunk[outlier] = prob_not_outlier[outlier] * (n * df)
            order = np.argsort(tail_p, kind="stable")
            shrunk[order] = np.maximum.accumulate(shrunk[order])
        return scale, shrunk, math.inf, {**info, "branch": "infinite_prior"}

    if math.isinf(regular_df):
        return regular_scale, np.full(n, regular_df), regular_df, {**info, "branch": "nonrobust_infinite_fallback"}

    def objective(linked):
        if linked == 1:
            variance = moment_var
        else:
            _, variance = _winsorized_moments(df, linked / (1 - linked), tails)
        return math.log(zwvar / variance)

    low = regular_df / (1 + regular_df)
    f_low = objective(low)
    if f_low >= 0:
        prior_df = regular_df
        branch = "nonrobust_lower_bound"
    else:
        root = _r_zeroin(objective, low, 1.0, f_low, math.log(zwvar / moment_var))
        prior_df = root / (1 - root)
        branch = "finite_prior"
    fitted_mean, _ = _winsorized_moments(df, prior_df, tails)
    corrected = ztrend + zwmean - fitted_mean
    scale = math.exp(corrected)
    statistic = np.exp(z - corrected)
    log_tail_p = _f_logsf(statistic, df, prior_df)
    empirical = np.log(n - rankdata(statistic, method="average") + 0.5) - math.log(n)
    log_prob_not_outlier = np.minimum(log_tail_p - empirical, 0)
    prob_not_outlier = np.exp(log_prob_not_outlier)
    prob_outlier = -np.expm1(log_prob_not_outlier)
    if np.any(log_prob_not_outlier < 0):
        minimum = float(np.min(log_tail_p))
        if minimum == -math.inf:
            outlier_df = 0.0
            shrunk = prob_not_outlier * prior_df
        else:
            outlier_df = math.log(0.5) / minimum * prior_df
            new_log_tail = float(_f_logsf(np.max(statistic), df, outlier_df))
            outlier_df *= math.log(0.5) / new_log_tail
            shrunk = prob_not_outlier * prior_df + prob_outlier * outlier_df
        order = np.argsort(log_tail_p, kind="stable")
        ordered = shrunk[order]
        cumulative_mean = np.cumsum(ordered) / np.arange(1, n + 1)
        minimum_index = int(np.argmin(cumulative_mean))
        ordered[:minimum_index+1] = cumulative_mean[minimum_index]
        shrunk[order] = np.maximum.accumulate(ordered)
    else:
        outlier_df = prior_df
        shrunk = np.full(n, prior_df)
    return scale, shrunk, prior_df, {**info, "branch": branch, "outlier_prior_df": outlier_df}


def squeeze_var(variance, df: float = 20.0, *, robust: bool = False,
                winsor_tail_p=(0.05, 0.10)) -> tuple[np.ndarray, float | np.ndarray, dict]:
    """Return (posterior_variance, prior_df, diagnostics), without calling R.

    ``variance`` must be a complete one-dimensional vector of >3 positive
    finite residual variances (q/d); ``df`` must be one finite scalar >1e-6.
    ``prior_df`` is scalar for standard fitting and per-observation for robust
    fitting, and can include infinity. Diagnostics contain ``prior_variance``
    and the global (before outlier adjustment) ``global_prior_df``.

    Default robust Winsor tails are limma's (0.05, 0.10). Both tails must lie
    strictly between 0 and 0.5, or both be zero to disable Winsorization.
    Small-sample, missing/zero variance, unequal-df and covariate branches of
    the general R API are deliberately unsupported, rather than imputed.
    """
    x = np.asarray(variance, dtype=float)
    if x.ndim != 1 or x.size <= 3 or not np.isfinite(x).all() or np.any(x <= 0):
        raise ValueError("Expected a full vector of >3 positive finite residual variances q/d")
    df_array = np.asarray(df, dtype=float)
    if df_array.ndim != 0 or not np.isfinite(df_array) or df_array <= 1e-6:
        raise ValueError("Expected common finite scalar residual df >1e-6")
    df = float(df_array)
    if not isinstance(robust, (bool, np.bool_)):
        raise TypeError("robust must be boolean")
    if robust:
        tail_array = np.asarray(winsor_tail_p, dtype=float)
        if tail_array.shape == ():
            tail_array = np.repeat(tail_array, 2)
        if (tail_array.shape != (2,) or not np.isfinite(tail_array).all()
                or not (np.all((tail_array > 0) & (tail_array < 0.5)) or np.all(tail_array == 0))):
            raise ValueError("Use two Winsor tail probabilities in (0, 0.5), or (0, 0)")
        tails = tuple(float(value) for value in tail_array)
        scale, prior_df, global_df, info = _fit_f_dist_robustly(x, df, tails)
        info["winsor_tail_p"] = list(tails)
        info["robust_floored_count"] = int(np.sum(x < np.median(x) * 1e-12))
    else:
        scale, prior_df, info = _fit_f_dist(x, df)
        global_df = prior_df
        info["branch"] = "infinite_prior" if math.isinf(global_df) else "finite_prior"

    dfs = np.broadcast_to(prior_df, x.shape)
    if np.isfinite(dfs).all():
        posterior = (df * x + dfs * scale) / (df + dfs)
    else:
        posterior = np.full_like(x, scale)
        if np.min(dfs) <= 1e100:
            finite = np.isfinite(dfs)
            posterior[finite] = (df * x[finite] + dfs[finite] * scale) / (df + dfs[finite])
    if not np.isfinite(posterior).all() or np.any(posterior <= 0) or np.isnan(dfs).any():
        raise FloatingPointError("Non-finite or non-positive limma moderation result")
    diagnostics = {
        "method": "limma_squeezeVar_classic_robust" if robust else "limma_squeezeVar_classic",
        "implementation": "python_port", "limma_release": LIMMA_RELEASE,
        "limma_version": LIMMA_VERSION, "legacy": True, "covariate": None,
        "robust": bool(robust), "n_variances": int(x.size), "residual_df": df,
        "pooled_residual_df": int(x.size) * df,
        "prior_variance": scale, "global_prior_df": global_df,
        "finite_prior_df_count": int(np.isfinite(dfs).sum()),
        "outlier_adjusted_count": int(np.sum(dfs < global_df)),
        **info,
    }
    return posterior, prior_df, diagnostics
