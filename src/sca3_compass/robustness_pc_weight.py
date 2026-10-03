"""Standalone, TRAIN/fixed-model power allocations for signed r=2/S=4 PC.

No fitting, runner registration, or held means enter the public builder/lookup.
All integrals concern MODEL-uniform U, not an empirical gene average. See
docs/robustness_pc_weight.md for the common-location assumptions and limitations.
Uses the project's existing NumPy/SciPy dependencies only.
"""
import hashlib
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType

import numpy as np
from scipy.special import betainc, betaincc, expit
from scipy.stats import chi2, f, norm, t

from .robustness_cone import cone_weights3
from .robustness_projection import positive_direction
from .robustness_simes import SIMES_POLICY, simes_eligibility

WEIGHT_GRID = (.05, .1, .2, .4, .7, 1., 1.4, 2., 3., 4., 6., 10., 20.)
METHODS = ("projection", "support_simes", "simes", "bonferroni",
           "profile_bonferroni", "cone")
DEFAULT_SEED = 731905
SUBSETS = tuple(combinations(range(4), 3))


def _positive(value, name, *, infinity=False):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a positive scalar")
    value = float(value)
    if value <= 0 or np.isnan(value) or (not infinity and not np.isfinite(value)):
        raise ValueError(f"{name} must be positive and finite" + (" or +inf" if infinity else ""))
    return value


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _geometry(shape, profiles):
    shape = np.array(shape, dtype=float, copy=True)
    if shape.shape != (4, 4):
        raise ValueError("A finite symmetric positive definite 4x4 shape is required")
    simes_eligibility(shape)
    profiles = np.array(profiles, dtype=float, copy=True)
    if (profiles.shape != (2, 4) or not np.isfinite(profiles).all()
            or np.any(profiles < 0) or np.any(profiles.max(axis=1) <= 0)):
        raise ValueError("Two finite nonzero nonnegative profiles, shape 2x4, required")
    # Sign 0 = positive alternative; sign 1 = negative alternative MAGNITUDE.
    # Each amplitude is the largest study's location shift, in original units.
    profiles /= profiles.max(axis=1, keepdims=True)
    return shape, profiles


def _readonly(value):
    array = np.ascontiguousarray(value, dtype=float)
    # A bytes backing store prevents callers from re-enabling writes.
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


def _unit_interval(value, name):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must contain finite values in [0, 1]")
    return value


def _model_u(q, dimension, prior_df, prior_scale):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError("Q must be finite and nonnegative (infinity is not an endpoint)")
    # Stable F CDF via Q/(Q+nu*sigma2), avoiding overflowing Q/(d*sigma2).
    with np.errstate(divide="ignore"):
        log_ratio = np.log(q) - np.log(prior_df) - np.log(prior_scale)
    u = np.where(log_ratio <= 0,
                 betainc(dimension / 2, prior_df / 2, expit(log_ratio)),
                 betaincc(prior_df / 2, dimension / 2, expit(-log_ratio)))
    return _unit_interval(u, "MODEL U")


def _conditional_variance(q, dimension, prior_df, prior_scale, projection_variance):
    with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        log_variance = (np.log(projection_variance)
                        + np.logaddexp(np.log(prior_df) + np.log(prior_scale), np.log(q))
                        - np.log(prior_df + dimension))
        variance = np.exp(log_variance)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise FloatingPointError("Conditional scale is not representable")
    return variance


def _student_bank(samples, df, shape, seed):
    rng = np.random.default_rng(seed)  # Private; never touches np.random global state.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        bank = rng.standard_normal((samples, 4)) @ np.linalg.cholesky(shape).T
        if np.isfinite(df):
            bank /= np.sqrt(rng.chisquare(df, size=(samples, 1)) / df)
    if not np.isfinite(bank).all():
        raise FloatingPointError("Nonfinite independent noise bank")
    return bank


def _simes(p, eligible):
    k = p.shape[-1]
    if not eligible:
        return np.minimum(1., k * p.min(axis=-1))
    value = (np.sort(p, axis=-1) * k / np.arange(1, k + 1)).min(axis=-1)
    return np.where(value <= .5, value, 1.)


class _PCScorer:
    """Internal synthetic-bank kernel; all geometry prepared once per build."""

    def __init__(self, shape, profiles, df, methods=METHODS):
        self.shape, self.profiles = _geometry(shape, profiles)
        self.df = _positive(df, "conditional df", infinity=True)
        self.methods = tuple(methods)
        if not self.methods or len(set(self.methods)) != len(self.methods) or set(self.methods) - set(METHODS):
            raise ValueError("Choose distinct supported methods")
        self.diagonal = np.sqrt(np.diag(self.shape))
        self.directions = np.zeros((2, 4, 4))
        self.intersections = []
        for k, indices in enumerate(SUBSETS):
            subshape = self.shape[np.ix_(indices, indices)]
            entry = {"indices": list(indices), "signs": [],
                     "eligible": simes_eligibility(subshape)["simes_eligible"]}
            for sign in range(2):
                profile = self.profiles[sign, list(indices)].copy()
                if not np.any(profile > 0):
                    profile = np.ones(3)
                if "projection" in self.methods:
                    self.directions[sign, k, list(indices)] = positive_direction(profile, subshape)
                keep = np.flatnonzero(profile > 0)
                entry["signs"].append({"keep": keep, "allocation": profile / profile.sum(),
                    "eligible": simes_eligibility(subshape[np.ix_(keep, keep)])["simes_eligible"]})
            if "cone" in self.methods:
                entry["cone_weights"] = cone_weights3(subshape)
                entry["faces"] = []
                for size in range(1, 4):
                    for active in combinations(range(3), size):
                        inactive = [j for j in range(3) if j not in active]
                        entry["faces"].append((list(active), inactive,
                            np.linalg.inv(subshape[np.ix_(active, active)]),
                            subshape[np.ix_(active, inactive)]))
            self.intersections.append(entry)
        if not np.isfinite(self.directions).all():
            raise FloatingPointError("Nonfinite projection directions")

    def pvalues(self, standardized_means):
        x = np.asarray(standardized_means, dtype=float)
        if x.ndim != 3 or x.shape[1:] != (2, 4) or not np.isfinite(x).all():
            raise ValueError("Finite Nx2x4 standardized synthetic means required")
        sf = norm.sf if np.isinf(self.df) else lambda z: t.sf(z, self.df)
        output = {method: np.zeros(x.shape[:2]) for method in self.methods}
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            if "projection" in output:
                # Maximum intersection p equals tail of MINIMUM of four scores.
                statistic = np.einsum("nsc,skc->nsk", x, self.directions).min(axis=-1)
                output["projection"] = np.where(statistic > 0, sf(statistic), 1.)
            marginal_methods = set(output) - {"projection", "cone"}
            if marginal_methods:
                statistic = x / self.diagonal
                marginal = np.where(statistic > 0, sf(statistic), 1.)
            for entry in self.intersections:
                indices = entry["indices"]
                if marginal_methods:
                    p = marginal[..., indices]
                for method in marginal_methods:
                    if method == "simes":
                        value = _simes(p, entry["eligible"])
                    elif method == "bonferroni":
                        value = np.minimum(1., 3 * p.min(axis=-1))
                    else:
                        value = np.empty(x.shape[:2])
                        for sign, info in enumerate(entry["signs"]):
                            kept = p[:, sign, info["keep"]]
                            if method == "support_simes":
                                value[:, sign] = _simes(kept, info["eligible"])
                            else:
                                value[:, sign] = np.minimum(1., (kept / info["allocation"][info["keep"]]).min(axis=-1))
                    output[method] = np.maximum(output[method], value)
                if "cone" in output:
                    subset = x[..., indices]
                    distance = np.where(np.all(subset <= 0, axis=-1), 0., np.inf)
                    for active, inactive, inverse, cross in entry["faces"]:
                        xa = subset[..., active]
                        multipliers = xa @ inverse
                        valid = np.all(multipliers >= -1e-12, axis=-1)
                        if inactive:
                            fitted = subset[..., inactive] - multipliers @ cross
                            valid &= np.all(fitted <= 1e-12, axis=-1)
                        candidate = np.maximum((xa * multipliers).sum(axis=-1), 0.)
                        distance = np.minimum(distance, np.where(valid, candidate, np.inf))
                    if not np.isfinite(distance).all():
                        raise FloatingPointError("No finite feasible cone projection")
                    value = np.zeros_like(distance)
                    for j in range(1, 4):
                        tail = chi2.sf(distance, j) if np.isinf(self.df) else f.sf(distance / j, j, self.df)
                        value += entry["cone_weights"][j] * tail
                    output["cone"] = np.maximum(output["cone"], np.where(distance <= 0, 1., value))
        for value in output.values():
            _unit_interval(value, "Synthetic PC p-values")
        return output


def _allocate(prediction, grid):
    """Lagrange bracket and convexification; no true-PC optimality claim."""
    def allocation(lagrange):
        index = np.argmax(prediction - lagrange * grid, axis=1)
        return grid[index], prediction[np.arange(len(index)), index]

    # Negative multipliers allow an exact equality budget even for flat power.
    low, high = -1., 1.
    for _ in range(80):
        if allocation(low)[0].mean() >= 1 and allocation(high)[0].mean() <= 1:
            break
        low *= 2
        high *= 2
    else:
        raise FloatingPointError("Unable to bracket model weight budget")
    for _ in range(70):
        middle = (low + high) / 2
        if allocation(middle)[0].mean() > 1:
            low = middle
        else:
            high = middle
    over, over_power = allocation(low)
    under, under_power = allocation(high)
    gap = over.mean() - under.mean()
    fraction = 0. if gap == 0 else float((1. - under.mean()) / gap)
    if not 0 <= fraction <= 1 or np.any(over < under):
        raise FloatingPointError("Invalid interpolation bracket")
    height = fraction * over + (1. - fraction) * under
    # Correct only rounding in the model integral, never observed-gene means.
    for _ in range(3):
        residual = len(height) - height.sum()
        if residual == 0:
            break
        slack = grid[-1] - height if residual > 0 else height - grid[0]
        height[np.argmax(slack)] += residual
    if (np.any(height < grid[0] - 1e-12) or np.any(height > grid[-1] + 1e-12)
            or abs(height.mean() - 1.) > 1e-14):
        raise FloatingPointError("Non-unit model weight integral")
    return height, over, under, fraction, float((fraction * over_power + (1 - fraction) * under_power).mean())


@dataclass(frozen=True)
class PCWeightTables:
    """Built once using TRAIN/fixed inputs; held observations supply Q only.

    Tables have shape (2, bins). Lookup returns q.shape + (2,). Sign rows
    are independent budgets, not normalized across genes or signs.
    """
    dimension: int
    prior_df: float
    prior_scale: float
    projection_variance: float
    shape: np.ndarray
    profiles: np.ndarray
    tables: object
    overspend: object
    underspend: object
    fractions: object
    predictions: object
    prediction_se: object
    diagnostics: object

    def lookup_u(self, u, method="projection", *, allocation="height"):
        """MODEL-U lookup. Bins [j/B,(j+1)/B); u=1 uses the final bin.

        Sub-bin over-height applies when local coordinate < fraction;
        equality uses under-height. Endpoint u=1 has local coordinate 1.
        """
        u = _unit_interval(u, "U")
        if method not in self.tables or allocation not in ("height", "subbin"):
            raise ValueError("Unsupported method or allocation (height/subbin)")
        bins = self.tables[method].shape[1]
        coordinate = u * bins
        index = np.minimum(coordinate.astype(np.int64), bins - 1)
        if allocation == "height":
            return np.moveaxis(self.tables[method][:, index], 0, -1).copy()
        local = coordinate - index
        over = np.moveaxis(self.overspend[method][:, index], 0, -1)
        under = np.moveaxis(self.underspend[method][:, index], 0, -1)
        fraction = self.fractions[method]
        choose_over = (local[..., None] < fraction) | (fraction == 1.)
        return np.where(choose_over, over, under)

    def lookup(self, q, method="projection", *, allocation="height"):
        """Lookup with finite nonnegative held Q; no held fitting is possible."""
        q = np.asarray(q, dtype=float)
        if not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError("Q must be finite and nonnegative")
        # Exact separate Gaussian branch, including zero and huge finite Q.
        u = np.zeros_like(q) if np.isinf(self.prior_df) else _model_u(q, self.dimension, self.prior_df, self.prior_scale)
        return self.lookup_u(u, method, allocation=allocation)

    def model_integral(self, method="projection", *, allocation="height"):
        if method not in self.tables or allocation not in ("height", "subbin"):
            raise ValueError("Unsupported method or allocation (height/subbin)")
        if allocation == "height":
            return self.tables[method].mean(axis=1)
        fraction = self.fractions[method]
        return (fraction * self.overspend[method].mean(axis=1)
                + (1 - fraction) * self.underspend[method].mean(axis=1))


def build_pc_weight_tables(dimension, prior_df, prior_scale, projection_variance,
                           shape, profiles, *, reference_level=.0003, bins=64,
                           effects=(2.5, 3.5, 4.5), weight_grid=WEIGHT_GRID,
                           noise_samples=4096, seed=DEFAULT_SEED, batch_size=512,
                           methods=METHODS):
    """Allocate MODEL-uniform U using actual method-specific PC-power proxies.

    Every argument must be fixed a priori or TRAIN-derived. The private bank
    is independent of all study observations. API provenance is a caller
    obligation, not something this function can verify. No true parameters
    are assumed available to the caller; fitted nuisance inference is unproved.
    """
    dimension = _integer(dimension, "dimension", 1, 1000000)
    prior_df = _positive(prior_df, "prior_df", infinity=True)
    prior_scale = _positive(prior_scale, "prior_scale")
    projection_variance = _positive(projection_variance, "projection_variance")
    bins = _integer(bins, "bins", 2, 1024)
    noise_samples = _integer(noise_samples, "noise_samples", 2, 65536)
    batch_size = _integer(batch_size, "batch_size", 1, 4096)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    reference_level = _positive(reference_level, "reference_level")
    grid = np.array(weight_grid, dtype=float, copy=True)
    effects = np.array(effects, dtype=float, copy=True)
    if (grid.ndim != 1 or not 2 <= len(grid) <= 64 or not np.isfinite(grid).all()
            or np.any(grid <= 0) or np.any(np.diff(grid) <= 0)
            or not grid[0] < 1 < grid[-1] or not np.any(grid == 1)
            or reference_level >= 1. / grid[-1]):
        raise ValueError("Positive increasing grid containing 1 and straddling it; level*max(grid)<1 required")
    if (effects.ndim != 1 or not 1 <= len(effects) <= 32
            or not np.isfinite(effects).all() or np.any(effects <= 0)):
        raise ValueError("A finite nonempty positive effect grid is required")
    scorer = _PCScorer(shape, profiles, prior_df + dimension, methods)
    shape, profiles, methods = scorer.shape, scorer.profiles, scorer.methods
    tables, overspend, underspend, fractions = {}, {}, {}, {}
    predictions, errors, surrogate = {}, {}, {}
    bank_digest = None
    if np.isinf(prior_df):
        with np.errstate(over="ignore", under="ignore"):
            scale = projection_variance * prior_scale
        if not np.isfinite(scale) or scale <= 0:
            raise FloatingPointError("Gaussian scale is not representable")
        for method in methods:
            tables[method] = overspend[method] = underspend[method] = np.ones((2, bins))
            fractions[method] = np.zeros(2)
    else:
        u = (np.arange(bins) + .5) / bins
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            q = dimension * prior_scale * f.ppf(u, dimension, prior_df)
        if not np.isfinite(q).all() or np.any(q <= 0):
            raise FloatingPointError("MODEL F midpoint energies are not representable")
        variance = _conditional_variance(q, dimension, prior_df, prior_scale, projection_variance)
        bank = _student_bank(noise_samples, prior_df + dimension, shape, seed)
        bank_digest = hashlib.sha256(bank.astype("<f8").tobytes()).hexdigest()
        for method in methods:
            predictions[method] = np.empty((2, bins, len(grid)))
            errors[method] = np.empty_like(predictions[method])
        for b in range(bins):
            counts = {method: np.zeros((2, len(grid)), dtype=np.int64) for method in methods}
            squares = {method: np.zeros((2, len(grid)), dtype=np.int64) for method in methods}
            for start in range(0, noise_samples, batch_size):
                noise = bank[start:start + batch_size]
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    samples = noise[None, :, None, :] + effects[:, None, None, None] * profiles[None, None] / np.sqrt(variance[b])
                pvalues = scorer.pvalues(samples.reshape(-1, 2, 4))
                for method, pvalue in pvalues.items():
                    p = pvalue.reshape(len(effects), len(noise), 2)
                    # Average amplitudes WITHIN each independent bank row.
                    success = (p[..., None] <= reference_level * grid).sum(axis=0)
                    counts[method] += success.sum(axis=0)
                    squares[method] += (success * success).sum(axis=0)
            for method in methods:
                count = counts[method].astype(float)
                predictions[method][:, b] = count / (noise_samples * len(effects))
                sample_var = np.maximum(squares[method] - count * count / noise_samples, 0.) / (noise_samples - 1)
                errors[method][:, b] = np.sqrt(sample_var / noise_samples) / len(effects)
        for method in methods:
            result = [_allocate(predictions[method][sign], grid) for sign in range(2)]
            tables[method], overspend[method], underspend[method] = [np.stack([row[j] for row in result]) for j in range(3)]
            fractions[method] = np.array([row[3] for row in result])
            surrogate[method] = [row[4] for row in result]
    diagnostics = {
        "bins": bins, "reference_level": reference_level, "effects": tuple(effects),
        "weight_grid": tuple(grid), "seed": seed, "noise_samples": noise_samples,
        "batch_size": batch_size, "bank_sha256": bank_digest,
        "gaussian_exact_ones": bool(np.isinf(prior_df)),
        "profile_normalization": "each sign row divided by its maximum; amplitude in original location units",
        "simes_policy": SIMES_POLICY,
        "support_simes_fallbacks": tuple(sum(not e["signs"][s]["eligible"] for e in scorer.intersections) for s in range(2)),
        "simes_fallbacks": sum(not e["eligible"] for e in scorer.intersections),
        "convexified_midpoint_proxy_power": surrogate,
        "uses_observed_means": False, "uses_truth_labels": False,
        "integral_measure": "MODEL uniform U, separately for each sign and method",
        "optimality_claim": False,
        "mc_uncertainty": "pointwise SE over independent bank rows, amplitudes clustered; selection and cross-bin dependence not covered",
    }
    maps = [MappingProxyType({k: _readonly(v) for k, v in values.items()})
            for values in (tables, overspend, underspend, fractions, predictions, errors)]
    return PCWeightTables(dimension, prior_df, prior_scale, projection_variance,
                          _readonly(shape), _readonly(profiles), *maps, MappingProxyType(diagnostics))
