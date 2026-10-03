"""Known-model joint-PC cell LP prototype. ALL fits are DEVELOPMENT, uncertified.

The finite Monte Carlo constraint design is NOT the continuous directional null.
No nuisance fitting, project integration, or pipeline-heterogeneous guarantee.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
import warnings

import numpy as np
from scipy.optimize import linprog
from scipy.stats import t


@dataclass(frozen=True)
class CellSpec:
    min_edges: tuple = (0., .5, 1., 1.5, 2., 2.5, 3., 4., 6.)
    second_largest_edges: tuple = (0., 1., 1.5, 2., 2.5, 3., 3.5, 4.5, 6.)
    spread_edges: tuple = (0., .5, 1., 1.5, 2., 2.5, 3.5, 4.5, 6.)
    lower: float = 0.
    upper: float = 6.

    def __post_init__(self):
        if self.lower != 0 or self.upper != 6:
            raise ValueError('Prototype observation domain is fixed [0,6]^4')
        for edges in self.edges:
            a = np.asarray(edges, float)
            if len(a) < 2 or not np.isfinite(a).all() or np.any(np.diff(a) <= 0) or a[0] != 0 or a[-1] != 6:
                raise ValueError('Strict finite edges covering [0,6] required')
        if self.count > 512:
            raise ValueError('At most 512 prespecified cells')

    @property
    def edges(self):
        return (self.min_edges, self.second_largest_edges, self.spread_edges)

    @property
    def count(self):
        return int(np.prod([len(e) - 1 for e in self.edges]))

    def receipt(self):
        return {**asdict(self), 'cell_count': self.count,
                'features': ['minimum', 'second_largest', 'maximum_minus_minimum'],
                'right_endpoint_6_included': True}


DEFAULT_CELLS = CellSpec()


def equicorrelation(rho=.65):
    if not np.isfinite(rho) or not -1 / 3 < rho < 1:
        raise ValueError('Four-dimensional SPD equicorrelation required')
    return (1 - rho) * np.eye(4) + rho * np.ones((4, 4))


def ordered_observations(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or x.shape[1] != 4 or not np.isfinite(x).all():
        raise ValueError('Finite N by 4 observations required')
    return np.sort(x, axis=1)


def cells_from_ordered(ordered, spec=DEFAULT_CELLS):
    """-1 outside original coordinate box; outside data are NOT discarded."""
    ordered = np.asarray(ordered, float)
    if ordered.ndim != 2 or ordered.shape[1] != 4 or not np.isfinite(ordered).all() or np.any(np.diff(ordered, axis=1) < 0):
        raise ValueError('Finite sorted N by 4 observations required')
    inside = (ordered[:, 0] >= spec.lower) & (ordered[:, 3] <= spec.upper)
    out = np.full(len(ordered), -1, dtype=np.int32)
    z = ordered[inside]
    features = (z[:, 0], z[:, 2], z[:, 3] - z[:, 0])
    indices = [np.minimum(np.searchsorted(edge, v, side='right') - 1, len(edge) - 2)
               for edge, v in zip(spec.edges, features)]
    if len(z):
        out[inside] = np.ravel_multi_index(tuple(indices), tuple(len(e) - 1 for e in spec.edges))
    return out


@lru_cache(maxsize=32)
def baseline_thresholds(level, df=25., rho=.65, kind='projection'):
    if not 0 < level < .5 or not np.isfinite(df) or df <= 0:
        raise ValueError('Positive finite Student df and small test level required')
    equicorrelation(rho)
    if kind == 'projection':
        return (float(t.isf(level, df) * np.sqrt(3 + 6 * rho)),)
    if kind == 'simes' and rho >= 0:
        return tuple(float(t.isf(level * k / 3, df)) for k in (1, 2, 3))
    raise ValueError('Known-model baseline must be projection or nonnegative-shape Simes')


def baseline_from_ordered(ordered, level, df=25., rho=.65, kind='projection'):
    thresholds = baseline_thresholds(float(level), float(df), float(rho), kind)
    if kind == 'projection':
        # Minimum over the four equally weighted triple projections = sum of
        # the three smallest coordinates. Do not test a four-study global null.
        return ordered[:, :3].sum(axis=1) >= thresholds[0]
    # Simes on the largest three marginal p-values, retaining the PC order.
    return ((ordered[:, 2] >= thresholds[0]) | (ordered[:, 1] >= thresholds[1])
            | (ordered[:, 0] >= thresholds[2]))


def cell_decision(x, weights, tau, *, baseline='projection', beta_fraction=.9,
                  df=25., rho=.65, spec=DEFAULT_CELLS):
    weights = np.asarray(weights, float)
    if weights.shape != (spec.count,) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError('One [0,1] weight per cell required')
    if not 0 < beta_fraction < 1:
        raise ValueError('Strictly smaller retained baseline level required')
    ordered = ordered_observations(x)
    retained = baseline_from_ordered(ordered, beta_fraction * tau, df, rho, baseline)
    ids = cells_from_ordered(ordered, spec)
    eligible = (ids >= 0) & ~retained
    phi = retained.astype(float)
    phi[eligible] = weights[ids[eligible]]
    return phi


def summarize_bank(noise, means, tau, *, baseline='projection', beta_fraction=.9,
                   df=25., rho=.65, spec=DEFAULT_CELLS):
    """Raw counts sufficient for mean phi, second moments and paired contrasts.

    The SAME noise bank is reused for every mean (common random numbers).
    Independence across constraint rows is never asserted.
    """
    noise = np.asarray(noise, float)
    means = np.asarray(means, float)
    if noise.ndim != 2 or noise.shape[1] != 4 or means.ndim != 2 or means.shape[1] != 4 or not np.isfinite(noise).all() or not np.isfinite(means).all():
        raise ValueError('Finite noise/means with four coordinates required')
    if len(noise) == 0:
        raise ValueError('Nonempty Monte Carlo bank required')
    n = len(noise)
    counts = np.zeros((len(means), spec.count), dtype=np.int64)
    overlap = np.zeros_like(counts)
    base_beta = np.zeros(len(means), dtype=np.int64)
    base_tau = np.zeros(len(means), dtype=np.int64)
    inside = np.zeros(len(means), dtype=np.int64)
    for j, mu in enumerate(means):
        ordered = ordered_observations(noise + mu)
        ids = cells_from_ordered(ordered, spec)
        lo = baseline_from_ordered(ordered, tau * beta_fraction, df, rho, baseline)
        hi = baseline_from_ordered(ordered, tau, df, rho, baseline)
        if np.any(lo & ~hi):
            raise ArithmeticError('Baseline nesting failed')
        eligible = (ids >= 0) & ~lo
        counts[j] = np.bincount(ids[eligible], minlength=spec.count)
        overlap[j] = np.bincount(ids[eligible & hi], minlength=spec.count)
        base_beta[j], base_tau[j], inside[j] = lo.sum(), hi.sum(), (ids >= 0).sum()
    return {'n': n, 'cell_counts': counts, 'overlap_tau_counts': overlap,
            'baseline_beta_counts': base_beta, 'baseline_tau_counts': base_tau,
            'inside_box_counts': inside}


def summarize_bank_many(noise, means, taus=(.001, .003), baselines=('projection', 'simes'),
                        *, beta_fraction=.9, df=25., rho=.65, spec=DEFAULT_CELLS):
    """Same counts as summarize_bank; share sorting and cells across comparators."""
    noise, means = np.asarray(noise, float), np.asarray(means, float)
    if noise.ndim != 2 or noise.shape[1] != 4 or means.ndim != 2 or means.shape[1] != 4 or not len(noise) or not np.isfinite(noise).all() or not np.isfinite(means).all():
        raise ValueError('Finite nonempty N by 4 bank and M by 4 means required')
    output = {}
    for kind, tau in product(baselines, taus):
        baseline_thresholds(float(tau), df, rho, kind)
        key = f'{kind}_{tau:g}'
        output[key] = {'n': len(noise), 'cell_counts': np.zeros((len(means), spec.count), dtype=np.int64),
                       'overlap_tau_counts': np.zeros((len(means), spec.count), dtype=np.int64),
                       'baseline_beta_counts': np.zeros(len(means), dtype=np.int64),
                       'baseline_tau_counts': np.zeros(len(means), dtype=np.int64),
                       'inside_box_counts': np.zeros(len(means), dtype=np.int64)}
    for j, mu in enumerate(means):
        ordered = ordered_observations(noise + mu)
        ids = cells_from_ordered(ordered, spec)
        inside = ids >= 0
        for kind, tau in product(baselines, taus):
            item = output[f'{kind}_{tau:g}']
            lo = baseline_from_ordered(ordered, tau * beta_fraction, df, rho, kind)
            hi = baseline_from_ordered(ordered, tau, df, rho, kind)
            if np.any(lo & ~hi):
                raise ArithmeticError('Baseline nesting failed')
            eligible = inside & ~lo
            item['cell_counts'][j] = np.bincount(ids[eligible], minlength=spec.count)
            item['overlap_tau_counts'][j] = np.bincount(ids[eligible & hi], minlength=spec.count)
            item['baseline_beta_counts'][j] = lo.sum()
            item['baseline_tau_counts'][j] = hi.sum()
            item['inside_box_counts'][j] = inside.sum()
    return output


def fit_cell_lp(null_summary, alternative_summary, tau, *, beta_fraction=.9,
                baseline_accounting='clipped_estimate'):
    """Finite-grid development LP, including all raw constraint diagnostics.

    strict_empirical: raw Monte Carlo baseline probabilities; can be infeasible.
    clipped_estimate: min(raw estimate, proven beta bound); NOT an upper CI.
    bound: use analytic beta for every row (conservative baseline accounting,
           but still only empirical finite-grid cell probabilities).
    """
    n, an = null_summary['n'], alternative_summary['n']
    probs = np.asarray(null_summary['cell_counts'], float) / n
    raw = np.asarray(null_summary['baseline_beta_counts'], float) / n
    objective = np.asarray(alternative_summary['cell_counts'], float).mean(axis=0) / an
    beta_level = tau * beta_fraction
    if baseline_accounting == 'strict_empirical':
        used = raw.copy()
    elif baseline_accounting == 'clipped_estimate':
        used = np.minimum(raw, beta_level)
    elif baseline_accounting == 'bound':
        used = np.full(len(raw), beta_level)
    else:
        raise ValueError('Unknown baseline accounting')
    upper = np.where(objective > 0, 1., 0.)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = linprog(-objective, A_ub=probs / tau, b_ub=(tau - used) / tau,
                         bounds=list(zip(np.zeros(len(objective)), upper)), method='highs',
                         options={'threads': 1, 'primal_feasibility_tolerance': 1e-8,
                                  'dual_feasibility_tolerance': 1e-8})
    weights = None if result.x is None else np.clip(result.x, 0, 1)
    receipt = {'status': 'DEVELOPMENT_ONLY_NOT_CERTIFIED', 'baseline_accounting': baseline_accounting,
               'success': bool(result.success), 'solver_status': int(result.status),
               'solver_message': result.message, 'iterations': int(result.nit),
               'tau': float(tau), 'beta': float(beta_level),
               'raw_baseline_above_tau_count': int(np.sum(raw > tau)),
               'raw_baseline_above_beta_count': int(np.sum(raw > beta_level)),
               'raw_baseline_above_tau_max': float(raw.max() / tau),
               'warnings': [str(w.message) for w in caught],
               'continuous_null_coverage': False, 'monte_carlo_upper_bound': False}
    if weights is not None:
        fitted = used + probs @ weights
        actual_empirical = raw + probs @ weights
        receipt.update({'fitted_max_Ephi_over_tau': float(fitted.max() / tau),
                        'raw_max_Ephi_over_tau': float(actual_empirical.max() / tau),
                        'active_weights': int(np.sum(weights > 1e-10)),
                        'fractional_weights': int(np.sum((weights > 1e-10) & (weights < 1 - 1e-10))),
                        'alternative_added_mass': float(objective @ weights),
                        'max_scaled_primal_violation': float(max(0, np.max((fitted - tau) / tau)))})
    return weights, receipt, {'baseline_used': used, 'baseline_raw': raw,
                               'objective': objective, 'cell_probabilities': probs,
                               'fitted_slack': None if weights is None else tau - used - probs @ weights,
                               'dual_marginals': None if not result.success else result.ineqlin.marginals}


def evaluate_summary(summary, weights, tau):
    n = summary['n']
    counts = np.asarray(summary['cell_counts'], float)
    weights = np.asarray(weights, float)
    b = np.asarray(summary['baseline_beta_counts'], float) / n
    target = np.asarray(summary['baseline_tau_counts'], float) / n
    added = counts @ weights / n
    mean = b + added
    moment2 = b + counts @ (weights * weights) / n
    covariance_term = b + np.asarray(summary['overlap_tau_counts'], float) @ weights / n
    delta = mean - target
    delta_second = moment2 + target - 2 * covariance_term
    se = np.sqrt(np.maximum(0, moment2 - mean**2) / max(1, n - 1))
    delta_se = np.sqrt(np.maximum(0, delta_second - delta**2) / max(1, n - 1))
    return {'baseline_beta': b, 'baseline_tau': target, 'added_cell_mass': added,
            'mean_phi': mean, 'Ephi_over_tau': mean / tau, 'mc_se': se,
            'gain_vs_tau': delta, 'paired_mc_se_gain': delta_se,
            'baseline_reduction_cost': target - b}


def student_location_lipschitz(df, dimension=4):
    if not np.isfinite(df) or df <= 0 or not isinstance(dimension, int) or dimension < 1:
        raise ValueError('Positive finite Student df and integer dimension required')
    return float((df + dimension) / (2 * np.sqrt(df)))


def parameter_box_radius(lower, upper, shape, center=None):
    """Vertex formula in FLOATING POINT, not interval-certified arithmetic."""
    lower, upper, shape = np.asarray(lower, float), np.asarray(upper, float), np.asarray(shape, float)
    if lower.shape != (4,) or upper.shape != (4,) or shape.shape != (4, 4) or not np.isfinite(np.r_[lower, upper, shape.ravel()]).all() or np.any(lower > upper) or not np.allclose(shape, shape.T):
        raise ValueError('Finite four-dimensional box and symmetric SPD shape required')
    np.linalg.cholesky(shape)
    center = (lower + upper) / 2 if center is None else np.asarray(center, float)
    if center.shape != (4,) or not np.isfinite(center).all():
        raise ValueError('Finite four-dimensional center required')
    vertices = np.array(list(product(*zip(lower, upper))))
    difference = vertices - center
    quadratic = np.einsum('ni,ij,nj->n', difference, np.linalg.inv(shape), difference)
    return float(np.sqrt(max(0, quadratic.max())))


def density_inflation(radius, df):
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Nonnegative finite radius required')
    with np.errstate(over='ignore'):
        return float(np.exp(student_location_lipschitz(df) * radius))


def outer_null_bound(radius, df, *, beta, observation_radius=6., sigma_max=1.):
    """beta + tail bounds additions in [0,6]^4 outside [-radius,radius]^4."""
    if not np.isfinite([radius, df, beta, observation_radius, sigma_max]).all() or radius <= observation_radius or df <= 0 or not 0 <= beta < 1 or observation_radius < 0 or sigma_max <= 0:
        raise ValueError('Outer radius larger than observation radius required')
    return float(min(1, beta + t.sf((radius - observation_radius) / sigma_max, df)))


def required_outer_radius(tau, beta, df, *, observation_radius=6., sigma_max=1.):
    if not 0 < beta < tau < .5 or df <= 0 or not np.isfinite(df) or observation_radius < 0 or sigma_max <= 0:
        raise ValueError('Strict positive tail budget required')
    return float(observation_radius + sigma_max * t.isf(tau - beta, df))
