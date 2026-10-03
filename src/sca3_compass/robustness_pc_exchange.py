"""Bounded known-t25/.65 PC exchange experiment; finite-row inference only.

The LP is development. Independent scalar bounds do not cover the continuous
directional null and do not certify an e-value. No production integration.
"""
from itertools import permutations, product

import numpy as np
from scipy.special import xlogy

from .robustness_pc_cells import (
    DEFAULT_CELLS, baseline_from_ordered, cells_from_ordered, equicorrelation,
    evaluate_summary, fit_cell_lp, ordered_observations,
)

TAUS = (.001, .003)
KINDS = ('projection', 'simes')
BETA_FRACTIONS = (.5, .75, .9)
PAIRS = tuple((f'{kind}_{tau:g}_b{fraction:g}', kind, tau, fraction)
              for kind, tau, fraction in product(KINDS, TAUS, BETA_FRACTIONS))
ARRAY_FIELDS = ('cell_counts', 'overlap_tau_counts', 'baseline_beta_counts',
                'baseline_tau_counts', 'inside_box_counts',
                'simple_enhanced_counts', 'overlap_simple_counts')


def _means(value):
    value = np.asarray(value, float)
    if value.ndim != 2 or value.shape[1] != 4 or not len(value) or not np.isfinite(value).all():
        raise ValueError('Nonempty finite M by 4 means required')
    return value


def face_membership(means):
    means = _means(means)
    membership = np.array([np.all(means[:, np.arange(4) != j] <= 0, axis=1)
                           for j in range(4)]).T
    if not membership.any(axis=1).all():
        raise ValueError('Every row must be in the directional PC null')
    return membership


def _face_rows(free, negative_patterns):
    rows = []
    for j in range(4):
        keep = np.delete(np.arange(4), j)
        for a, negative in product(free, negative_patterns):
            mu = np.zeros(4)
            mu[j], mu[keep] = a, negative
            rows.append(mu)
    return rows


def null_design():
    """All four faces; .0625 boundary spacing and unequal negative scans.

    This is a fixed finite search design, NOT a cover of the null faces.
    """
    coarse = _face_rows((-4., 0., 1., 2., 3., 4., 6., 8., 12.),
                        ((0., 0., 0.), (-.25, 0., 0.), (0., -1., 0.),
                         (-.0625, -.25, -1.)))
    coarse = np.unique(coarse, axis=0)
    boundary = np.r_[[-12., -4., -1.], np.arange(0., 6.0001, .0625),
                     [6.5, 7., 8., 10., 12., 20.]]
    negatives = sorted(set(permutations((-.0625, 0., 0.)))
                       | set(permutations((-.25, 0., 0.)))
                       | set(permutations((-.0625, -.25, -1.)))
                       | set(permutations((-1., -3., 0.))))
    rows = _face_rows(boundary, ((0., 0., 0.),))
    rows += _face_rows(np.arange(.125, 6., .25), negatives)
    rows += list(permutations((-.0625, -.25, -1., -3.)))
    pool = np.unique(np.vstack((coarse, rows)), axis=0)
    lookup = {tuple(row): i for i, row in enumerate(pool)}
    initial = np.array([lookup[tuple(row)] for row in coarse], dtype=int)
    return pool, initial, face_membership(pool)


def audit_sentinels():
    """Frozen offsets absent from TRAIN, including unequal negatives/tails."""
    rows = _face_rows((.03125, .53125, 1.03125, 1.53125, 2.03125, 2.53125,
                       3.53125, 4.53125, 5.53125, 7.53125, 12., 20.),
                      ((0., 0., 0.),))
    rows += _face_rows((1.28125, 2.78125, 4.28125),
                       ((-.03125, -.09375, -.1875),))
    rows += [[0., 0., 0., 0.], [-.03125, -.09375, -.1875, -.375]]
    result = np.unique(rows, axis=0)
    face_membership(result)
    return result


def alternative_design():
    rows, labels = [], []
    for effect in (.5, 1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6.):
        rows.append([effect] * 4)
        labels.append(f'all4_{effect:g}')
    for support in (2, 3):
        for effect in (2.5, 3.5, 4.5):
            rows.append([effect] * support + [0.] * (4 - support))
            labels.append(f'support{support}_{effect:g}')
    rows.extend([[1., 2., 3., 4.], [.5, 2.5, 2.5, 2.5]])
    labels.extend(['unequal_all4', 'one_weak_all4'])
    return np.asarray(rows), labels


def noise_batches(n, seed, batch_size=16384):
    """Independent normal/radial streams make the bank batching-invariant.

    Each draw has ONE radial variable shared across its four coordinates.
    """
    if not isinstance(n, int) or n < 1 or not isinstance(batch_size, int) or not 1 <= batch_size <= 65536:
        raise ValueError('Positive integer n and batch_size <=65536 required')
    normals, radial = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    factor = np.linalg.cholesky(equicorrelation(.65)).T
    for start in range(0, n, batch_size):
        size = min(batch_size, n - start)
        yield ((normals.normal(size=(size, 4)) @ factor)
               / np.sqrt(radial.chisquare(25., size=(size, 1)) / 25.))


def summarize_batches(batches, means, progress=None):
    """Bound memory by observation batch; CRN across all means/comparators.

    The simple enhancement is b_own(f*tau) OR b_other((1-f)*tau), a fixed
    union-bound comparator. Every count includes observations outside [0,6]^4.
    """
    means = _means(means)
    result = {}
    for key, _, _, _ in PAIRS:
        result[key] = {'n': 0}
        for field in ARRAY_FIELDS:
            shape = (len(means), DEFAULT_CELLS.count) if field in (
                'cell_counts', 'overlap_tau_counts', 'overlap_simple_counts') else (len(means),)
            result[key][field] = np.zeros(shape, dtype=np.int32)
    total = 0
    for noise in batches:
        noise = _means(noise)
        total += len(noise)
        if total > np.iinfo(np.int32).max:
            raise ValueError('Bounded experiment requires n <= int32 maximum')
        for j, mu in enumerate(means):
            ordered = ordered_observations(noise + mu)
            ids = cells_from_ordered(ordered)
            inside = ids >= 0
            for key, kind, tau, fraction in PAIRS:
                lo = baseline_from_ordered(ordered, fraction * tau, kind=kind)
                hi = baseline_from_ordered(ordered, tau, kind=kind)
                other = 'simes' if kind == 'projection' else 'projection'
                simple = lo | baseline_from_ordered(ordered, (1 - fraction) * tau, kind=other)
                if np.any(lo & ~hi):
                    raise ArithmeticError('Baseline nesting failed')
                eligible = inside & ~lo
                item = result[key]
                for field, mask in (('cell_counts', eligible), ('overlap_tau_counts', eligible & hi),
                                    ('overlap_simple_counts', eligible & simple)):
                    item[field][j] += np.bincount(ids[mask], minlength=DEFAULT_CELLS.count)
                item['baseline_beta_counts'][j] += lo.sum()
                item['baseline_tau_counts'][j] += hi.sum()
                item['inside_box_counts'][j] += inside.sum()
                item['simple_enhanced_counts'][j] += simple.sum()
        if progress is not None:
            progress(total)
    if total == 0:
        raise ValueError('Nonempty bank required')
    for item in result.values():
        item['n'] = total
    return result


def subset_summary(summary, indices):
    return {k: v if k == 'n' else np.asarray(v)[indices] for k, v in summary.items()}


def exchange_fit(null_summaries, alt_summaries, initial_indices, *, max_rounds=6, add_per_pair=24):
    """Union adverse rows across BOTH methods/taus, then refit all on same rows.

    All pool frequencies are TRAIN. No CI is asserted for these adapted LPs.
    A capped loop records unresolved finite-pool violations rather than hiding
    them. Every fitted weight, objective, slack, dual and solver receipt survives.
    """
    if not isinstance(max_rounds, int) or not 1 <= max_rounds <= 12 or not isinstance(add_per_pair, int) or add_per_pair < 1:
        raise ValueError('Bounded positive exchange limits required')
    row_count = len(null_summaries[PAIRS[0][0]]['cell_counts'])
    active = np.unique(np.asarray(initial_indices, dtype=int))
    if not len(active) or np.any((active < 0) | (active >= row_count)):
        raise ValueError('Nonempty valid initial constraint indices required')
    rounds = []
    for iteration in range(max_rounds + 1):
        fits, additions = {}, set()
        for key, _, tau, fraction in PAIRS:
            summary = null_summaries[key]
            weights, receipt, arrays = fit_cell_lp(subset_summary(summary, active), alt_summaries[key],
                                                  tau, beta_fraction=fraction, baseline_accounting='bound')
            if weights is None or not receipt['success']:
                # Analytic-baseline constraints always admit zero; preserve and
                # expose a solver failure before the runner stops.
                fits[key] = {'receipt': receipt, 'arrays': arrays, 'weights': weights}
                continue
            added = np.asarray(summary['cell_counts']) @ weights / summary['n']
            ratio = added / ((1 - fraction) * tau)
            violated = np.flatnonzero(ratio > 1 + 1e-7)
            available = np.setdiff1d(violated, active)
            ranked = available[np.argsort(-ratio[available], kind='stable')][:add_per_pair]
            additions.update(ranked.tolist())
            fits[key] = {'receipt': receipt, 'arrays': arrays, 'weights': weights,
                         'pool_added_budget_ratio': ratio,
                         'pool_violation_count': len(violated),
                         'selected_indices': ranked}
        rounds.append({'iteration': iteration, 'active_indices': active.copy(), 'fits': fits,
                       'next_indices': np.array(sorted(additions), dtype=int)})
        if any(not item['receipt']['success'] for item in fits.values()) or not additions or iteration == max_rounds:
            break
        active = np.unique(np.r_[active, sorted(additions)])
    return rounds


def kl_upper(mean, n, alpha, *, bound=1.):
    """One-sided Hoeffding-Chernoff upper bound for a FROZEN scalar in [0,b].

    Solve n*kl(mean/b, u/b)=log(1/alpha). Fractional observations are allowed;
    this is NOT a binomial interval on fractional counts. Returned bisection
    endpoint is the conservative upper side, in ordinary floating point.
    """
    mean = np.asarray(mean, float)
    if not isinstance(n, (int, np.integer)) or n < 1 or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError('Positive integer sample count and alpha in (0,1) required')
    if not np.isfinite(bound) or bound < 0 or not np.isfinite(mean).all() or np.any((mean < 0) | (mean > bound)):
        raise ValueError('Finite means within the stated nonnegative range required')
    if bound == 0:
        return np.zeros_like(mean)
    p = mean / bound
    lower, upper = p.copy(), np.ones_like(p)
    target = -np.log(alpha) / n
    with np.errstate(divide='ignore', invalid='ignore'):
        for _ in range(64):
            mid = (lower + upper) / 2
            divergence = (xlogy(p, p) - xlogy(p, mid)
                          + xlogy(1 - p, 1 - p) - xlogy(1 - p, 1 - mid))
            smaller = divergence < target
            lower = np.where(smaller, mid, lower)
            upper = np.where(smaller, upper, mid)
    return np.minimum(bound, np.nextafter(upper * bound, np.inf))


def kl_lower(mean, n, alpha, *, bound=1.):
    mean = np.asarray(mean, float)
    # kl_upper validates mean and its range even for a degenerate bound.
    kl_upper(mean, n, alpha, bound=bound)
    return np.maximum(0., np.nextafter(bound - kl_upper(bound - mean, n, alpha, bound=bound), -np.inf))


def scalar_added_bounds(summary, weights, alpha):
    """Condition on weights/rows fixed before this bank; alpha is per row."""
    weights = np.asarray(weights, float)
    if weights.shape != (DEFAULT_CELLS.count,) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError('One finite [0,1] weight for each fixed cell required')
    added = np.asarray(summary['cell_counts']) @ weights / summary['n']
    bound = float(weights.max())
    return {'mean': added, 'upper': kl_upper(added, summary['n'], alpha, bound=bound),
            'range_upper': bound, 'alpha_per_row': alpha}


def calibrate_once(summaries, weights_by_key, *, family_alpha=.01, guard=.8):
    """One prespecified shrink; this bank is thereafter DEVELOPMENT.

    With probability >=1-family_alpha, each returned decision meets the
    analytic-baseline plus guarded added-mass budget on these finite rows.
    This is not an unconditional e-validity or continuous-null statement.
    """
    if not 0 < family_alpha < 1 or not 0 < guard <= 1:
        raise ValueError('Positive subunit error and guard <=1 required')
    multiplicity = sum(len(summaries[key]['cell_counts']) for key, _, _, _ in PAIRS)
    alpha = family_alpha / multiplicity
    result = {}
    for key, _, tau, fraction in PAIRS:
        weights = np.asarray(weights_by_key[key], float)
        bounds = scalar_added_bounds(summaries[key], weights, alpha)
        maximum = float(bounds['upper'].max())
        shrink = 1. if maximum == 0 else min(1., guard * (1 - fraction) * tau / maximum)
        result[key] = {**bounds, 'shrink': shrink, 'weights': weights * shrink,
                       'family_alpha': family_alpha, 'multiplicity': multiplicity,
                       'guard': guard, 'development_after_use': True,
                       'finite_row_high_probability_total_upper': fraction * tau + shrink * bounds['upper'],
                       'continuous_null_coverage': False}
    return result


def audit_frozen(summaries, weights_by_key, *, family_alpha=.01):
    if not 0 < family_alpha < 1:
        raise ValueError('family_alpha in (0,1) required')
    multiplicity = sum(len(summaries[key]['cell_counts']) for key, _, _, _ in PAIRS)
    result = {}
    for key, _, tau, fraction in PAIRS:
        bounds = scalar_added_bounds(summaries[key], weights_by_key[key], family_alpha / multiplicity)
        total = fraction * tau + bounds['upper']
        result[key] = {**bounds, 'total_upper': total, 'total_upper_over_tau': total / tau,
                       'rows_upper_above_tau': int(np.sum(total > tau)),
                       'family_alpha': family_alpha, 'multiplicity': multiplicity,
                       'continuous_null_coverage': False,
                       'empirical': evaluate_summary(summaries[key], weights_by_key[key], tau)}
    return result


def power_contrast(summary, weights, tau, *, comparison='tau', alpha_per_tail=.00001):
    """Signed paired gain from disjoint nonnegative gain/loss components.

    Four one-sided scalar tails bound BOTH ends of the contrast. The caller
    accounts for all rows and comparators. No independence of CRN rows needed.
    """
    weights = np.asarray(weights, float)
    scalar_added_bounds(summary, weights, alpha_per_tail)  # validate
    if comparison not in ('tau', 'simple'):
        raise ValueError('comparison must be tau or simple')
    overlap_field = 'overlap_tau_counts' if comparison == 'tau' else 'overlap_simple_counts'
    target_field = 'baseline_tau_counts' if comparison == 'tau' else 'simple_enhanced_counts'
    n = summary['n']
    cells, overlap = np.asarray(summary['cell_counts']), np.asarray(summary[overlap_field])
    target = np.asarray(summary[target_field]) / n
    beta = np.asarray(summary['baseline_beta_counts']) / n
    positive = (cells - overlap) @ weights / n
    negative = np.maximum(0., target - beta - overlap @ weights / n)
    positive_upper = kl_upper(positive, n, alpha_per_tail, bound=float(weights.max()))
    positive_lower = kl_lower(positive, n, alpha_per_tail, bound=float(weights.max()))
    negative_upper = kl_upper(negative, n, alpha_per_tail)
    negative_lower = kl_lower(negative, n, alpha_per_tail)
    gain = positive - negative
    second = ((cells - overlap) @ (weights**2) / n + target - beta
              + overlap @ (weights**2 - 2 * weights) / n)
    return {'comparison': comparison, 'baseline_power': target,
            'candidate_power': beta + cells @ weights / n, 'gain': gain,
            'positive_mass': positive, 'negative_mass': negative,
            'paired_mc_se': np.sqrt(np.maximum(0, second - gain**2) / max(1, n - 1)),
            'simultaneous_gain_lower': positive_lower - negative_upper,
            'simultaneous_gain_upper': positive_upper - negative_lower,
            'alpha_per_tail': alpha_per_tail, 'four_tails_per_contrast': True}
