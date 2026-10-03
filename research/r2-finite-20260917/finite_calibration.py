"""R2 prototype. Separate from frozen V1. Exact-arithmetic theory in THEORY.md.

No claim that float64 linear algebra is interval-certified. Input failures
raise explicitly. All randomized shape references are drawn anew per call.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
from fractions import Fraction
from math import comb
from pathlib import Path
import sys
import numpy as np
from scipy.linalg import helmert
from scipy.stats import beta, f, t

VERSION = 'R2-FC-prototype-0.1'
sys.dont_write_bytecode = True
PROJECT = next(p for p in Path(__file__).resolve().parents if (p/'releases/K-NR-1.0.0/release.json').exists())
RELEASE = PROJECT / 'releases/K-NR-1.0.0/source/src'
if str(RELEASE) not in sys.path:
    sys.path.insert(0, str(RELEASE))
from sca3_compass.robustness_patterns import fit_pattern_mixture
from sca3_compass.robustness_projection import positive_direction
from sca3_compass.robustness_calibrators import focused_calibrator
from sca3_compass.molecular_methods import ebh, fdr_adjust


def contrasts(j):
    return helmert(j, full=False).T


def shape_fit(y, iterations=12):
    """Fixed-iteration affine-equivariant, block-radial-invariant shape.

    y shape (..., n_blocks, S, J-1); J-1>=S. No ridge or diag normalization.
    Initialization sum(W_g/det(W_g)**(1/S)) is affine equivariant up to scale.
    A fixed number of Tyler updates preserves this without assuming convergence.
    """
    y = np.asarray(y, float)
    if y.ndim < 3 or not np.isfinite(y).all():
        raise ValueError('finite contrast blocks required')
    n, s, d = y.shape[-3:]
    if d < s or n < 2 or iterations < 0:
        raise ValueError('requires J-1>=S and at least two independent blocks')
    scale = np.max(np.abs(y), axis=(-1, -2), keepdims=True)
    if np.any(scale <= 0):
        raise ValueError('zero contrast block outside model')
    y = y / scale
    w = y @ np.swapaxes(y, -1, -2)
    sign, logdet = np.linalg.slogdet(w)
    if np.any(sign <= 0):
        raise ArithmeticError('singular initialization')
    h = np.mean(w * np.exp(-logdet / s)[..., None, None], axis=-3)
    h *= (s / np.trace(h, axis1=-2, axis2=-1))[..., None, None]
    x = np.swapaxes(y, -1, -2).reshape(y.shape[:-3] + (n*d, s))
    # Independent positive scaling of columns has no effect on Tyler updates.
    x = x / np.linalg.norm(x, axis=-1, keepdims=True)
    for _ in range(iterations):
        inverse = np.linalg.inv(h)
        q = np.einsum('...ni,...ij,...nj->...n', x, inverse, x)
        h = np.swapaxes(x, -1, -2) @ (x / q[..., None])
        h *= (s / np.trace(h, axis1=-2, axis2=-1))[..., None, None]
    if not np.isfinite(h).all() or np.any(np.linalg.eigvalsh(h)[..., 0] <= 0):
        raise ArithmeticError('non-positive shape')
    return h


def reference_cutoff(rng, blocks, studies, contrasts_count, draws=199, iterations=12):
    """Exchangeability cutoff, failure <=1/(draws+1), not a plug-in quantile.

    Fresh independent references per analysis. Reusing a fixed table while
    asserting a conditional theorem is not allowed by this interface.
    """
    if draws < 1:
        raise ValueError('positive reference count required')
    values = []
    for start in range(0, draws, 32):
        y = rng.normal(size=(min(32, draws-start), blocks, studies, contrasts_count))
        h = shape_fit(y, iterations)
        eigen = np.linalg.eigvalsh(h)
        values.extend((eigen[:, -1] / eigen[:, 0]).tolist())
    return float(max(values)), np.asarray(values)


@lru_cache(maxsize=128)
def kappa_order(n, j, delta):
    """Design-only choice: minimize a deterministic median inflation proxy.

    Each possible k separately has exact coverage. Choosing k using only n,J,
    delta, never observed calibration statistics, preserves this coverage.
    """
    if n < 2 or j < 2 or not 0 < delta < 1:
        raise ValueError('invalid calibration design')
    orders = np.arange(1, n+1)
    denom = f.ppf(beta.ppf(delta, orders, n+1-orders), 1, j-1)
    typical = f.ppf(beta.ppf(.5, orders, n+1-orders), 1, j-1)
    k = int(np.argmin(typical / denom))
    if not np.isfinite(denom[k]) or denom[k] <= 0:
        raise ArithmeticError('invalid coverage quantile')
    return k+1, float(denom[k])


def kappa_upper(calibration_blocks, delta=.005):
    """Use first fixed study direction once per independent calibration BLOCK."""
    c = np.asarray(calibration_blocks, float)
    if c.ndim != 3 or not np.isfinite(c).all():
        raise ValueError('calibration shape (independent_blocks,studies,pipelines)')
    n, _, j = c.shape
    row = c[:, 0, :]
    scale = np.max(np.abs(row), axis=-1, keepdims=True)
    if np.any(scale <= 0):
        raise ValueError('zero calibration vector')
    row = row / scale
    y = row @ contrasts(j)
    den = np.sum(y*y, axis=-1)/(j-1)
    if np.any(den <= 0):
        raise ArithmeticError('zero calibration contrasts')
    a = row.mean(-1)**2/den
    order, quantile = kappa_order(n, j, delta)
    upper = float(np.partition(a, order-1)[order-1]/quantile)
    return upper, {'blocks': n, 'order': order, 'quantile': quantile,
                   'delta': delta, 'row_direction': 0, 'ratios': a}


@lru_cache(maxsize=64)
def matrix_order(n, delta):
    """S4/J6 exact-rational certification of order-statistic denominator.

    F(4,2) CDF is (2x/(1+2x))**2. Beta/order coverage is a finite binomial
    polynomial. SciPy proposes, integer arithmetic certifies a lower quantile.
    This certifies the coverage QUANTILE, not all floating matrix algebra.
    """
    orders = np.arange(1, n+1)
    denominator = f.ppf(beta.ppf(delta, orders, n+1-orders), 4, 2)
    typical = f.ppf(beta.ppf(.5, orders, n+1-orders), 4, 2)
    order = int(np.argmin(typical/denominator))+1
    c = float(denominator[order-1])
    budget = Fraction(str(delta))
    for _ in range(256):
        x = Fraction.from_float(c)
        prob = (2*x/(1+2*x))**2
        a, b = prob.numerator, prob.denominator
        numerator = sum(comb(n, i)*a**i*(b-a)**(n-i) for i in range(order, n+1))
        if numerator*budget.denominator <= budget.numerator*b**n:
            return order, c
        c = np.nextafter(c, 0.)
    raise ArithmeticError('quantile could not be certified, no unchecked fallback')


def matrix_kappa_upper(cal, delta=.005):
    c = np.asarray(cal, float)
    if c.ndim != 3 or c.shape[1:] != (4, 6) or len(c) < 4 or not np.isfinite(c).all():
        raise ValueError('N>=4 independent centered 4x6 calibration blocks required')
    scale = np.max(np.abs(c), axis=(-1, -2), keepdims=True)
    if np.any(scale <= 0):
        raise ValueError('zero calibration block')
    c = c/scale
    x = c.mean(-1)
    y = c @ contrasts(6)
    w = y @ np.swapaxes(y, -1, -2)
    score = .5*np.einsum('ni,ni->n', x, np.linalg.solve(w, x[..., None])[..., 0])
    if np.any(score <= 0) or not np.isfinite(score).all():
        raise ArithmeticError('invalid matrix calibration ratio')
    order, denominator = matrix_order(len(c), delta)
    upper = float(np.partition(score, order-1)[order-1]/denominator)
    median = float(np.median(score)/f.ppf(.5, 4, 2))
    return upper, median, {'blocks': len(c), 'df': [4, 2], 'order': order,
                           'denominator': denominator, 'delta': delta,
                           'quantile_integer_certified': True, 'ratios': score}


def projection_p(z, direction, shape, kappa, shape_factor=1.):
    z = np.asarray(z, float)
    a = np.asarray(direction, float)
    if np.any(a < 0) or not np.any(a > 0):
        raise ValueError('nonnegative nonzero intersection direction required')
    s, j = z.shape[-2:]
    y = z @ contrasts(j)
    energy = np.einsum('...ik,ij,...jk->...', y, np.linalg.inv(shape), y)
    variance = kappa * shape_factor * float(a @ shape @ a) * energy/(s*(j-1))
    if np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ArithmeticError('invalid target scale')
    mean = z.mean(-1) @ a
    stat = mean/np.sqrt(variance)
    return np.where(mean > 0, t.sf(stat, s*(j-1)), 1.)


def pc_components(z, shape, kappa, factor, profiles, direction_shape=None):
    """r=2, arbitrary study dependence. Max over all S-1 null intersections."""
    g, s, j = z.shape
    direction_shape = shape if direction_shape is None else direction_shape
    out = np.empty((g, 2, 2))  # signed, ordinary/projection
    basis = np.eye(s)
    for sign_index, sign in enumerate([1., -1.]):
        rows = np.column_stack([projection_p(sign*z, a, shape, kappa, factor) for a in basis])
        ordinary, projected = [], []
        for subset in combinations(range(s), s-1):
            ordinary.append(np.minimum(1., (s-1)*rows[:, subset].min(axis=1)))
            a = np.zeros(s)
            profile = profiles[sign_index, list(subset)]
            if not np.any(profile):
                profile = np.ones(s-1)
            a[list(subset)] = positive_direction(profile, direction_shape[np.ix_(subset, subset)])
            projected.append(projection_p(sign*z, a, shape, kappa, factor))
        out[:, sign_index, 0] = np.max(ordinary, axis=0)
        out[:, sign_index, 1] = np.max(projected, axis=0)
    return out


def learn_profiles(train, shape, kappa):
    y = train @ contrasts(train.shape[-1])
    energy = np.einsum('gik,ij,gjk->g', y, np.linalg.inv(shape), y)
    var = kappa*energy/(train.shape[1]*(train.shape[2]-1))
    fit = fit_pattern_mixture(train.mean(-1), var, shape, train.shape[1]*(train.shape[2]-1))
    profiles = np.ones((2, train.shape[1])); gamma = np.zeros(2)
    if fit['diagnostics']['converged']:
        for d, key in enumerate(['positive', 'negative']):
            r = fit['replicated'][key]
            profiles[d] = np.maximum(0., (1 if d == 0 else -1)*r['dominant_pattern'])
            if r['mass'] > max(.03, 3/len(train)):
                gamma[d] = np.clip((r['dominant_mass_fraction']-.4)/.4, 0, 1)
    return profiles, gamma, bool(fit['diagnostics']['converged'])


def calibrate(held_p, train_p, family, q):
    count = int(np.sum(fdr_adjust(train_p, 'BH') <= .05))
    threshold = .5*q*max(1, count)/train_p.size
    return focused_calibrator(held_p, family, threshold, .8 if count else 0., q,
                              min(64, family)), {'count': count, 'threshold': threshold}


def evaluate(z, calibration_blocks, *, seed, reference_draws=199, delta_kappa=.005,
             oracle=None, mismatch_bound=1.):
    """Primary plus minimal same-information ordinary and BB references.

    oracle=(R,kappa) is simulation-only; never provided in deployable use.
    At most one fixed shape-reference bank is shared between equal-size folds;
    union bound counts BOTH actual shape estimation events. Fresh bank per run.
    """
    z = np.asarray(z, float)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('nonnegative integer seed required')
    if isinstance(reference_draws, (bool, np.bool_)) or not isinstance(reference_draws, (int, np.integer)) or reference_draws < 1:
        raise ValueError('positive integer reference count required')
    if not np.isfinite(delta_kappa) or not 0 < delta_kappa < .05 or delta_kappa+2/(reference_draws+1) >= .05:
        raise ValueError('invalid coverage/FDR budget')
    if z.ndim != 3 or z.shape[1:] != (4, 6) or len(z)%2 or len(z) < 8 or not np.isfinite(z).all():
        raise ValueError('even G>=8 and target shape Gx4x6 required')
    if not np.isfinite(mismatch_bound) or mismatch_bound < 1:
        raise ValueError('external kappa ratio bound must be >=1')
    g, s, j = z.shape; m = 2*g
    upper, median, cal_receipt = matrix_kappa_upper(calibration_blocks, delta_kappa)
    cutoff, refs = reference_cutoff(np.random.default_rng(seed), g//2, s, j-1, reference_draws)
    delta_shape = 1/(reference_draws+1)
    delta_total = delta_kappa+2*delta_shape
    q = .05-delta_total
    if q <= 0:
        raise ValueError('error budget exhausted')
    tags = ['R2_main', 'R2_ordinary', 'BB_eBH', 'BB_BY', 'known_shape', 'known_all'] if oracle is not None else ['R2_main', 'R2_ordinary', 'BB_eBH', 'BB_BY']
    evidence = {tag: np.empty((g, 2)) for tag in tags if tag != 'BB_BY'}
    bb = np.empty((g, 2)); receipts = []; p_store = np.empty((g, 2, 2))
    for fold in range(2):
        held = np.arange(g)%2 == fold; train = z[~held]
        h = shape_fit(train @ contrasts(j))
        profiles, gamma, converged = learn_profiles(train, h, median)
        # No HELD quantities enter profiles, gamma, or the pilot calibration.
        training_p = pc_components(train, h, median, 1., profiles)
        held_p = pc_components(z[held], h, upper*mismatch_bound, cutoff, profiles)
        p_store[held] = held_p
        components = []
        pilot = []
        for component in range(2):
            value, receipt = calibrate(held_p[:, :, component], training_p[:, :, component], m, q)
            components.append(value); pilot.append(receipt)
        evidence['R2_ordinary'][held] = components[0]
        evidence['R2_main'][held] = (1-gamma)*components[0]+gamma*components[1]
        # Classical single-hypothesis bad-event/BB penalty; same protected p.
        bb[held] = np.minimum(1., held_p[:, :, 0]+delta_kappa+delta_shape)
        bb_components = [calibrate(np.minimum(1., held_p[:, :, c]+delta_kappa+delta_shape),
                                   training_p[:, :, c], m, .05)[0] for c in range(2)]
        evidence['BB_eBH'][held] = (1-gamma)*bb_components[0]+gamma*bb_components[1]
        info = {'fold': fold, 'shape': h, 'cutoff': cutoff, 'gamma': gamma,
                'profiles': profiles, 'pattern_converged': converged, 'pilots': pilot}
        if oracle is not None:
            true_shape, true_kappa = oracle
            for tag, kap in [('known_shape', upper*mismatch_bound), ('known_all', true_kappa)]:
                actual = pc_components(z[held], true_shape, kap, 1., profiles, direction_shape=h)
                # Same actual directions, pilots, mixture and q; only nuisance denominator changes.
                values = [calibrate(actual[:, :, c], training_p[:, :, c], m, q)[0] for c in range(2)]
                evidence[tag][held] = (1-gamma)*values[0]+gamma*values[1]
            from scipy.linalg import eigh
            eigen = eigh(h, true_shape, eigvals_only=True)
            info['true_shape_distortion'] = float(eigen[-1]/eigen[0])
        receipts.append(info)
    decisions = {tag: ebh(value, .05 if tag == 'BB_eBH' else q)
                 for tag, value in evidence.items()}
    decisions['BB_BY'] = fdr_adjust(bb, 'BY') <= .05
    return {'decisions': decisions, 'evidence': evidence, 'bb_p': bb, 'p': p_store,
            'calibration': cal_receipt, 'kappa_upper': upper, 'kappa_median': median,
            'shape_reference': refs, 'folds': receipts, 'q_internal': q,
            'delta_kappa': delta_kappa, 'delta_shape_each': delta_shape,
            'numerics': 'float64 matrices/tails; exact integer coverage quantile, not fully interval-certified',
            'method_version': VERSION}
