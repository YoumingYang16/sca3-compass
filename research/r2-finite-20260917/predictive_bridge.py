"""Full-contrast pivotal predictive bridge; separate PILOT/LEARN/TEST.

Prototype exact-arithmetic theorem in PREDICTIVE_BRIDGE.md. Numerical matrix
operations are standard floating point, not a formal interval implementation.
"""
from itertools import combinations
import time
import numpy as np
from finite_calibration import (contrasts, shape_fit, matrix_kappa_upper,
                                learn_profiles, pc_components, calibrate,
                                positive_direction, ebh)
from grid_calibration import grid_calibrate

VERSION = 'R2-PB-1.0.0'
F42_MEDIAN = (1+np.sqrt(2))/2


def kappa_estimator(blocks):
    """Only a scale-equivariant median; no discarded confidence computation.

    For centered calibration the finite sampling law is pivotal. For PILOT
    means this is just a learning heuristic, NOT a claimed coverage bound.
    SVD avoids squaring the conditioning of the contrast matrix.
    """
    scale = np.max(np.abs(blocks), axis=(-1, -2), keepdims=True)
    if np.any(scale <= 0): raise ArithmeticError('zero estimator block')
    c = blocks/scale; x = c.mean(-1); y = c@contrasts(6)
    u, singular, _ = np.linalg.svd(y, full_matrices=False)
    if not np.isfinite(singular).all() or np.any(singular <= 0):
        raise ArithmeticError('singular calibration contrast block')
    coordinates = np.einsum('nsi,ns->ni', u, x)
    a = .5*np.sum((coordinates/singular)**2, axis=1)
    value = float(np.median(a)/F42_MEDIAN)
    if not np.isfinite(value) or value <= 0:
        raise ArithmeticError('invalid scale estimate')
    return value


def reference_bank(rng, train_blocks, cal_blocks, draws=4095, iterations=2):
    values = []
    for start in range(0, draws, 64):
        count = min(64, draws-start)
        h = shape_fit(rng.normal(size=(count, train_blocks, 4, 5)), iterations)
        eigen = np.linalg.eigvalsh(h)
        # A/kappa~F4,2; F4,2 median=(1+sqrt(2))/2.
        b = np.median(rng.f(4, 2, size=(count, cal_blocks)), axis=1)/F42_MEDIAN
        if not np.isfinite(eigen).all() or np.any(eigen <= 0) or not np.isfinite(b).all() or np.any(b <= 0):
            raise ArithmeticError('invalid reference shape or scale')
        q = np.sum(rng.chisquare(5, size=(count, 4))/eigen, axis=1)
        den2 = b*eigen[:, 0]*q/20
        if not np.isfinite(q).all() or np.any(q <= 0) or not np.isfinite(den2).all() or np.any(den2 <= 0):
            raise ArithmeticError('invalid reference denominator')
        w = rng.normal(size=count)/np.sqrt(den2)
        values.append(w)
    value = np.concatenate(values)
    if not np.isfinite(value).all():
        raise ArithmeticError('invalid reference, no draw silently discarded')
    return np.sort(value)


def rank_tail(statistic, reference):
    x = np.asarray(statistic)
    if not np.isfinite(x).all():
        raise ArithmeticError('nonfinite observed statistic')
    # >= ties conservative, +1 includes target. Exact discrete rank, not an
    # uncorrected Monte Carlo estimate of a tiny survival probability.
    p = (1+len(reference)-np.searchsorted(reference, x, side='left'))/(len(reference)+1)
    return np.where(x > 0, p, 1.)


def components(z, h, kappa_estimate, profiles, reference):
    g = len(z); result = np.zeros((g, 2, 2))
    scale = np.max(np.abs(z), axis=(-1, -2), keepdims=True)
    if np.any(scale <= 0):
        raise ArithmeticError('zero target block')
    z = z/scale
    y = z @ contrasts(6); means = z.mean(-1)
    q = np.einsum('gik,ij,gjk->g', y, np.linalg.inv(h), y)
    if np.any(q <= 0):
        raise ArithmeticError('invalid contrast energy')
    denominator = np.sqrt(kappa_estimate*q/20)
    for sign_index, sign in enumerate([1, -1]):
        marginal = rank_tail(sign*means/(denominator[:, None]*np.sqrt(np.diag(h))), reference)
        for subset in combinations(range(4), 3):
            ids = list(subset)
            profile = profiles[sign_index, ids]
            if not np.any(profile > 0): profile = np.ones(3)
            a = positive_direction(profile, h[np.ix_(ids, ids)])
            statistic = sign*(means[:, ids]@a)/(denominator*np.sqrt(a@h[np.ix_(ids, ids)]@a))
            result[:, sign_index, 0] = np.maximum(result[:, sign_index, 0], np.minimum(1, 3*marginal[:, ids].min(axis=1)))
            result[:, sign_index, 1] = np.maximum(result[:, sign_index, 1], rank_tail(statistic, reference))
    return result


def _evaluate(z, cal, seed, reference_draws, iterations, mismatch_bound):
    started = time.perf_counter()
    g = len(z); m = 2*g
    kappa = kappa_estimator(cal)
    ref = reference_bank(np.random.default_rng(seed), g//2, len(cal), reference_draws, iterations)
    reference_seconds = time.perf_counter()-started
    e = {key: np.empty((g, 2)) for key in ['PB_main', 'PB_ordinary', 'PB_grid', 'PB_grid_ordinary']}
    p = np.empty((g, 2, 2)); receipts = []
    for fold in range(4):
        held = np.arange(g)%4 == fold
        pilot = np.arange(g)%4 == (fold+1)%4
        learn = ~held & ~pilot
        h = shape_fit(z[learn]@contrasts(6), iterations)
        profiles, _, training_ok = learn_profiles(z[learn], h, kappa)
        # Pilot is an entirely separate information source. It does not use
        # calibration, references, LEARN data, or previously crossfitted scores.
        hp = shape_fit(z[pilot]@contrasts(6), iterations)
        kp = kappa_estimator(z[pilot])
        pp, gamma, pilot_ok = learn_profiles(z[pilot], hp, kp)
        pilot_scores = pc_components(z[pilot], hp, kp, 1., pp)
        actual = components(z[held], h, kappa*mismatch_bound, profiles, ref)
        p[held] = actual
        ec = []; pilots = []; grid_values = []; grid_receipts = []
        for c in range(2):
            value, receipt = calibrate(actual[:, :, c], pilot_scores[:, :, c], m, .05)
            ec.append(value); pilots.append(receipt)
            gv, gr = grid_calibrate(actual[:, :, c], receipt, m, reference_draws, c)
            grid_values.append(gv); grid_receipts.append(gr)
        e['PB_ordinary'][held] = ec[0]
        e['PB_main'][held] = (1-gamma)*ec[0]+gamma*ec[1]
        e['PB_grid_ordinary'][held] = grid_values[0]
        e['PB_grid'][held] = (1-gamma)*grid_values[0]+gamma*grid_values[1]
        receipts.append({'fold': fold, 'pilot_fold': (fold+1)%4, 'train_count': int(learn.sum()),
                         'pilot_count': int(pilot.sum()), 'shape': h, 'profiles': profiles,
                         'gamma_from_pilot': gamma, 'pilot_kappa': kp, 'pilots': pilots, 'grid_calibrators': grid_receipts,
                         'training_pattern_converged': training_ok, 'pilot_pattern_converged': pilot_ok})
    return {'status': 'completed', 'version': VERSION,
            'decisions': {k: ebh(v, .05) for k, v in e.items()}, 'evidence': e,
            'p': p, 'reference': ref, 'folds': receipts, 'kappa_estimate': kappa,
            'reference_seconds': reference_seconds, 'seconds': time.perf_counter()-started,
            'calibration_blocks': len(cal), 'target_shape_iterations': iterations, 'mismatch_bound': mismatch_bound,
            'numeric_scope': 'rank-count exact; float64 matrix calculations, not interval-certified'}


def evaluate(z, cal, *, seed, reference_draws=4095, iterations=2, mismatch_bound=1.):
    z = np.asarray(z, float); cal = np.asarray(cal, float)
    if z.ndim != 3 or z.shape[1:] != (4, 6) or len(z) < 16 or len(z)%4 or not np.isfinite(z).all():
        raise ValueError('finite Gx4x6, G>=16 divisible by4 required')
    if cal.ndim != 3 or cal.shape[1:] != (4, 6) or len(cal) < 4 or not np.isfinite(cal).all():
        raise ValueError('finite Nx4x6 centered independent calibration blocks required')
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) for x in [seed, reference_draws, iterations]):
        raise ValueError('integer seed, reference count and iteration count required')
    if seed < 0 or reference_draws < 1 or iterations < 0:
        raise ValueError('invalid randomized design')
    denominator=reference_draws+1
    if denominator & (denominator-1) or denominator>65536:
        raise ValueError('reference count plus one must be power of two <=65536')
    if not np.isfinite(mismatch_bound) or mismatch_bound < 1:
        raise ValueError('externally specified kappa ratio bound >=1 required')
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return _evaluate(z, cal, seed, reference_draws, iterations, mismatch_bound)
    except (ArithmeticError, np.linalg.LinAlgError, ValueError, RuntimeError) as err:
        if isinstance(err, ValueError) and str(err) not in [
            'finite contrast blocks required', 'zero contrast block outside model',
            'Positive definite study shape required',
            'means must be finite',
            'variance must be a finite positive vector with N entries',
            'shape must be a finite symmetric positive definite 4 x 4 matrix',
            'shape must be positive definite',
            'A finite nonzero nonnegative profile is required']:
            raise
        if isinstance(err, RuntimeError) and 'maximum number of iterations' not in str(err).lower():
            raise
        # Defined whole-family no-discovery outcome; retain reason. Not a
        # successful method evaluation, and never drop this family's metrics.
        return {'status': 'conservative_numerical_failure', 'error': repr(err), 'version': VERSION,
                'decisions': {k: np.zeros((len(z), 2), bool) for k in ['PB_main', 'PB_ordinary', 'PB_grid', 'PB_grid_ordinary']},
                'evidence': {k: np.zeros((len(z), 2)) for k in ['PB_main', 'PB_ordinary', 'PB_grid', 'PB_grid_ordinary']},
                'p': np.ones((len(z), 2, 2)), 'reference': np.array([]), 'folds': [],
                'kappa_estimate': None, 'reference_seconds': None, 'seconds': None}
