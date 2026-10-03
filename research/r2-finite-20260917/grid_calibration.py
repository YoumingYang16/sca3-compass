"""Discrete-support p-to-e normalization, NOT a new general calibration theorem.

If superuniform p has support t_k, the least upper bound for decreasing h
is sum_k (t_k-t_(k-1))*h(t_k). PILOT selection must be independent as proved
in PREDICTIVE_BRIDGE.md. The normalizer is rounded upwards from an exact
rational sum of represented h values; individual quotients are rounded down.
This narrow certificate does not certify upstream matrix/PRNG/gamma arithmetic.
"""
from fractions import Fraction
import numpy as np
from finite_calibration import focused_calibrator


def grid_calibrate(p, receipt, family, draws, component):
    if isinstance(draws,(bool,np.bool_)) or not isinstance(draws,(int,np.integer)) or component not in [0, 1] or draws < 1:
        raise ValueError('ordinary0/projection1 and positive draws required')
    denominator = draws+1
    if denominator & (denominator-1) or denominator>65536:
        raise ValueError('certified support interface requires power-of-two denominator <=65536')
    step = 3 if component == 0 else 1
    p=np.asarray(p,float)
    rank=p*denominator
    if not np.isfinite(p).all() or np.any((p<=0)|(p>1)) or np.any(rank!=np.floor(rank)):
        raise ValueError('p must belong exactly to the declared rank grid')
    if component==0 and np.any((rank<denominator)&(rank%3!=0)):
        raise ValueError('ordinary PC p must belong to three-rank grid or equal1')
    numerators = np.r_[np.arange(step, denominator, step), denominator]
    weights = np.diff(np.r_[0, numerators])
    support = numerators/denominator
    h = focused_calibrator(support, family, receipt['threshold'],
                           .8 if receipt['count'] else 0., .05, min(64, family))
    if not np.isfinite(h).all() or np.any(h<0) or np.any(np.diff(h)>0):
        raise ArithmeticError('calibrator table must be finite nonnegative decreasing')
    # At most65 distinct step heights, not thousands of Fraction operations.
    values, inverse = np.unique(h, return_inverse=True)
    mass = np.bincount(inverse, weights=weights).astype(np.int64)
    if int(mass.sum())!=denominator: raise ArithmeticError('incorrect grid mass')
    exact = sum((Fraction.from_float(float(v))*int(w) for v,w in zip(values, mass)), Fraction(0))/denominator
    if exact == 0:
        return np.zeros_like(p), {'normalizer': 0., 'rank_step': step, 'empty_support': True}
    normalizer = float(exact)
    if Fraction.from_float(normalizer) < exact:
        normalizer = np.nextafter(normalizer, np.inf)
    quotients=[]
    for v in values:
        ratio=Fraction.from_float(float(v))/Fraction.from_float(normalizer)
        q=float(ratio)
        if Fraction.from_float(q)>ratio: q=np.nextafter(q,0.)
        quotients.append(q)
    table=np.asarray(quotients)[inverse]
    location=(rank.astype(np.int64)+step-1)//step-1
    value=table[location]
    return value, {'normalizer': normalizer, 'rank_step': step, 'empty_support': False}
