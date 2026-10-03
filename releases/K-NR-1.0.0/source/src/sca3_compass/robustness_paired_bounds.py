"""Fixed-n independent-family empirical Bernstein intervals.

Maurer & Pontil (2009), Theorem 11, https://arxiv.org/abs/0907.3740 .
Independent variables need NOT be identically distributed. For D in [-1,1],
apply the theorem to (D+1)/2 and to (1-D)/2. No optional stopping, no baseline
selection; the compared methods must be frozen before these observations.
"""
import numpy as np


def paired_interval(arrays, delta_side):
    """Equal-scene estimand; equal replication counts are mandatory.

    A whole simulated family supplies ONE difference. Pooling equal-sized
    fixed scenes preserves their equal weights. The pooled sample variance
    includes between-scene mean variation (conservative), not just MC noise.
    Each endpoint has failure probability <= delta_side. Floating-point
    padding is conservative engineering, not certified interval arithmetic.
    """
    xs = [np.asarray(x, float) for x in arrays]
    if not xs or not 0 < delta_side < 1:
        raise ValueError('Nonempty fixed scenes and a valid endpoint error required')
    if len({len(x) for x in xs}) != 1:
        raise ValueError('Unequal counts would change the equal-scene estimand')
    for x in xs:
        if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all() or np.any(np.abs(x) > 1):
            raise ValueError('Bounded independent whole-family differences required')
    x = np.concatenate(xs)
    log = np.log(2 / delta_side)
    variance = float(x.var(ddof=1))
    radius = float(np.sqrt(2 * variance * log / len(x)) + 14 * log / (3 * (len(x)-1)))
    mean = float(x.mean())
    return {'mean_difference': mean, 'lower': max(-1., mean-radius-2e-14),
            'upper': min(1., mean+radius+2e-14), 'radius': radius,
            'pooled_sample_variance': variance, 'n_independent_families': len(x),
            'n_scenes': len(xs), 'delta_each_endpoint': delta_side,
            'paired_mcse': float(np.sqrt(sum(v.var(ddof=1)/len(v) for v in xs))/len(xs)),
            'between_scene_mean_sd': float(np.std([v.mean() for v in xs])),
            'method': 'Maurer-Pontil2009 Theorem11, range2; fixed equal-weight scenes',
            'warning': 'MCSE is not this simultaneous interval; no gene-level replication or optional stopping'}
