"""Fixed p-to-e decision-budget calibrators (established e-BH machinery).

Every component is decreasing and integrates to one on a uniform null.
Convex mixtures remain e-calibrators under arbitrary dependence, provided
the input p-values really are superuniform. No outcome-dependent tuning.
"""
import numpy as np
from .molecular_methods import checked_probabilities


def by_calibrator(p, family, alpha=.05, cap=None):
    """Step calibrator that reproduces BY through e-BH, including sparse tail."""
    p=checked_probabilities(p)
    if not isinstance(family,int) or family<1 or not 0<alpha<1:
        raise ValueError("Positive integer family and alpha in (0,1) required")
    cap=family if cap is None else cap
    if not isinstance(cap,int) or not 1<=cap<=family:
        raise ValueError("Integer rank cap in [1,family] required")
    harmonic=np.sum(1/np.arange(1,cap+1))
    index=np.maximum(1,np.ceil(p*family*harmonic/alpha))
    return np.where(index<=cap,family/(alpha*index),0.)


def focused_calibrator(p,family,threshold=.001,fraction=.5,alpha=.05,cap=None):
    """Mix BY-shaped evidence with I(p<=threshold)/threshold.

    Concentrating evidence around anticipated useful discovery counts can
    improve power, but may hurt sparse/weak alternatives. These are fixed
    development candidates; never choose the largest output using test data.
    """
    p=checked_probabilities(p)
    if not 0<threshold<1 or not 0<=fraction<=1:
        raise ValueError("Threshold in (0,1) and mixture fraction in [0,1]")
    return (1-fraction)*by_calibrator(p,family,alpha,cap)+fraction*(p<=threshold)/threshold
