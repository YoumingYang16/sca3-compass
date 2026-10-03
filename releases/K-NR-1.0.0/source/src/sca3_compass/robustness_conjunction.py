"""Classical Hunter union bound / closed partial conjunction, not new theory.

Known multivariate Student shape and df, one-sided common thresholds. Estimated
shape/df calibration is a separate issue; this module does not certify it.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.special import ndtr, ndtri, stdtr, stdtrit
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=8)
def quadrature(nodes):
    x,w=leggauss(nodes)
    return (x+1)/2,w/2


def diagonal_joint_tail(threshold, rho, df, nodes=48):
    """P(X>h,Y>h) by integrating over the smaller coordinate.

    2 int_h^infinity f(x) P(Y>x|X=x) dx, evaluated after a tail-probability
    change of variable. Smooth Gaussian/Student integrals; numerical accuracy
    must be tested independently, not assumed a rigorous quadrature bound.
    """
    h=np.asarray(threshold,float)
    if not -1<rho<1 or df<=0:
        raise ValueError("Interior correlation and positive df required")
    u,w=quadrature(nodes)
    if np.isinf(df):
        marginal=ndtr(-h)
        x=-ndtri(np.maximum(marginal[...,None]*u,np.finfo(float).tiny))
        conditional=ndtr(-x*np.sqrt((1-rho)/(1+rho)))
    else:
        marginal=stdtr(df,-h)
        x=-stdtrit(df,np.maximum(marginal[...,None]*u,np.finfo(float).tiny))
        conditional=stdtr(df+1,-x*np.sqrt((df+1)*(1-rho)/(1+rho)/(df+x*x)))
    return 2*marginal*np.sum(conditional*w,axis=-1)


def hunter_partial_conjunction(statistic, shapes, df, r=2, nodes=48):
    """Max over all (S-r+1)-subset intersection-null union-tail p-values.

    S=4,r=2 => triples; any two edges form a spanning tree. Choose the largest
    two correlations (joint equal-threshold tail is monotone in correlation).
    Means in a true intersection are componentwise <=0. No null-effect labels
    or case distribution identifiers enter the method.
    """
    z=np.asarray(statistic,float)
    g,signs,s=z.shape
    if s!=4 or r!=2 or signs!=2:
        raise ValueError("This audited implementation covers S=4,r=2,two signs")
    result=np.zeros((g,2))
    for fold,shape in enumerate(shapes):
        held=np.arange(g)%2==fold
        current=np.zeros((held.sum(),2))
        for subset in combinations(range(s),s-r+1):
            threshold=z[held][:,:,subset].max(axis=-1)
            edges=sorted([float(shape[i,j]) for i,j in combinations(subset,2)],reverse=True)[:2]
            marginal=ndtr(-threshold) if np.isinf(df) else stdtr(df,-threshold)
            union=3*marginal
            for rho in edges:
                union-=diagonal_joint_tail(threshold,rho,df,nodes)
            # Nonpositive thresholds are irrelevant to small-p discovery; 1
            # avoids relying on quadrature of near-unit tails in this branch.
            union=np.where(threshold>0,np.clip(union,0,1),1.)
            current=np.maximum(current,union)
        result[held]=current
    return result
