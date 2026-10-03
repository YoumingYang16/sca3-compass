"""Fixed-n bounded independent FAMILY confidence bounds, not gene-level CIs.

Envelope lower bound: Jensen concavity + McDiarmid; no baseline union factor.
Upper bound: independent-DEVELOPMENT selected baseline + weighted Hoeffding.
FDP: inverse binary-KL Chernoff bound for independent [0,1] variables, not a
binomial assumption about dependent gene discoveries within a family.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import xlogy,xlog1py

def binary_kl(p,q):
    if not 0<=p<=1 or not 0<=q<=1:raise ValueError('Unit interval required')
    if q==0:return 0. if p==0 else np.inf
    if q==1:return 0. if p==1 else np.inf
    return float(xlogy(p,p/q)+xlogy(1-p,(1-p)/(1-q)))

def kl_interval(mean,n,delta_side):
    """Each endpoint has failure probability <=delta_side at FIXED n."""
    if not 0<=mean<=1 or n<1 or not 0<delta_side<1:raise ValueError('Invalid inputs')
    target=np.log(1/delta_side)/n;eps=np.finfo(float).tiny
    top=1-np.finfo(float).eps
    lower=0. if mean==0 or binary_kl(mean,eps)<target else brentq(lambda q:binary_kl(mean,q)-target,eps,mean,xtol=1e-15)
    upper=1. if mean==1 or binary_kl(mean,top)<target else brentq(lambda q:binary_kl(mean,q)-target,mean,top,xtol=1e-15)
    return float(max(0,lower-2e-14)),float(min(1,upper+2e-14))

def envelope_interval(differences,dev_indices,delta_lower,delta_upper,weights=None):
    """One R_j x B_j matrix per scene. b_j is fixed using independent DEV.

    Estimand=sum_j w_j min_b E D_jb. Replacing one replicate changes sample
    envelope at most2w_j/n_j; sum squares=4sum(w_j²/n_j).
    E[sample envelope]<=estimand. Hence lower=sample−sqrt(2log(1/delta)sum w²/n).
    Upper uses the DEVELOPMENT-selected comparator, whose true difference
    bounds the oracle-baseline difference from above, with the same radius.
    """
    xs=[np.asarray(x,float) for x in differences];j=len(xs)
    if not j or len(dev_indices)!=j:raise ValueError('Matching scenes/indices needed')
    w=np.ones(j)/j if weights is None else np.asarray(weights,float)
    if w.shape!=(j,) or np.any(w<0) or not np.isclose(w.sum(),1):raise ValueError('Fixed probability weights needed')
    if not 0<delta_lower<1 or not 0<delta_upper<1:raise ValueError('Valid tail error needed')
    for x,b in zip(xs,dev_indices):
        if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all() or np.any(np.abs(x)>1+1e-14) or not 0<=b<x.shape[1]:raise ValueError('Bounded whole-family difference matrices required')
    means=[x.mean(0) for x in xs];sample=float(sum(a*m.min() for a,m in zip(w,means)))
    fixed=float(sum(a*m[b] for a,m,b in zip(w,means,dev_indices)))
    mass=float(sum(a*a/len(x) for a,x in zip(w,xs)))
    lower=max(-1.,sample-np.sqrt(2*np.log(1/delta_lower)*mass))
    upper=min(1.,fixed+np.sqrt(2*np.log(1/delta_upper)*mass))
    se=float(np.sqrt(sum(a*a*np.var(x[:,b],ddof=1)/len(x) for a,x,b in zip(w,xs,dev_indices))))
    return {'sample_envelope_difference':sample,'fixed_development_comparator_difference':fixed,
        'lower':float(lower),'upper':float(upper),'delta_lower':delta_lower,'delta_upper':delta_upper,
        'fixed_comparator_mcse':se,'sum_weight_squared_over_n':mass,
        'method':'Jensen-McDiarmid lower; DEV-selected comparator weighted-Hoeffding upper',
        'warning':'Conservative finite-n bounds; sample envelope biased downward, not an unbiased effect estimate; no optional stopping'}
