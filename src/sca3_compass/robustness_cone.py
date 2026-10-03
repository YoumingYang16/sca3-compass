"""Classical chi-bar-square/F orthant tests in dimension three.

Known conditional elliptical shape: a Gaussian orthant-projection distance
has chi-bar-square distribution. Dividing by independent chi-square_df/df
gives the corresponding chi-bar-F mixture. Not an original asymptotic theorem.
"""
from itertools import combinations
import numpy as np
from scipy.stats import chi2, f

from .robustness_universal import orthant_distance


def orthant_probability3(covariance):
    scale=np.sqrt(np.diag(covariance))
    correlation=covariance/scale[:,None]/scale[None,:]
    rho=correlation[np.triu_indices(3,1)]
    return float(.125+np.arcsin(np.clip(rho,-1,1)).sum()/(4*np.pi))


def cone_weights3(shape):
    if np.shape(shape)!=(3,3) or np.linalg.eigvalsh(shape).min()<=0:
        raise ValueError("Positive definite 3D covariance required")
    w0=orthant_probability3(shape)
    w3=orthant_probability3(np.linalg.inv(shape))
    weights=np.array([w0,.5-w3,.5-w0,w3])
    if weights.min() < -1e-12 or not np.isclose(weights.sum(),1):
        raise ArithmeticError("Invalid chi-bar weights")
    return np.maximum(weights,0)


def cone_tail(distance,shape,df):
    q=np.asarray(distance,float)
    weights=cone_weights3(shape)
    p=np.zeros_like(q)
    for j in range(1,4):
        p+=weights[j]*(chi2.sf(q,j) if np.isinf(df) else f.sf(q/j,j,df))
    return np.where(q<=0,1.,p)


def cone_partial_conjunction(mean,variance,shapes,df):
    variance=np.broadcast_to(np.asarray(variance,float),(len(mean),))
    result=np.zeros(mean.shape[:2])
    for fold,shape in enumerate(shapes):
        held=np.arange(len(mean))%2==fold
        for subset in combinations(range(4),3):
            subshape=shape[np.ix_(subset,subset)]
            distance=orthant_distance(mean[held][:,:,list(subset)],subshape)/variance[held,None]
            result[held]=np.maximum(result[held],cone_tail(distance,subshape,df))
    return result
