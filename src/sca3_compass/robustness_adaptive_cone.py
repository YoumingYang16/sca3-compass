"""Active-face adaptive cone test: Al Mohamad et al. (2020), Biometrika.

Known Gaussian covariance: exact level using (1-w0)*chi-square survival of
the projection with its active-face dimension. Student version is an explicit
EXPERIMENTAL extension, not covered by the paper's independent variance
estimator theorem for this location-mixture conditioning construction.
"""
from itertools import combinations
import numpy as np
from scipy.stats import chi2, f
from .robustness_cone import orthant_probability3


def projection_face(x,covariance):
    x=np.asarray(x,float)
    covariance=np.asarray(covariance,float)
    d=x.shape[-1]
    if covariance.shape!=(d,d) or not np.isfinite(x).all() or np.linalg.eigvalsh(covariance).min()<=0:
        raise ValueError("Finite vector and positive definite covariance required")
    distance=np.full(x.shape[:-1],np.inf)
    rank=np.full(x.shape[:-1],-1,int)
    for size in range(d+1):
        for active in combinations(range(d),size):
            inactive=[j for j in range(d) if j not in active]
            if size==0:
                valid=np.all(x<=0,axis=-1)
                q=np.zeros(x.shape[:-1])
            else:
                xa=x[...,list(active)]
                mult=xa@np.linalg.inv(covariance[np.ix_(active,active)])
                valid=np.all(mult>=-1e-12,axis=-1)
                q=np.maximum(0,np.sum(xa*mult,axis=-1))
                if inactive:
                    fitted=x[...,inactive]-mult@covariance[np.ix_(active,inactive)]
                    valid &= np.all(fitted<=1e-12,axis=-1)
            # In exact face-boundary ties retain the smaller active rank.
            update=valid & (q<distance-1e-12)
            distance=np.where(update,q,distance)
            rank=np.where(update,size,rank)
    if not np.isfinite(distance).all() or np.any(rank<0):
        raise ArithmeticError("No feasible cone face")
    return distance,rank


def adaptive_cone_pc(mean,variance,shapes,df):
    if mean.shape[1:]!=(2,4):
        raise ValueError("G x 2 x 4 means required")
    variance=np.broadcast_to(np.asarray(variance,float),(len(mean),))
    if np.any(variance<=0):
        raise ValueError("Positive variances required")
    result=np.zeros(mean.shape[:2])
    for fold,shape in enumerate(shapes):
        held=np.arange(len(mean))%2==fold
        for subset in combinations(range(4),3):
            sigma=shape[np.ix_(subset,subset)]
            q,rank=projection_face(mean[held][:,:,list(subset)],sigma)
            q=q/variance[held,None]
            lam=1-orthant_probability3(sigma)
            p=np.ones_like(q)
            for j in range(1,4):
                tail=chi2.sf(q,j) if np.isinf(df) else f.sf(q/j,j,df)
                p=np.where(rank==j,lam*tail,p)
            result[held]=np.maximum(result[held],p)
    return result
