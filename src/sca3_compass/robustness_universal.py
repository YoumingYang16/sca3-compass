"""Universal likelihood evidence for a composite directional replication null.

The numerator is a fixed proper density. The denominator maximizes the null
likelihood over all location vectors with fewer than r positive coordinates.
This is an established universal-inference inequality, not a novel theorem.
Known conditional scale/shape needed for exact validity; plug-in is empirical.
"""
from itertools import combinations
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp


def orthant_distance(x,covariance):
    """Squared Mahalanobis distance to the negative orthant, d<=4.

    Enumerate active zero-mean constraints. For active A, fitted inactive means
    are x_I-Sigma_IA Sigma_AA^-1 x_A; active multipliers must be nonnegative.
    """
    x=np.asarray(x,float)
    dimension=x.shape[-1]
    output=np.full(x.shape[:-1],np.inf)
    for size in range(dimension+1):
        for active in combinations(range(dimension),size):
            inactive=[j for j in range(dimension) if j not in active]
            if not active:
                valid=np.all(x<=0,axis=-1)
                distance=np.zeros(x.shape[:-1])
            else:
                xa=x[...,list(active)]
                inverse=np.linalg.inv(covariance[np.ix_(active,active)])
                multipliers=xa@inverse
                valid=np.all(multipliers>=-1e-12,axis=-1)
                distance=np.sum(xa*multipliers,axis=-1)
                if inactive:
                    fitted=x[...,inactive]-multipliers@covariance[np.ix_(active,inactive)]
                    valid&=np.all(fitted<=1e-12,axis=-1)
            output=np.minimum(output,np.where(valid,np.maximum(distance,0),np.inf))
    if not np.isfinite(output).all():
        raise ArithmeticError("Orthant projection has no feasible active set")
    return output


def pc_null_distance(x,shape,r=2):
    s=x.shape[-1]
    distances=[]
    for null_indices in combinations(range(s),s-r+1):
        distances.append(orthant_distance(x[...,list(null_indices)],shape[np.ix_(null_indices,null_indices)]))
    return np.min(distances,axis=0)


@lru_cache(maxsize=8)
def alternatives(s=4,effects=(2.5,3.5,4.5),size_weights=(.45,.1,.45)):
    locations=[]
    weights=[]
    for size,size_weight in zip(range(2,s+1),size_weights):
        subsets=list(combinations(range(s),size))
        for subset in subsets:
            for effect in effects:
                vector=np.zeros(s)
                vector[list(subset)]=effect
                locations.append(vector)
                weights.append(size_weight/(len(subsets)*len(effects)))
    weights=np.asarray(weights,float)
    if (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError("Positive normalized mixture weights required")
    return np.asarray(locations),weights


def universal_pc_e(mean,variance,shapes,df,effects=(2.5,3.5,4.5)):
    mean=np.asarray(mean,float)
    variance=np.broadcast_to(np.asarray(variance,float),(len(mean),))
    if mean.shape[1:]!=(2,4) or np.any(variance<=0):
        raise ValueError("G x 2 x 4 means and positive G variances required")
    locations,weights=alternatives(effects=tuple(effects))
    output=np.zeros(mean.shape[:2])
    for fold,shape in enumerate(shapes):
        held=np.arange(len(mean))%2==fold
        x=mean[held]
        v=variance[held]
        null_distance=pc_null_distance(x,shape)/v[:,None]
        difference=x[:,:,None,:]-locations[None,None,:,:]
        inverse=np.linalg.inv(shape)
        alt_distance=np.einsum("gami,ij,gamj->gam",difference,inverse,difference)/v[:,None,None]
        if np.isinf(df):
            ratios=.5*(null_distance[:,:,None]-alt_distance)
        else:
            ratios=(df+4)/2*(np.log1p(null_distance[:,:,None]/df)-np.log1p(alt_distance/df))
        log_e=logsumexp(ratios+np.log(weights)[None,None,:],axis=-1)
        output[held]=np.exp(np.minimum(log_e,np.log(1e290)))
    return output
