"""Independent Monte Carlo comparison banks, classical monotone tests.

No test outcomes enter banks or projection selection. Adaptive scan is itself
calibrated as a maximum, never an unadjusted minimum of selected p-values.
"""
from itertools import combinations
import time

import numpy as np

from .molecular_envelope import correlation_envelope, envelope_pvalues, rank_tail
from .molecular_methods import covariance_root, equicorrelation


def projection_weights(shape, subset):
    weights=[]
    for size in range(1,len(subset)+1):
        for selected in combinations(subset,size):
            w=np.zeros(len(shape))
            w[list(selected)]=1
            weights.append(w/np.sqrt(w@shape@w))
    return np.asarray(weights).T


def conditional_scan(statistic, shapes, df, rng, draws=131071):
    g,signs,s=statistic.shape
    output={name:np.zeros((g,signs)) for name in ["scan","max","sum"]}
    for fold,shape in enumerate(shapes):
        held=np.arange(g)%2==fold
        bank=rng.normal(size=(draws,s))@covariance_root(shape).T
        if np.isfinite(df):
            bank/=np.sqrt(rng.chisquare(df,size=(draws,1))/df)
        for subset in combinations(range(s),s-1):
            w=projection_weights(shape,subset)
            bank_projected=bank@w
            observed=statistic[held]@w
            for name,ref,obs in [
                ("scan",bank_projected.max(axis=-1),observed.max(axis=-1)),
                ("max",bank_projected[:,:len(subset)].max(axis=-1),observed[:,:,:len(subset)].max(axis=-1)),
                ("sum",bank_projected[:,-1],observed[:,:,-1])]:
                output[name][held]=np.maximum(output[name][held],rank_tail(ref,obs))
    return output


def additional_comparators(z,calibration,diagnostics,rng,draws=131071):
    started=time.perf_counter()
    g,s,k=z.shape
    signed=np.stack((z,-z),axis=1)
    fit=diagnostics["fit"]
    p={}
    original=np.empty((g,2,s))
    empirical=np.empty_like(original)
    fallback=[]
    for study in range(s):
        env=correlation_envelope(calibration[study],.005/s)
        original[:,:,study]=envelope_pvalues(signed[:,:,study],env,draws,rng)
        empirical[:,:,study]=rank_tail(calibration[study].max(axis=1),signed[:,:,study].max(axis=-1))
        fallback.append(env.fallback)
    p["gaussian_CE_original"]=original
    p["rank_max_simple"]=empirical
    fitted_pipeline=(1-fit["rho"])*np.eye(k)+fit["rho"]*np.ones((k,k))
    bank=rng.normal(size=(draws,k))@covariance_root(fitted_pipeline).T
    bank*=np.sqrt(fit["scatter"])
    if not fit["gaussian_bic_selected"]:
        bank/=np.sqrt(rng.chisquare(fit["df"],size=(draws,1))/fit["df"])
    p["student_maxT_repaired"]=rank_tail(bank.max(axis=-1),signed.max(axis=-1))
    shapes=[np.asarray(d["matrix"]) for d in diagnostics["shape"]]
    # Full residuals are signal-annihilating; recompute without fitting again.
    from .robustness_methods import contrasts
    y=z@contrasts(k)
    q=np.empty(g)
    for fold,shape in enumerate(shapes):
        held=np.arange(g)%2==fold
        q[held]=np.einsum("gsk,st,gtk->g",y[held],np.linalg.inv(shape),y[held])/(1-fit["rho"])
    d=s*(k-1)
    variance=(fit["scatter"]*np.ones(g) if fit["gaussian_bic_selected"] else (fit["df"]*fit["scatter"]+q)/(fit["df"]+d))
    df=np.inf if fit["gaussian_bic_selected"] else fit["df"]+d
    v=(1+(k-1)*fit["rho"])/k
    statistic=signed.mean(-1)/np.sqrt(v*variance[:,None,None])
    calibrated=conditional_scan(statistic,shapes,df,rng,draws)
    for name,values in calibrated.items():
        p[f"conditional_t_{name}_PC"]=values
    return p,{"elapsed_seconds":time.perf_counter()-started,"draws":draws,
              "original_fallback_fraction":float(np.mean(fallback)),"min_p":1/(draws+1)}
