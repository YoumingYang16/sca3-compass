"""Own-contrast covariate weighting, not across-gene outcome normalization.

For known common-location inverse-gamma radial law, U=F_d,nu(Q/(d*sigma²))
is uniform and the PC p-value is conditionally valid given Q. A fixed positive
weight of mean one preserves marginal p validity: E[min(t W,1)] <= t E[W].
The fitted-parameter implementation has no automatic finite-sample guarantee.
"""
import numpy as np
from scipy.stats import beta, f, t


def radial_weight(q,dimension,prior_df,prior_scale,a,b,floor=.1):
    if a<=0 or b<=0 or not 0<=floor<1 or prior_df<=0 or prior_scale<=0:
        raise ValueError("Positive parameters and floor in [0,1) required")
    u=f.cdf(np.asarray(q)/(dimension*prior_scale),dimension,prior_df)
    # This screen uses a,b>=1, hence no endpoint blow-up.
    weight=floor+(1-floor)*beta.pdf(u,a,b)
    if not np.isfinite(weight).all() or np.any(weight<=0):
        raise FloatingPointError("Invalid own-covariate weights")
    return weight


def power_weight_table(dimension,prior_df,prior_scale,projection_variance,
                       reference_level=.0003,bins=64,effects=(2.5,3.5,4.5)):
    """Model-predicted allocation, independent of observed study means.

    Uniform-U bins have equal known-model probabilities. Maximize a marginal
    shifted-Student power proxy on a fixed weight grid under mean weight one.
    A deterministic convex interpolation satisfies the budget exactly; it is
    NOT asserted to attain the global optimum of the nonconcave true PC power.
    """
    u=(np.arange(bins)+.5)/bins
    q=dimension*prior_scale*f.ppf(u,dimension,prior_df)
    scale=np.sqrt(projection_variance*(prior_df*prior_scale+q)/(prior_df+dimension))
    weight_grid=np.array([.05,.1,.2,.4,.7,1.,1.4,2.,3.,4.,6.,10.,20.])
    cutoff=t.isf(reference_level*weight_grid,prior_df+dimension)
    prediction=t.sf(cutoff[None,:,None]-np.asarray(effects)[None,None,:]/scale[:,None,None],prior_df+dimension).mean(axis=-1)
    def allocation(lagrange):
        return weight_grid[np.argmax(prediction-lagrange*weight_grid[None,:],axis=1)]
    low,high=0.,1.
    for _ in range(55):
        middle=(low+high)/2
        if allocation(middle).mean()>1:
            low=middle
        else:
            high=middle
    overspend,underspend=allocation(low),allocation(high)
    gap=overspend.mean()-underspend.mean()
    fraction=0 if gap<=1e-14 else float(np.clip((1-underspend.mean())/gap,0,1))
    weights=fraction*overspend+(1-fraction)*underspend
    weights/=weights.mean()
    return weights


def power_radial_weight(q,dimension,prior_df,prior_scale,projection_variance):
    table=power_weight_table(dimension,prior_df,prior_scale,projection_variance)
    u=f.cdf(np.asarray(q)/(dimension*prior_scale),dimension,prior_df)
    index=np.minimum((u*len(table)).astype(int),len(table)-1)
    return table[index],{"table":table.tolist(),"mean_under_working_null":float(table.mean()),
        "bins":len(table),"reference_level":.0003,"effect_proxy":[2.5,3.5,4.5],
        "uses_observed_means":False,"uses_truth_labels":False}
