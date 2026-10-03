"""Conditional directional likelihood evidence and classical e-PC aggregation.

For a KNOWN symmetric decreasing conditional location density, a positive-
halfline-truncated alternative likelihood ratio has expectation one at zero
and <=one under negative shifts. This is a likelihood-ratio e-value, not new
e-BH or a new unconditional plug-in guarantee. See method ledger.
"""
import numpy as np
from scipy.special import log_ndtr, logsumexp
from scipy.stats import t


def directional_likelihood_e(mean,scale,df,effects=(2.5,3.5,4.5)):
    mean=np.asarray(mean,float)
    scale=np.asarray(scale,float)
    df=np.asarray(df,float)
    effects=np.asarray(effects,float)
    if (not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale<=0)
            or np.isnan(df).any() or np.any(df<=0) or effects.ndim!=1
            or not effects.size or not np.isfinite(effects).all() or np.any(effects<=0)):
        raise ValueError("Finite means, positive scales, df, fixed positive alternatives needed")
    statistic=mean/scale
    alternative=effects/scale[...,None]
    if np.isinf(df).all():
        log_ratio=alternative*statistic[...,None]-.5*alternative**2-log_ndtr(alternative)
    else:
        expanded_df=np.broadcast_to(df,mean.shape)[...,None]
        log_ratio=t.logpdf(statistic[...,None]-alternative,expanded_df)-t.logpdf(statistic[...,None],expanded_df)-t.logcdf(alternative,expanded_df)
    log_e=logsumexp(log_ratio,axis=-1)-np.log(len(effects))
    # Capping an e-value downward cannot increase its expectation. Exponential
    # range cap is numeric protection, NOT outlier removal from the data.
    result=np.exp(np.minimum(log_e,np.log(1e290)))
    return np.where(mean>=0,result,0.)


def partial_conjunction_e(study_e,r=2):
    e=np.asarray(study_e,float)
    if not np.isfinite(e).all() or np.any(e<0) or not 1<=r<=e.shape[-1]:
        raise ValueError("Finite nonnegative e-values and valid replication count required")
    return np.sort(e,axis=-1)[...,:e.shape[-1]-r+1].mean(axis=-1)
