"""Positive learned-direction tests for directional partial conjunction.

For a FIXED nonnegative vector a, a'M/sqrt(a'Sigma a) has nonpositive null
location whenever all component means are nonpositive. For r=2 of S=4,
take the maximum of the four three-study intersection p-values. Known
Gaussian/conditional-Student model gives validity regardless of whether
the alternative profile is correctly learned. Training/test independence
is required for data-learned directions; fitted nuisance/gene dependence
are not automatically covered. This is a classical directional test.
"""
from itertools import combinations
import numpy as np
from scipy.optimize import nnls
from scipy.stats import norm,t
from .robustness_simes import SIMES_POLICY, simes_eligibility, simes_intersection


def positive_direction(profile,shape):
    """Constrained optimal linear direction for a proposed positive profile."""
    a=np.asarray(profile,float)
    shape=np.asarray(shape,float)
    if a.ndim!=1 or np.any(a<0) or not np.isfinite(a).all() or not np.any(a>0):
        raise ValueError('A finite nonzero nonnegative profile is required')
    if shape.shape!=(len(a),len(a)) or not np.allclose(shape,shape.T):
        raise ValueError('Symmetric matching study shape required')
    eigen,vec=np.linalg.eigh(shape)
    if eigen.min()<=0:
        raise ValueError('Positive definite study shape required')
    root=(vec*np.sqrt(eigen))@vec.T
    inverse_root=(vec/np.sqrt(eigen))@vec.T
    weight,_=nnls(root,inverse_root@a)
    weight/=np.sqrt(weight@shape@weight)
    return weight


def projection_pc(means,variance,shape,df,profiles,*,bonferroni=False,support_simes=False,diagnostics=None):
    """Single-fold Gx2 PC; profiles 2x4 are fixed from OTHER genes.

    Bonferroni comparator allocates component level proportional to the
    profile itself. Zero profile entries receive zero testing weight. A
    profile with no positive coordinate in a subset uses uniform weights.
    Support Simes checks only the retained subshape and uses predeclared
    Bonferroni if any retained correlation is negative. Optional diagnostics
    exports every sign/subset decision for the caller's fold receipt.
    """
    m=np.asarray(means,float)
    v=np.broadcast_to(np.asarray(variance,float),(len(m),))
    a=np.asarray(profiles,float)
    shape=np.asarray(shape,float)
    if m.ndim!=3 or m.shape[1:]!=(2,4) or a.shape!=(2,4):
        raise ValueError('Gx2x4 signed means and 2x4 profiles required')
    if not np.isfinite(m).all() or not np.isfinite(v).all() or np.any(v<=0) or np.any(a<0) or not np.isfinite(a).all():
        raise ValueError('Finite values and positive variances required')
    if shape.shape!=(4,4):
        raise ValueError('Matching four-study shape required')
    simes_eligibility(shape)
    output=np.zeros((len(m),2))
    decisions=[]
    for sign in range(2):
        for indices in combinations(range(4),3):
            subprofile=a[sign,list(indices)]
            if not np.any(subprofile>0):
                subprofile=np.ones(3)
            subshape=shape[np.ix_(indices,indices)]
            subset=m[:,sign,list(indices)]
            if support_simes:
                keep=subprofile>0
                statistic=subset[:,keep]/np.sqrt(v[:,None]*np.diag(subshape)[keep])
                marginal=norm.sf(statistic) if np.isinf(df) else t.sf(statistic,df)
                marginal=np.where(statistic>0,marginal,1.)
                retained=np.flatnonzero(keep)
                decision={}
                value=simes_intersection(marginal,subshape[np.ix_(retained,retained)],diagnostics=decision)
                decisions.append({'sign':sign,'intersection':list(indices),
                    'retained_studies':np.asarray(indices)[keep].tolist(),**decision})
            elif bonferroni:
                allocation=subprofile/subprofile.sum()
                statistic=subset/np.sqrt(v[:,None]*np.diag(subshape))
                marginal=norm.sf(statistic) if np.isinf(df) else t.sf(statistic,df)
                marginal=np.where(statistic>0,marginal,1.)
                ratio=np.full_like(marginal,np.inf)
                np.divide(marginal,allocation,out=ratio,where=allocation>0)
                value=np.minimum(1,np.min(ratio,axis=1))
            else:
                weights=positive_direction(subprofile,subshape)
                statistic=(subset@weights)/np.sqrt(v)
                value=norm.sf(statistic) if np.isinf(df) else t.sf(statistic,df)
                value=np.where(statistic>0,value,1.)
            output[:,sign]=np.maximum(output[:,sign],value)
    if diagnostics is not None and support_simes:
        fallback_count=sum(not item['simes_eligible'] for item in decisions)
        diagnostics.update({'policy':SIMES_POLICY,'intersections':decisions,
            'intersection_count':len(decisions),'fallback_intersection_count':fallback_count,
            'fallback_intersection_fraction':fallback_count/len(decisions),
            'uses_pvalues_for_selection':False})
    return output
