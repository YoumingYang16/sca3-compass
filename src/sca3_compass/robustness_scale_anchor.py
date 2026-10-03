"""Training-only information-criterion anchor for an added scale parameter.

Both fits use the SAME finite location dictionary and .02 log-weight penalty.
This is a regularized model-selection HEURISTIC, not ordinary maximum-
likelihood BIC, a Bayes factor, a significance test, or a fitted FDR theorem.
It cannot repair a misspecified alternative distribution.
"""
import math
import numpy as np
from scipy.special import logsumexp
from .robustness_patterns import fit_pattern_mixture,pattern_logpdf


def anchor_joint_scale(means,base_variance,shape,df,joint):
    if not joint['diagnostics']['converged']:
        raise ValueError('Only a converged joint fit may be anchored')
    variance=np.asarray(base_variance,float)
    n=len(variance)
    if n<2:
        raise ValueError('At least two training rows required')
    fixed=fit_pattern_mixture(means,variance,shape,df,pseudocount=.02)
    weights=np.asarray(joint['weights'],float)
    log_density=pattern_logpdf(means,variance*joint['tau'],shape,df)
    if (weights.shape!=(log_density.shape[1],) or not np.isfinite(weights).all()
            or np.any(weights<=0) or not np.isclose(weights.sum(),1,rtol=1e-12,atol=1e-12)):
        raise ValueError('Joint weights must match the finite location dictionary')
    terms=np.r_[logsumexp(log_density+np.log(weights),axis=1),.02*np.log(weights)]
    free_objective=math.fsum(terms.tolist())
    rounding=64*np.finfo(float).eps*max(1.,float(np.abs(terms).sum()))
    if abs(free_objective-joint['diagnostics']['objective'])>rounding:
        raise ValueError('Joint objective does not match supplied training data and nuisance inputs')
    fixed_objective=fixed['diagnostics']['objective']
    improvement=2*(free_objective-fixed_objective)
    threshold=float(np.log(n))
    converged=bool(fixed['diagnostics']['converged'])
    select_free=bool(not converged or improvement>threshold)
    tau=float(joint['tau']) if select_free else 1.
    return tau,{'converged':converged,'selected_free_scale':select_free,
        'selected_tau':tau,'free_tau':float(joint['tau']),'fixed_tau':1.,
        'free_objective_recomputed':free_objective,'fixed_objective':fixed_objective,
        'twice_penalized_objective_gain':improvement,'threshold':threshold,
        'criterion':'2*(penalized_objective_free-penalized_objective_fixed)>log(training_genes)',
        'fixed_fit':fixed['diagnostics'],'fallback_when_fixed_fit_fails':'retain converged free fit; record failure',
        'free_fit_worse_than_fixed':bool(improvement < -2*rounding),
        'uses_held_genes':False,'uses_truth_labels':False,
        'interpretation':'regularized information-criterion heuristic; not ordinary BIC or a p-value'}
