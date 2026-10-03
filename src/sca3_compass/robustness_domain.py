"""Development radial-invariant target/calibration scale mismatch checks.

For known separable shapes and common pipeline location, M/sqrt(Q/d) has
null scale sqrt(v_target*(1-rho_cal)/(1-rho_target)) times t_d, whatever the
positive common radial law. A shifted absolute normal is stochastically
larger than its centered version, also after dividing by its independent
positive denominator. Arbitrary effects therefore inflate absolute-score
quantiles rather than being silently removed as outliers.

Only one prespecified study per training gene is used for the binomial
order-statistic bound; dependent studies are not counted as independent
samples. Fitted training shapes and gene dependence invalidate an automatic
finite-sample bound. The full deployed method is still empirical.
"""
import numpy as np
from scipy.stats import binom, t, f
from scipy.optimize import minimize_scalar


def scale_from_absolute_scores(scores, dimension, probability=.25, error=.005):
    """Point and one-sided upper null VARIANCE scale estimates.

    U = X_(k)/q_p, k=1+BinomPPF(1-error;n,p); known-shape independent
    scores give Pr(U < null_sd)<=error. If k>n no finite bound exists.
    Point estimate is not a confidence bound. No label or trimmed target.
    """
    x=np.sort(np.asarray(scores,float))
    if x.ndim!=1 or len(x)<8 or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('At least eight finite nonnegative scores required')
    if dimension<=0 or not 0<probability<1 or not 0<error<1:
        raise ValueError('Positive dimension and probabilities in (0,1) required')
    cutoff=float(t.ppf((1+probability)/2,dimension))
    rank=int(binom.ppf(1-error,len(x),probability))+1
    point=float(np.quantile(x,probability,method='inverted_cdf')/cutoff)**2
    upper=np.inf if rank>len(x) else float(x[rank-1]/cutoff)**2
    return point,upper,{'training_gene_count':len(x),'probability':probability,
        'error_probability':error,'upper_order_statistic_rank':rank,
        'absolute_t_cutoff':cutoff,'point_variance':point,
        'upper_variance':None if np.isinf(upper) else upper,
        'bound_requires_independent_genes_and_fixed_true_shape':True}


def central_f_scale(ratios,numerator_df,denominator_df):
    """Selection-corrected central F scale MLE, an empirical-null candidate.

    Fits the lower 40% using log f(x/scale)-log scale-log F(cutoff/scale).
    The zero assumption (negligible nonnull contribution in this region)
    is additional and not guaranteed by stochastic ordering. All held genes
    remain tested: this is nuisance fitting, not deletion of extreme data.
    The point upper anchor is contaminated-quantile based, not a CI.
    """
    x=np.asarray(ratios,float)
    if x.ndim!=1 or len(x)<16 or not np.isfinite(x).all() or np.any(x<=0):
        raise ValueError('Positive finite target ratios required')
    threshold=float(np.quantile(x,.4))
    central=x[x<=threshold]
    anchor=float(np.quantile(x,.25)/f.ppf(.25,numerator_df,denominator_df))
    low,high=np.log(anchor/9),np.log(anchor)
    def objective(log_scale):
        scale=np.exp(log_scale)
        return -float((f.logpdf(central/scale,numerator_df,denominator_df)-log_scale
                       -f.logcdf(threshold/scale,numerator_df,denominator_df)).mean())
    fit=minimize_scalar(objective,bounds=(low,high),method='bounded',options={'xatol':1e-8,'maxiter':100})
    if not fit.success or not np.isfinite(fit.fun):
        raise FloatingPointError('Central empirical-null fit failed')
    scale=float(np.exp(fit.x))
    return scale,{'converged':bool(fit.success),'iterations':int(fit.nfev),
        'threshold':threshold,'central_training_genes':len(central),
        'training_genes':len(x),'objective':float(fit.fun),'scale':scale,
        'quantile_anchor':anchor,'bound_active':bool(min(fit.x-low,high-fit.x)<1e-4),
        'zero_assumption':'nonnull contribution to lower 40% is negligible; empirical, not guaranteed',
        'selection_corrected_likelihood':True,'changes_tested_observations':False}


def domain_variance(training_means,training_q,dimension,base_variance,mode,shape=None):
    """A training-only scale mechanism with always-on and gated ablations.

    gated_* requires point variance > 4*base. This large-shift engineering
    gate limits signal-induced inflation; it is NOT an identification or
    validity guarantee. Always_* exposes the cost of removing that gate.
    Modes not starting with transport only increase the supplied base.
    transport_* are bidirectional empirical scale fits, not bounds.
    transport_pooled_fcentral pools diagonal-standardized component ratios
    in a COMPOSITE objective; dependent studies are not independent genes.
    """
    m=np.asarray(training_means,float)
    q=np.asarray(training_q,float)
    if m.ndim!=2 or q.shape!=(len(m),) or np.any(q<=0) or base_variance<=0:
        raise ValueError('Compatible training means, positive energies and variance required')
    if mode not in ['gated_point','gated_upper','always_point','always_upper','gated_fcentral','always_fcentral','shrink_fcentral','transport_fcentral','transport_pooled_fcentral','anchored_fcentral']:
        raise ValueError('Unknown domain scale mode')
    rows=np.arange(len(m))
    scores=np.abs(m[rows,rows % m.shape[1]])/np.sqrt(q/dimension)
    point,upper,info=scale_from_absolute_scores(scores,dimension)
    trigger=bool(mode.startswith(('always','shrink','transport','anchored')) or point>4*base_variance)
    proposed=upper if mode.endswith('upper') else point
    if mode.endswith('fcentral'):
        if shape is None or np.shape(shape)!=(m.shape[1],m.shape[1]):
            raise ValueError('Study shape required for central F fit')
        if mode=='transport_pooled_fcentral':
            diagonal=np.diag(shape)
            if not np.isfinite(diagonal).all() or np.any(diagonal<=0):
                raise ValueError('Positive finite study diagonals required')
            ratio=((m*m/diagonal)/(q[:,None]/dimension)).reshape(-1)
            proposed,central_info=central_f_scale(ratio,1,dimension)
            central_info.update(training_coordinates=central_info.pop('training_genes'),
                central_training_coordinates=central_info.pop('central_training_genes'),
                training_genes=len(m),studies=m.shape[1],numerator_df=1,
                independent_coordinate_claim=False,selection_corrected_likelihood=False,
                selection_corrected_composite_likelihood=True,
                zero_assumption='nonnull contribution to lower 40% of marginal coordinate ratios is negligible; empirical',
                uncertainty='No independent-coordinate standard error or confidence bound is reported')
        else:
            mean_energy=np.einsum('gs,st,gt->g',m,np.linalg.inv(shape),m)
            ratio=(mean_energy/m.shape[1])/(q/dimension)
            proposed,central_info=central_f_scale(ratio,m.shape[1],dimension)
        if mode.startswith(('shrink','anchored')):
            selected_region=ratio[ratio<=central_info['threshold']]
            base_objective=-float((f.logpdf(selected_region/base_variance,m.shape[1],dimension)
                -np.log(base_variance)-f.logcdf(central_info['threshold']/base_variance,m.shape[1],dimension)).mean())
            statistic=max(0.,2*len(selected_region)*(base_objective-central_info['objective']))
            if mode=='anchored_fcentral':
                # One free scale versus the fixed calibration scale on EXACTLY
                # the same selected region. The region is data-selected and
                # the zero assumption may fail: this is NOT a regular LR test,
                # ordinary BIC proof, Bayes factor or FDR certificate.
                penalty=float(np.log(len(selected_region)))
                adopted=bool(statistic>penalty)
                central_info.update(base_objective=base_objective,likelihood_ratio_statistic=statistic,
                    complexity_penalty=penalty,free_scale_adopted=adopted,
                    comparison_training_genes=len(selected_region),
                    rule='2*conditional_loglik_gain > log(number of central training genes)',
                    inference='information-criterion heuristic; random selected region and zero assumption')
                if not adopted:
                    proposed=base_variance
            else:
                fraction=statistic/(1+statistic)
                proposed=base_variance+fraction*max(0.,proposed-base_variance)
                central_info.update(base_objective=base_objective,likelihood_ratio_statistic=statistic,
                    shrinkage_fraction=fraction,shrinkage_rule='LR/(1+LR); heuristic, not a posterior probability or confidence bound')
        info['central_f']=central_info
    selected=max(base_variance,proposed) if trigger else base_variance
    if mode.startswith(('transport','anchored')):
        selected=proposed
    if not np.isfinite(selected):
        raise ValueError('Insufficient training genes for finite scale bound')
    return selected,{**info,'mode':mode,'base_variance':base_variance,
        'selected_variance':selected,'trigger':trigger,'gate_variance_ratio':4.,
        'uses_held_gene':False,'uses_truth_labels':False,
        'allows_variance_decrease':mode.startswith(('transport','anchored')),
        'interpretation':'development scale correction; not a universal plug-in FDR guarantee'}
