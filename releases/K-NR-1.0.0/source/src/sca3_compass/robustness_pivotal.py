"""R1: split-calibration predictive Studentization, not fitted-null plug-in.

Known compound/common-location normal-scale-mixture model, independent genes
and calibration rows. TRAIN chooses directions/pilot/gates; disjoint VALIDATE
calibration is used only by the predictive pivot. Unknown study shape and
arbitrary common positive radial scales cancel from each scalar Student pivot.
See the window protocol/review for proof scope; no general drift guarantee.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
import numpy as np
from scipy.special import expit
from scipy.stats import beta,t
from .robustness_methods import contrasts,tyler_shape
from .robustness_prior import fit_energy_prior
from .robustness_patterns import fit_pattern_mixture
from .robustness_projection import positive_direction
from .robustness_pilot_selection import pilot_training_receipt,apply_pilot_multiplier

MULTIPLIERS=(.25,.5,.65,.8)
SAFE_BASES=('ordinary_bonf','support_bonf','weighted_bonf','support_simes_by')
EMPIRICAL_BASES=('ordinary_simes_empirical','support_simes_empirical')
MODES=('projection','uniform_projection',*SAFE_BASES,*EMPIRICAL_BASES)


def _finite(x,ndim):
    x=np.asarray(x,float)
    if x.ndim!=ndim or not np.isfinite(x).all():raise ValueError('Finite matching input required')
    return x


def ratio_components(z):
    """Scale each row/gene by a positive number to avoid overflow; ratio invariant."""
    z=np.asarray(z,float)
    scale=np.max(np.abs(z),axis=tuple(range(1,z.ndim)),keepdims=True)
    if np.any(scale==0):raise ValueError('Zero whole vector is outside the continuous pivot model')
    safe=z/scale;means=safe.mean(-1);y=(safe-means[...,None])@contrasts(z.shape[-1])
    return means,y


def calibration_scale(calibration):
    cal=_finite(calibration,2);n,k=cal.shape
    if n<4 or k<2:raise ValueError('At least4 calibration vectors and2 pipelines required')
    mean,y=ratio_components(cal);den=np.sqrt(np.sum(y*y,axis=-1)/(k-1))
    if np.any(den<=0):raise ValueError('Degenerate calibration contrasts')
    ratios=np.abs(mean/den);order=(n+1)//2
    value=float(np.partition(ratios,order-1)[order-1]/t.ppf(.75,k-1))
    if not np.isfinite(value) or value<=0:raise FloatingPointError('Invalid calibration order-statistic scale')
    return value,{'n':n,'order':order,'df':k-1,'scale':value,'estimator':'lower_absolute_order_median_divided_by_abs_t_median'}


@lru_cache(maxsize=24)
def predictive_table(n,df,integration_nodes=4096,statistic_nodes=2048):
    """Monotone quadrature upper/lower sums; NOT a black-box exact quadrature.

    V=F_Beta(U) is uniform. At each increasing v endpoint, B(v) is known.
    Student survival(z*B(v)) decreases in v for z>0. Left/right sums bound
    the integral in exact arithmetic. Floating special-function roundoff is
    audited separately, with an explicit1e-10 guard; not formal interval arithmetic.
    A down-rounded statistic grid preserves conservative interpolation.
    """
    if not isinstance(n,int) or n<4 or not isinstance(df,int) or df<1:raise ValueError('Invalid pivot dimensions')
    order=(n+1)//2
    v=np.r_[0.,expit(np.linspace(-28,28,integration_nodes+1)),1.]
    u=beta.ppf(v,order,n+1-order)
    residual=float(np.max(np.abs(beta.cdf(u,order,n+1-order)-v)))
    if residual>1e-10 or np.any(np.diff(u)<0):raise FloatingPointError('Order-statistic quantile audit failed')
    b=t.ppf((1+u)/2,df)/t.ppf(.75,df)
    if np.any(np.diff(b)<0):raise FloatingPointError('Nonmonotone abs-t quantiles')
    z=np.r_[0.,t.isf(np.geomspace(.49,1e-12,statistic_nodes),df)]
    lower=np.empty(len(z));upper=np.empty(len(z));lower[0]=upper[0]=.5
    width=np.diff(v)
    for begin in range(1,len(z),128):
        end=min(begin+128,len(z));f=t.sf(z[begin:end,None]*b[None,:],df)
        lower[begin:end]=f[:,1:]@width
        upper[begin:end]=f[:,:-1]@width
    if np.any(lower>upper+1e-14) or not np.isfinite(upper).all():raise FloatingPointError('Tail enclosure failure')
    guard=1e-10
    upper=np.minimum(.5,upper+guard)
    lower=np.maximum(0,lower-guard)
    z.flags.writeable=lower.flags.writeable=upper.flags.writeable=False
    return {'z':z,'lower':lower,'upper':upper,'diagnostics':{'n':n,'df':df,'order':order,
        'integration_nodes':integration_nodes,'statistic_nodes':statistic_nodes,'roundoff_guard':guard,
        'inverse_beta_max_residual':residual,'max_integration_gap':float(np.max(upper-lower)),
        'max_smalltail_relative_gap':float(np.max((upper-lower)[upper<=.01]/upper[upper<=.01])),
        'interpolation':'left statistic endpoint, conservative; no extrapolation past last node',
        'numerical_claim':'monotone exact-arithmetic bound with audited floating-point special functions, not interval-certified'}}


def predictive_tail(statistic,n,df):
    x=np.asarray(statistic,float)
    if not np.isfinite(x).all():raise ValueError('Finite studentized statistic required')
    table=predictive_table(int(n),int(df));index=np.searchsorted(table['z'],x,side='right')-1
    index=np.clip(index,0,len(table['z'])-1)
    return np.where(x>0,table['upper'][index],1.)


def learn(train,learn_cal):
    """Accuracy affects power only. ALL arguments belong to permitted TRAIN."""
    g,s,k=train.shape;means=train.mean(-1);residual=(train-means[...,None])@contrasts(k)
    cal_scale,cal_info=calibration_scale(learn_cal)
    kap=cal_scale**2;rho=float((k*kap-1)/(k*kap+k-1))
    receipt={'uses_validation_calibration':False,'training_genes':g,'learning_calibration':cal_info,
        'rho_for_learning_only':rho,'fallback':False,'failure_stage':None}
    shape=np.eye(s);profiles=np.ones((2,s));gamma=np.zeros(2)
    try:
        shape,shape_info=tyler_shape(residual.transpose(0,2,1).reshape(-1,s))
        receipt['shape']=shape_info
        if not shape_info['converged']:raise ArithmeticError('Training shape failed convergence')
        q=np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(shape),residual)/(1-rho)
        prior=fit_energy_prior(q,s*(k-1));receipt['prior']={key:(None if key=='df' and np.isinf(value) else value) for key,value in prior.items()}
        if not prior['converged']:raise ArithmeticError('Training prior failed convergence')
        df=np.inf if prior['gaussian_bic_selected'] else prior['df']+s*(k-1)
        v=(1+(k-1)*rho)/k
        variance=np.full(g,v*prior['scatter']) if np.isinf(df) else v*(prior['df']*prior['scatter']+q)/df
        fitted=fit_pattern_mixture(means,variance,shape,df)
        # Full fits stay in per-repetition records; caller serializes numpy safely.
        receipt['pattern_fit']=fitted
        if not fitted['diagnostics']['converged']:raise ArithmeticError('Training pattern failed convergence')
        for sign,key in enumerate(['positive','negative']):
            summary=fitted['replicated'][key]
            profiles[sign]=np.maximum(0,(1 if sign==0 else -1)*summary['dominant_pattern'])
            reliable=summary['mass']>max(.03,3/g)
            gamma[sign]=np.clip((summary['dominant_mass_fraction']-.4)/.4,0,1) if reliable else 0.
    except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
        receipt.update(fallback=True,failure_stage=str(error));shape=np.eye(s);profiles[:]=1.;gamma[:]=0.
    receipt.update(profiles=profiles.tolist(),gamma=gamma.tolist(),shape_for_direction=shape.tolist())
    return profiles,shape,gamma,cal_scale,receipt


def pc_components(z,profiles,shape,cal_scale,cal_n):
    """All triple tests share the same held observations and calibration."""
    g,s,k=z.shape;df=k-1
    m,y=ratio_components(z);den=np.sqrt(np.sum(y*y,axis=-1)/df)
    if np.any(den<=0):raise ValueError('Degenerate held study contrasts')
    marginal=np.stack((predictive_tail(m/den/cal_scale,cal_n,df),predictive_tail(-m/den/cal_scale,cal_n,df)),1)
    output={mode:np.zeros((g,2)) for mode in MODES}
    for sign in range(2):
        for ix in combinations(range(4),3):
            ids=list(ix);a=profiles[sign,ids]
            if not np.any(a>0):a=np.ones(3)
            p=marginal[:,sign,ids];keep=a>0;count=int(keep.sum());pk=p[:,keep]
            safe=np.minimum(1,3*p.min(1));support=np.minimum(1,count*pk.min(1))
            allocated=np.minimum(1,(pk/(a[keep]/a.sum())).min(1))
            support_simes=np.minimum(1,(np.sort(pk,axis=1)*count/np.arange(1,count+1)).min(1))
            ordinary_simes=np.minimum(1,(np.sort(p,axis=1)*3/np.arange(1,4)).min(1))
            vals={'ordinary_bonf':safe,'support_bonf':support,'weighted_bonf':allocated,
                'support_simes_by':np.minimum(1,support_simes*np.sum(1/np.arange(1,count+1))),
                'support_simes_empirical':support_simes,'ordinary_simes_empirical':ordinary_simes}
            for mode,profile in [('projection',a),('uniform_projection',np.ones(3))]:
                direction=positive_direction(profile,shape[np.ix_(ids,ids)])
                direction/=np.max(direction)
                numerator=(m[:,ids]@direction)*(1 if sign==0 else -1)
                projected_y=np.einsum('i,gij->gj',direction,y[:,ids,:])
                denominator=np.sqrt(np.sum(projected_y*projected_y,axis=1)/df)
                if np.any(denominator<=0):raise ValueError('Degenerate projected contrasts')
                vals[mode]=predictive_tail(numerator/denominator/cal_scale,cal_n,df)
            for mode,value in vals.items():output[mode][:,sign]=np.maximum(output[mode][:,sign],value)
    return output


def evaluate(z,calibration):
    z=_finite(z,3);cal=_finite(calibration,3);g,s,k=z.shape
    if s!=4 or cal.shape[0]!=s or cal.shape[-1]!=k or g<8 or k<2:raise ValueError('Four studies, G>=8, K>=2 required')
    flat=cal.reshape(-1,k);learn_cal=flat[::2];valid_cal=flat[1::2]
    scale,cal_info=calibration_scale(valid_cal)
    output={f'R1_pilotc{c}_{mode}_eBH':np.empty((g,2)) for c in MULTIPLIERS for mode in MODES}
    for c in MULTIPLIERS:output[f'R1_pilotc{c}_projection_gate_eBH']=np.empty((g,2))
    receipts=[];pout={mode:np.empty((g,2)) for mode in MODES}
    for fold in range(2):
        held=np.arange(g)%2==fold
        profiles,shape,gamma,training_scale,info=learn(z[~held],learn_cal)
        training=pc_components(z[~held],profiles,shape,training_scale,len(learn_cal))
        actual=pc_components(z[held],profiles,shape,scale,len(valid_cal))
        pilots={mode:pilot_training_receipt(p) for mode,p in training.items()}
        for mode in MODES:
            pout[mode][held]=actual[mode]
            for c in MULTIPLIERS:
                output[f'R1_pilotc{c}_{mode}_eBH'][held]=apply_pilot_multiplier(actual[mode],pilots[mode],2*g,c)[0]
        for c in MULTIPLIERS:
            output[f'R1_pilotc{c}_projection_gate_eBH'][held]=(
                gamma*output[f'R1_pilotc{c}_projection_eBH'][held]+(1-gamma)*output[f'R1_pilotc{c}_ordinary_bonf_eBH'][held])
        info.update(fold=fold,pilot=pilots);receipts.append(info)
    if any(not np.isfinite(v).all() or np.any(v<0) for v in output.values()):raise FloatingPointError('Invalid R1 evidence')
    return output,{'method_version':'R1-development','calibration_validation':cal_info,
        'learn_cal_rows':list(range(0,len(flat),2)),'validate_cal_rows':list(range(1,len(flat),2)),
        'folds':receipts,'held_pvalues':pout,'tail_table':predictive_table(len(valid_cal),k-1)['diagnostics'],
        'own_Q_weights':False,'null_gene_floor':False,'validation_used_for_learning':False,
        'empirical_only_simes_labels':list(EMPIRICAL_BASES),
        'scope':'independent genes/calibration, common pipeline location, compound angular transport, gene-common normal-scale mixture; no arbitrary covariance drift guarantee'}
