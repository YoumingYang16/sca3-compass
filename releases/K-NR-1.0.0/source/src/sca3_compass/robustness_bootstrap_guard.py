"""R1 backup: whole-gene nuisance-bootstrap sensitivity guard.

NOT a Berger--Boos confidence set, not a bootstrap-p-value theorem. This
empirical repair keeps conditional Student pooling, drops own-Q weights,
and takes the upper intersection tail over the fitted model and a finite
predeclared nuisance perturbation bank. Correctness at the true nuisance
does not prove correctness of this estimated finite bank. Scoped C1 is required.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.stats import norm,t
from .robustness_methods import contrasts,tyler_shape,fit_calibration
from .robustness_prior import fit_energy_prior
from .robustness_patterns import fit_pattern_mixture
from .robustness_projection import positive_direction
from .robustness_simes import simes_intersection
from .robustness_pilot_selection import pilot_training_receipt,apply_pilot_multiplier

MULTIPLIERS=(.25,.5,.65,.8)
SAFE_BASES=('ordinary_bonf','support_bonf','weighted_bonf','support_simes_by')
EMPIRICAL_BASES=('ordinary_simes_empirical','support_simes_empirical')
MODES=('projection','uniform_projection',*SAFE_BASES,*EMPIRICAL_BASES)

def nuisance(residual,cal):
    rho_fit=fit_calibration(cal)
    if not rho_fit.converged:raise ArithmeticError('Calibration optimizer did not converge')
    shape,info=tyler_shape(residual.transpose(0,2,1).reshape(-1,4))
    if not info['converged']:raise ArithmeticError('Study-shape optimizer did not converge')
    rho=rho_fit.rho;dim=4*residual.shape[-1]
    q=np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(shape),residual)/(1-rho)
    prior=fit_energy_prior(q,dim)
    if not prior['converged']:raise ArithmeticError('Energy-prior optimizer did not converge')
    return {'rho':rho,'shape':shape,'prior':prior,'shape_fit':info,'calibration_fit':vars(rho_fit),'dimension':dim}

def variance(residual,theta):
    k=residual.shape[-1]+1;rho=theta['rho'];prior=theta['prior'];d=theta['dimension']
    q=np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(theta['shape']),residual)/(1-rho)
    v=(1+(k-1)*rho)/k
    if prior['gaussian_bic_selected']:return np.full(len(residual),v*prior['scatter']),np.inf
    df=prior['df']+d
    return v*(prior['df']*prior['scatter']+q)/df,df

def learn_profiles(means,residual,theta):
    var,df=variance(residual,theta)
    fit=fit_pattern_mixture(means,var,theta['shape'],df)
    if not fit['diagnostics']['converged']:
        return np.ones((2,4)),np.zeros(2),{'fallback':True,'reason':'pattern convergence','fit':fit}
    profiles=[];gamma=[]
    for sign,key in [(1,'positive'),(-1,'negative')]:
        s=fit['replicated'][key];profiles.append(np.maximum(0,sign*s['dominant_pattern']))
        gamma.append(float(np.clip((s['dominant_mass_fraction']-.4)/.4,0,1)) if s['mass']>max(.03,3/len(means)) else 0.)
    return np.array(profiles),np.array(gamma),{'fallback':False,'fit':fit}

def bootstrap_bank(train_residual,cal,draws,rng):
    """Resample whole gene matrices; never treat their columns as iid genes."""
    original=nuisance(train_residual,cal);bank=[original];receipts=[]
    for b in range(draws):
        genes=rng.integers(0,len(train_residual),len(train_residual));rows=rng.integers(0,len(cal),len(cal))
        try:
            value=nuisance(train_residual[genes],cal[rows]);bank.append(value)
            receipts.append({'draw':b,'status':'completed','gene_indices':genes.tolist(),'calibration_indices':rows.tolist()})
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as err:
            # No unrecorded drop. A failed perturbation blocks guarded evidence.
            receipts.append({'draw':b,'status':'failed','error':repr(err),'gene_indices':genes.tolist(),'calibration_indices':rows.tolist()})
            return original,bank,receipts,False
    return original,bank,receipts,True

def components(means,residual,profiles,original,bank):
    """Directions frozen at original TRAIN; max tails BEFORE combination/PC."""
    outputs={tag:{mode:np.zeros((len(means),2)) for mode in MODES} for tag in ['plugin','guard']}
    variance_bank=[variance(residual,theta) for theta in bank]
    for sign in range(2):
        signed=means*(1 if sign==0 else -1)
        for triple in combinations(range(4),3):
            ids=list(triple);a=profiles[sign,ids]
            if not np.any(a>0):a=np.ones(3)
            keep=a>0;count=int(keep.sum())
            directions={mode:positive_direction(profile,original['shape'][np.ix_(ids,ids)]) for mode,profile in [('projection',a),('uniform_projection',np.ones(3))]}
            upper={mode:np.zeros(len(means)) for mode in MODES}
            for b,(theta,(var,df)) in enumerate(zip(bank,variance_bank)):
                shape=theta['shape'];sub=shape[np.ix_(ids,ids)]
                statistic=signed[:,ids]/np.sqrt(var[:,None]*np.diag(sub))
                marginal=np.where(statistic>0,norm.sf(statistic) if np.isinf(df) else t.sf(statistic,df),1.)
                pk=marginal[:,keep]
                unadjusted=np.minimum(1,(np.sort(pk,axis=1)*count/np.arange(1,count+1)).min(1))
                vals={'ordinary_bonf':np.minimum(1,3*marginal.min(1)),
                    'support_bonf':np.minimum(1,count*pk.min(1)),
                    'weighted_bonf':np.minimum(1,(pk/(a[keep]/a.sum())).min(1)),
                    'support_simes_by':np.minimum(1,unadjusted*np.sum(1/np.arange(1,count+1))),
                    'ordinary_simes_empirical':simes_intersection(marginal,sub),
                    'support_simes_empirical':simes_intersection(pk,sub[np.ix_(keep,keep)])}
                for mode,direction in directions.items():
                    statistic=(signed[:,ids]@direction)/np.sqrt(var*(direction@sub@direction))
                    vals[mode]=np.where(statistic>0,norm.sf(statistic) if np.isinf(df) else t.sf(statistic,df),1.)
                for mode,value in vals.items():
                    upper[mode]=np.maximum(upper[mode],value)
                    if b==0:outputs['plugin'][mode][:,sign]=np.maximum(outputs['plugin'][mode][:,sign],value)
            for mode in MODES:outputs['guard'][mode][:,sign]=np.maximum(outputs['guard'][mode][:,sign],upper[mode])
    return outputs

def evaluate(z,calibration,*,draws=16,seed=4151601):
    z=np.asarray(z,float);cal=np.asarray(calibration,float)
    if z.ndim!=3 or z.shape[1]!=4 or cal.ndim!=3 or cal.shape[0]!=4 or z.shape[2]!=cal.shape[2] or not np.isfinite(z).all() or not np.isfinite(cal).all():raise ValueError('Finite matching4-study arrays required')
    g,s,k=z.shape
    if draws<1 or g<8 or k<2:raise ValueError('Invalid sizes')
    means=z.mean(-1);residual=(z-means[...,None])@contrasts(k);flat=cal.reshape(-1,k)
    output={f'R1B_{tag}_pilotc{c}_{m}_eBH':np.empty((g,2)) for tag in ['plugin','guard'] for c in MULTIPLIERS for m in [*MODES,'projection_gate']}
    pout={f'{tag}_{m}':np.empty((g,2)) for tag in ['plugin','guard'] for m in MODES};folds=[]
    for fold in range(2):
        held=np.arange(g)%2==fold;rng=np.random.default_rng(np.random.SeedSequence([seed,fold]))
        try:original,bank,boot,success=bootstrap_bank(residual[~held],flat,draws,rng)
        except (ValueError,ArithmeticError,np.linalg.LinAlgError) as err:
            # No plug-in evidence at all if its own fit failed; fail family visibly.
            raise ArithmeticError('Original nuisance failed:'+repr(err)) from err
        profiles,gamma,pattern=learn_profiles(means[~held],residual[~held],original)
        training=components(means[~held],residual[~held],profiles,original,bank)
        actual=components(means[held],residual[held],profiles,original,bank)
        if not success:
            for m in MODES:actual['guard'][m][:]=1.;training['guard'][m][:]=1.
        pilots={}
        for tag in ['plugin','guard']:
            for mode in MODES:
                p=pilot_training_receipt(training[tag][mode]);pilots[tag+'_'+mode]=p
                pout[tag+'_'+mode][held]=actual[tag][mode]
                for c in MULTIPLIERS:output[f'R1B_{tag}_pilotc{c}_{mode}_eBH'][held]=apply_pilot_multiplier(actual[tag][mode],p,2*g,c)[0]
            for c in MULTIPLIERS:
                output[f'R1B_{tag}_pilotc{c}_projection_gate_eBH'][held]=gamma*output[f'R1B_{tag}_pilotc{c}_projection_eBH'][held]+(1-gamma)*output[f'R1B_{tag}_pilotc{c}_ordinary_bonf_eBH'][held]
        folds.append({'fold':fold,'training_genes':int((~held).sum()),'fallback':not success or pattern['fallback'],
            'guard_success':success,'profiles':profiles.tolist(),'gamma':gamma.tolist(),'shape_for_direction':original['shape'],
            'bank':bank,'bootstrap_receipts':boot,'pattern':pattern,'pilot':pilots,
            'uses_held_for_learning':False,'uses_truth':False})
    return output,{'method_version':'R1B-development','folds':folds,'held_pvalues':pout,
        'tail_table':{'not_applicable':True,'method':'analytic normal/Student tails over finite bootstrap bank'},
        'own_Q_weights':False,'null_gene_floor':False,'bootstrap_draws':draws,'bootstrap_seed':seed,
        'calibration_policy':'all independent calibration is permitted TRAIN; no separate exact-predictive VAL claim',
        'validity':'EMPIRICAL finite nuisance perturbation guard, no coverage certificate, no exact bootstrap-p-value claim',
        'scope':'common pipeline location, target/calibration compound angular transport, independent gene/calibration units; target radial Gaussian or IG working model'}
