"""Scoped callable K numerical-repair reference. EMPIRICAL ONLY, not novel.

No labels or true parameters enter inference. Historical implementations remain
unchanged; this sidecar selects ONE existing K label and a fixed comparator set.
"""
from __future__ import annotations

import time
import numpy as np
from scipy.special import ndtr
from .molecular_methods import ebh, fdr_adjust, partial_conjunction
from .robustness_methods import contrasts, tyler_shape, student_tail
from .robustness_prior import fit_energy_prior
from .robustness_bootstrap_guard import learn_profiles, components
from .robustness_calibration_efficiency import rho_guard_theta
from .robustness_calibration_numeric_repair import fit_calibration_with_diagnostics
from .robustness_pilot_selection import pilot_training_receipt, apply_pilot_multiplier

VERSION = 'K-NR-1.0.0'
METHODS = ('K_NR', 'B_original', 'B_fair_plugin', 'B_fair_guard', 'B_common', 'B_strong',
           'B_fair_conditional_eBH','B_fair_conditional_BY')
ASSUMPTIONS = [
    'Common pipeline location for each gene/study; comparable prespecified effects.',
    'Independent genes and centered independent calibration vectors.',
    'Calibration and target share compound pipeline angular shape.',
    'Separable study/pipeline Gaussian noise with gene-common positive scale.',
    'Gaussian or inverse-gamma target radial working model; finite mean.',
    'These distributional assumptions are NOT certified by numerical diagnostics.',
]
SCOPE = 'D1 finite simulation research scope in V1_ACCEPTANCE.md; G256,S4,K6,n>=4; no clinical certification'


def check_inputs(z, calibration, acknowledged):
    if acknowledged is not True:
        raise ValueError('Explicit acknowledgement of untestable model assumptions required')
    for value in (z, calibration):
        if np.ma.isMaskedArray(value) or np.asarray(value).dtype.kind not in 'fiu':
            raise ValueError('Unmasked real numeric arrays required')
    z, cal = np.asarray(z, float), np.asarray(calibration, float)
    if (z.shape != (256,4,6) or cal.ndim != 3 or cal.shape[0] != 4
            or cal.shape[2] != 6 or cal.shape[1] < 4
            or not np.isfinite(z).all() or not np.isfinite(cal).all()):
        raise ValueError('Validated dimensions G256,S4,K6, calibration4xn x6 n>=4 and finite inputs required')
    if np.any(np.sum((z-z.mean(-1,keepdims=True))**2,axis=(1,2)) <= 0):
        raise ValueError('Degenerate whole-gene pipeline contrasts are unsupported')
    return z, cal


def _fit(cal):
    fitted, detail = fit_calibration_with_diagnostics(cal)
    if not fitted.converged:
        raise ArithmeticError('Calibration optimizer did not converge')
    return fitted, detail


def _nuisance(residual, fitted):
    shape, info = tyler_shape(residual.transpose(0,2,1).reshape(-1,4))
    if not info['converged']:
        raise ArithmeticError('Study-shape optimizer did not converge')
    rho = fitted.rho
    dimension = 4*residual.shape[-1]
    q = np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(shape),residual)/(1-rho)
    prior = fit_energy_prior(q,dimension)
    if not prior['converged']:
        raise ArithmeticError('Energy-prior optimizer did not converge')
    return {'rho':rho,'shape':shape,'prior':prior,'shape_fit':info,
            'calibration_fit':vars(fitted),'dimension':dimension}


def evaluate(z, calibration, *, seed=17091601, acknowledge_scope=False):
    """One fixed primary algorithm; comparator tails use the same calibration.

    Numeric failure raises explicitly. Bootstrap failure retains the historical
    p=1 fold guard with an explicit receipt. No estimated coverage claim.
    """
    z, cal = check_inputs(z,calibration,acknowledge_scope)
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('Nonnegative integer algorithm seed required')
    begin, cpu = time.perf_counter(),time.process_time()
    g,s,k = z.shape
    means = z.mean(-1)
    residual = (z-means[...,None])@contrasts(k)
    flat = cal.reshape(-1,k)
    fitted, fit_detail = _fit(flat)
    signed = np.stack((z,-z),axis=1)
    gaussian = ndtr(-signed)
    original = partial_conjunction(np.minimum(1,k*gaussian.min(-1)),2)
    raw = student_tail(signed,1.,fitted)
    plugin = partial_conjunction(np.minimum(1,k*raw.min(-1)),2)
    pc = {'B_original':original,'B_fair_plugin':plugin,
          'B_fair_guard':np.empty((g,2)),'B_common':np.empty((g,2)),
          'K_projection':np.empty((g,2)),'K_ordinary_bonf':np.empty((g,2))}
    e = {'K_NR':np.empty((g,2)),'B_strong':np.empty((g,2))}
    folds = []
    for fold in range(2):
        held = np.arange(g)%2 == fold
        rng = np.random.default_rng(np.random.SeedSequence([seed,fold]))
        theta = _nuisance(residual[~held],fitted)
        profiles, gamma, pattern = learn_profiles(means[~held],residual[~held],theta)
        rhos, fits, resampling = [fitted.rho], [fitted], []
        success = True
        for draw in range(16):
            rng.integers(0,int((~held).sum()),int((~held).sum()))  # historic K stream
            rows = rng.integers(0,len(flat),len(flat))
            try:
                fb, fd = _fit(flat[rows])
                fits.append(fb); rhos.append(fb.rho)
                resampling.append({'draw':draw,'status':'completed','rows':rows.tolist(),
                                   'fit':vars(fb),'numeric':fd})
            except (ValueError,ArithmeticError,np.linalg.LinAlgError) as error:
                success = False
                resampling.append({'draw':draw,'status':'failed','rows':rows.tolist(),'error':repr(error)})
                break
        guarded = rho_guard_theta(theta,rhos)
        train = components(means[~held],residual[~held],profiles,theta,[theta,guarded])
        actual = components(means[held],residual[held],profiles,theta,[theta,guarded])
        if not success:
            for mode in ('projection','ordinary_bonf'):
                train['guard'][mode][:] = 1.
                actual['guard'][mode][:] = 1.
        converted, pilots = {}, {}
        for mode in ('projection','ordinary_bonf'):
            pilot = pilot_training_receipt(train['guard'][mode])
            converted[mode] = apply_pilot_multiplier(actual['guard'][mode],pilot,2*g,.5)[0]
            pilots[mode] = pilot
            pc['K_'+mode][held] = actual['guard'][mode]
        e['K_NR'][held] = gamma*converted['projection']+(1-gamma)*converted['ordinary_bonf']
        pilot = pilot_training_receipt(train['plugin']['support_simes_by'])
        e['B_strong'][held] = apply_pilot_multiplier(actual['plugin']['support_simes_by'],pilot,2*g,.5)[0]
        pilots['B_strong'] = pilot
        protected = np.maximum.reduce([student_tail(signed[held],1.,f) for f in fits])
        if not success:
            protected[:] = 1.
        pc['B_fair_guard'][held] = partial_conjunction(np.minimum(1,k*protected.min(-1)),2)
        pc['B_common'][held] = partial_conjunction(protected[...,0],2)
        folds.append({'fold':fold,'held_indices':np.flatnonzero(held).tolist(),
                      'training_genes':int((~held).sum()),'original':theta,'guarded':guarded,
                      'profiles':profiles,'gamma':gamma,'pattern':pattern,'pilots':pilots,
                      'guard_success':success,'resampling':resampling,'rho_bank':rhos,
                      'uses_test_labels':False,'uses_held_for_learning':False})
    for key,value in {**pc,**e}.items():
        if not np.isfinite(value).all() or np.any(value<0) or (key in pc and np.any(value>1)):
            raise FloatingPointError('Invalid inference array: '+key)
    # Existing ordinary component, not a new candidate. Reconstruct its e-values
    # using the SAME stored TRAIN pilot; no held-dependent selection.
    e['B_fair_conditional_eBH'] = np.empty((g,2))
    pc['B_fair_conditional_BY'] = pc['K_ordinary_bonf'].copy()
    for fold in folds:
        held = np.arange(g)%2 == fold['fold']
        e['B_fair_conditional_eBH'][held] = apply_pilot_multiplier(
            pc['K_ordinary_bonf'][held],fold['pilots']['ordinary_bonf'],2*g,.5)[0]
    discoveries = {m:(ebh(e[m],.05) if m in e else fdr_adjust(pc[m],'BY')<=.05) for m in METHODS}
    adjusted = {m:fdr_adjust(pc[m],'BY') for m in METHODS if m in pc}
    return {'method_version':VERSION,'release_status':'V0_REFERENCE_PENDING_INDEPENDENT_ACCEPTANCE',
            'interface_version':'evaluation-2','evidence_level':'EMPIRICAL_ONLY','supported_scope':SCOPE,'assumptions':ASSUMPTIONS,
            'status':'COMPUTED_WITH_DECLARED_FOLD_FALLBACK' if any(not f['guard_success'] for f in folds) else 'COMPUTED',
            'alpha':.05,'family':'2G signed PC r2 of4','seed':int(seed),
            'discoveries':discoveries,'p_values':pc,'e_values':e,'adjusted_p_values':adjusted,
            'diagnostics':{'initial_calibration_numeric':fit_detail,'folds':folds,
                           'untestable_assumptions_verified':False,'uses_truth':False,
                           'statistical_guarantee':'NO general finite-calibration FDR guarantee',
                           'cpu_seconds':time.process_time()-cpu,'elapsed_seconds':time.perf_counter()-begin}}
