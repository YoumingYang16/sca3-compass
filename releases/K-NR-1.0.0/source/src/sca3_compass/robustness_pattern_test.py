"""Development cross-fitted pattern-directed PC testing and cone mixtures.

The pattern prior only chooses nonnegative directions, NOT posterior error
rates or truth labels. Each held fold uses the OTHER genes' mixture fit.
All nuisance shapes/energy priors are the same as target-only cone baselines.
Exact known-nuisance validity needs independent training/held genes; fitted
nuisance parameters and correlated genes need empirical validation.
"""
import numpy as np
from .robustness_methods import contrasts
from .robustness_patterns import fit_pattern_mixture
from .robustness_projection import projection_pc
from .robustness_weighting import power_radial_weight
from .robustness_calibrators import focused_calibrator
from .robustness_pilot import pilot_calibrator
from .robustness_cone import cone_partial_conjunction
from .robustness_block_e import block_bh_evalues


def _json_value(value):
    if isinstance(value,np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value,dict):
        return {key:_json_value(v) for key,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value,(float,np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value,np.generic):
        return value.item()
    return value


def pattern_test_candidates(z,diagnostics,base_weighted_pc,*,pilot=False,geometries=None,block=False,predictive=False,directional=False,pilot_selection=False,rank_budget=False):
    g,s,k=z.shape
    if s!=4 or 'energy_prior' not in diagnostics:
        raise ValueError('Four studies and previously fitted target energy priors required')
    if (predictive or directional or pilot_selection or rank_budget) and not pilot:
        raise ValueError('Predictive mixing requires the matching pilot calibrators')
    rho=diagnostics['fit']['rho']
    means=z.mean(-1)
    signed=np.stack((means,-means),axis=1)
    residual=(z-means[...,None])@contrasts(k)
    d=s*(k-1)
    original_means,original_signed,original_d=means,signed,d
    if geometries is not None and len(geometries)!=2:
        raise ValueError('Exactly two optional fold geometries required')
    modes=['uniform_projection','pattern_projection','pattern_mean_projection','pattern_bonf','pattern_mean_bonf','pattern_support_simes']
    result={f'{mode}{suffix}_PC':np.empty((g,2)) for mode in modes for suffix in ['','_weighted']}
    pilot_modes=[*modes,'cone','ordinary_bonf','ordinary_simes']
    if pilot:
        for fraction in [.5,.8]:
            for mode in pilot_modes:
                result[f'pilot{int(100*fraction)}_{mode}_eBH']=np.empty((g,2))
    if block:
        for level in [.025,.01]:
            for mode in pilot_modes:
                result[f'block{level}_{mode}_eBH']=np.empty((g,2))
    if predictive:
        for mode in ['pattern_projection','pattern_support_simes','selector']:
            result[f'predictive80_{mode}_eBH']=np.empty((g,2))
    budget_modes=['pattern_projection_raw','pattern_projection_gated',
        'pattern_support_simes_raw','pattern_support_simes_gated','cone']
    if directional:
        for mode in budget_modes:
            result[f'budget80_{mode}_eBH']=np.empty((g,2))
    selection_modes=['pattern_projection','pattern_support_simes','cone','ordinary_bonf','ordinary_simes']
    selection_prefixes=['pilotc0.25','pilotc0.5','pilotc0.65','pilotc0.8','pilotcv']
    if pilot_selection:
        for prefix in selection_prefixes:
            for mode in selection_modes:
                result[f'{prefix}_{mode}_eBH']=np.empty((g,2))
    if rank_budget:
        for mode in selection_modes:
            result[f'rank80_{mode}_eBH']=np.empty((g,2))
    gamma=np.zeros((g,2))
    support_gamma=np.zeros((g,2))
    fits=[]
    for fold,info in enumerate(diagnostics['energy_prior']['folds']):
        held=np.arange(g)%2==fold
        geometry=None if geometries is None else geometries[fold]
        means,signed,d=original_means,original_signed,original_d
        shape=np.asarray(info['study_shape'])
        prior=info['target_only']
        v=info['projection_variance']
        if geometry is None:
            if info.get('custom_geometry',False):
                raise ValueError('Missing custom geometry used for this energy prior')
            q=np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(shape),residual)/(1-rho)
        else:
            means=np.asarray(geometry['means'],float);q=np.asarray(geometry['q'],float)
            d=geometry['dimension']
            if (not info.get('custom_geometry',False) or means.shape!=(g,s) or q.shape!=(g,)
                    or d!=info['residual_dimension'] or not np.isfinite(means).all()
                    or not np.isfinite(q).all() or np.any(q<=0)
                    or not np.array_equal(shape,np.asarray(geometry['study_shape']))):
                raise ValueError('Pattern testing and energy-prior geometry disagree')
            signed=np.stack((means,-means),axis=1)
        if prior['gaussian_bic_selected']:
            df=np.inf
            variance=np.full(g,v*prior['scatter'])
            all_weight=np.ones(g)
        else:
            df=prior['df']+d
            variance=v*(prior['df']*prior['scatter']+q)/df
            all_weight=power_radial_weight(q,d,prior['df'],prior['scatter'],v)[0]
        weight=all_weight[held]
        fit=fit_pattern_mixture(means[~held],variance[~held],shape,df)
        dominant=[]
        average=[]
        for sign,key in [(1,'positive'),(-1,'negative')]:
            summary=fit['replicated'][key]
            dominant.append(np.maximum(0,sign*summary['dominant_pattern']))
            locations=np.maximum(0,sign*summary['patterns'])*summary['amplitudes'][:,None]
            average.append(summary['conditional_weights']@locations)
            reliable=(fit['diagnostics']['converged'] and summary['mass']>max(.03,3/(~held).sum()))
            mixing=float(np.clip((summary['dominant_mass_fraction']-.4)/.4,0,.9)) if reliable else 0.
            gamma[held,0 if sign==1 else 1]=mixing
            support_gamma[held,0 if sign==1 else 1]=(float(np.clip((summary['dominant_mass_fraction']-.4)/.4,0,1.)) if reliable else 0.)
        profiles={'uniform_projection':np.ones((2,s)),
            'pattern_projection':np.asarray(dominant),'pattern_mean_projection':np.asarray(average),
            'pattern_bonf':np.asarray(dominant),'pattern_mean_bonf':np.asarray(average),
            'pattern_support_simes':np.asarray(dominant)}
        for mode,profile in profiles.items():
            value=projection_pc(signed[held],variance[held],shape,df,profile,bonferroni='bonf' in mode,support_simes='simes' in mode)
            result[f'{mode}_PC'][held]=value
            result[f'{mode}_weighted_PC'][held]=np.minimum(1,value/weight[:,None])
        pilot_info={}
        block_info={}
        pilot_selection_info={}
        rank_info={}
        if pilot or block:
            pilot_profiles={**profiles,'ordinary_bonf':np.ones((2,s)),'ordinary_simes':np.ones((2,s))}
            for mode in pilot_modes:
                if mode=='cone':
                    training_value=cone_partial_conjunction(signed[~held],variance[~held],[shape,shape],df)
                    held_weighted=base_weighted_pc[held]
                else:
                    training_value=projection_pc(signed[~held],variance[~held],shape,df,pilot_profiles[mode],
                        bonferroni='bonf' in mode,support_simes='simes' in mode)
                    held_weighted=(result[f'{mode}_weighted_PC'][held] if mode in modes else
                        np.minimum(1,projection_pc(signed[held],variance[held],shape,df,pilot_profiles[mode],
                            bonferroni='bonf' in mode,support_simes='simes' in mode)/weight[:,None]))
                training_weighted=np.minimum(1,training_value/all_weight[~held,None])
                for fraction in ([.5,.8] if pilot else []):
                    key=f'pilot{int(100*fraction)}_{mode}_eBH'
                    result[key][held],pilot_info[key]=pilot_calibrator(held_weighted,training_weighted,2*g,fraction)
                if pilot_selection and mode in selection_modes:
                    from .robustness_pilot_selection import apply_pilot_multiplier,selected_pilot_calibrator,MULTIPLIERS
                    for multiplier in MULTIPLIERS:
                        label=f'pilotc{multiplier}_{mode}_eBH'
                        result[label][held],pilot_selection_info[label]=apply_pilot_multiplier(
                            held_weighted,pilot_info[f'pilot80_{mode}_eBH'],2*g,multiplier)
                    label=f'pilotcv_{mode}_eBH'
                    result[label][held],pilot_selection_info[label]=selected_pilot_calibrator(
                        held_weighted,training_weighted,2*g)
                if rank_budget and mode in selection_modes:
                    from .robustness_rank_budget import rank_budget_calibrator
                    label=f'rank80_{mode}_eBH'
                    result[label][held],rank_info[label]=rank_budget_calibrator(held_weighted,training_weighted,g)
                if block:
                    capped=focused_calibrator(held_weighted,2*g,.001,0.,cap=max(1,g//4))
                    for level in [.025,.01]:
                        evidence,receipt=block_bh_evalues(held_weighted,level)
                        key=f'block{level}_{mode}_eBH'
                        result[key][held]=.2*capped+.8*evidence
                        block_info[key]=_json_value(receipt)
        predictive_info={}
        budget_info={}
        if predictive or directional:
            from .robustness_predictive_gate import simulate_predictive_e,select_predictive_gamma
            from .robustness_directional_budget import directional_budget
            cone=result['pilot80_cone_eBH'][held]
            if not fit['diagnostics']['converged']:
                for mode in (['pattern_projection','pattern_support_simes','selector'] if predictive else []):
                    result[f'predictive80_{mode}_eBH'][held]=cone
                predictive_info={'not_run':True,'reason':'upstream pattern fit did not converge',
                    'fallback_to_cone':True}
                if directional:
                    for mode in budget_modes:
                        result[f'budget80_{mode}_eBH'][held]=cone
                    budget_info={'not_run':True,'reason':'upstream pattern fit did not converge',
                        'fallback_to_unbudgeted_cone':True,'weights':[1.,1.]}
            else:
                receipts={}
                for mode,kind in [('pattern_projection','projection'),('pattern_support_simes','support_simes')]:
                    simulation=simulate_predictive_e(fit['weights'],variance[~held],all_weight[~held],shape,df,
                        profiles[mode],g,pilot_info['pilot80_cone_eBH'],pilot_info[f'pilot80_{mode}_eBH'],candidate=kind)
                    cone_sim,alternative_sim,truth_sim,receipt=simulation
                    alternative=result[f'pilot80_{mode}_eBH'][held]
                    if predictive:
                        mix,receipt=select_predictive_gamma(*simulation)
                        result[f'predictive80_{mode}_eBH'][held]=(1-mix)*cone+mix*alternative
                    if directional:
                        old_gamma=gamma[held][0]
                        variants=[(mode+'_raw',alternative_sim,alternative),
                            (mode+'_gated',(1-old_gamma)*cone_sim+old_gamma*alternative_sim,
                                (1-old_gamma)*cone+old_gamma*alternative)]
                        if mode=='pattern_projection':
                            variants.append(('cone',cone_sim,cone))
                        for label,training_simulation,held_e in variants:
                            allocation,allocation_receipt=directional_budget(training_simulation,truth_sim)
                            result[f'budget80_{label}_eBH'][held]=held_e*allocation
                            budget_info[label]={'allocation':allocation_receipt,
                                'simulation_sha256':receipt['simulation_sha256'],
                                'old_gamma':old_gamma.tolist() if label.endswith('_gated') else None,
                                'simulator_policy':'training_predictive_e_components_v1',
                                'simulation_model':receipt if not predictive else {key:value for key,value in receipt.items()
                                    if key not in ['utility_table']}}
                    receipts[mode]=receipt
                a,b=receipts['pattern_projection'],receipts['pattern_support_simes']
                if (a['simulation_sha256']!=b['simulation_sha256'] or
                        a['total_simulated_true_claims']!=b['total_simulated_true_claims']):
                    raise ValueError('Predictive experts must use identical simulated families')
                if predictive:
                    choose_simes=b['selected_total_true_positives']>=a['selected_total_true_positives']
                    selected='pattern_support_simes' if choose_simes else 'pattern_projection'
                    result['predictive80_selector_eBH'][held]=result[f'predictive80_{selected}_eBH'][held]
                    predictive_info={'experts':receipts,'selected_expert':selected,'simes_branch_selected':choose_simes,
                        'selected_gamma':receipts[selected]['selected_gamma'],
                        'selection_rule':'larger pooled predictive integer TP; ties prefer Simes',
                        'uses_held_observations':False,'uses_real_truth_labels':False,
                        'interpretation':'training-only expert choice; Simes branch is explicit simple-method fallback'}
        fits.append({'fold':fold,'held_genes':int(held.sum()),'training_genes':int((~held).sum()),
            'fit':_json_value(fit),'dominant_positive_profiles':np.asarray(dominant).tolist(),
            'average_positive_profiles':np.asarray(average).tolist(),
            'mixing_fraction_by_sign':gamma[held][0].tolist(),'pilot_calibration':pilot_info,
            'block_calibration':block_info,**({'predictive_mixing':predictive_info} if predictive else {}),
            **({'directional_budget':budget_info} if directional else {})})
        if pilot_selection:
            fits[-1]['pilot_multiplier_selection']=pilot_selection_info
            fits[-1]['full_support_gate_fraction_by_sign']=support_gamma[held][0].tolist()
        if rank_budget:
            fits[-1]['rank_budget_calibration']=rank_info
            fits[-1]['full_support_gate_fraction_by_sign']=support_gamma[held][0].tolist()
    calibrate=lambda p:focused_calibrator(p,2*g,.001,0.,cap=max(1,g//4))
    base_e=calibrate(base_weighted_pc)
    for mode in modes:
        for suffix in ['','_weighted']:
            result[f'capped_{mode}{suffix}_eBH']=calibrate(result[f'{mode}{suffix}_PC'])
    for mode in ['pattern_projection','pattern_mean_projection','pattern_bonf','pattern_support_simes']:
        target=result[f'capped_{mode}_weighted_eBH']
        result[f'{mode}_fixedmix_eBH']=.5*base_e+.5*target
        result[f'{mode}_gatedmix_eBH']=(1-gamma)*base_e+gamma*target
        if pilot:
            for fraction in [.5,.8]:
                prefix=f'pilot{int(100*fraction)}'
                alternative=result[f'{prefix}_{mode}_eBH']
                baseline=result[f'{prefix}_cone_eBH']
                result[f'{prefix}_{mode}_gatedmix_eBH']=(1-gamma)*baseline+gamma*alternative
        if block:
            for level in [.025,.01]:
                prefix=f'block{level}'
                result[f'{prefix}_{mode}_gatedmix_eBH']=(1-gamma)*result[f'{prefix}_cone_eBH']+gamma*result[f'{prefix}_{mode}_eBH']
    diagnostics['pattern_testing']={'folds':fits,'gate':'mass>max(.03,3/N); gamma=clip((dominant support fraction-.4)/.4,0,.9)',
        'fallback_signed_gene_fraction':float((gamma==0).mean()),'mean_projection_mixture_fraction':float(gamma.mean()),
        'uses_held_means_for_learning':False,'uses_truth_labels':False,
        'pilot_calibration_enabled':pilot,
        'block_calibration_enabled':block,
        'fitted_inference':'empirical; known-nuisance conditional validity does not cover arbitrary gene dependence',
        'novelty_status':'classical EB/pattern learning and positive projection ingredients; contribution not established'}
    if pilot_selection:
        for prefix in selection_prefixes:
            for mode in ['pattern_projection','pattern_support_simes']:
                result[f'{prefix}_{mode}_gatedmix_eBH']=(1-gamma)*result[f'{prefix}_cone_eBH']+gamma*result[f'{prefix}_{mode}_eBH']
                result[f'{prefix}_{mode}_support_gate_eBH']=(1-support_gamma)*result[f'{prefix}_cone_eBH']+support_gamma*result[f'{prefix}_{mode}_eBH']
    if rank_budget:
        for mode in ['pattern_projection','pattern_support_simes']:
            result[f'rank80_{mode}_gatedmix_eBH']=(1-gamma)*result['rank80_cone_eBH']+gamma*result[f'rank80_{mode}_eBH']
            result[f'rank80_{mode}_support_gate_eBH']=(1-support_gamma)*result['rank80_cone_eBH']+support_gamma*result[f'rank80_{mode}_eBH']
    return result
