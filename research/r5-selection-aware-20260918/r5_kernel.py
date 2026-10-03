"""R5-A0 full-chain DEVELOPMENT prototype; all four acceptance gates OPEN.

Observational inputs exactly Z,Cs,Ct,D. No truth, true drift, true shape or pilot.
The outer profile is applied to SINGLE tests before ordinary/projected PC.
"""
import time
from itertools import combinations
import numpy as np
from r5_common import fc,pb,model,grid
from r4_kernel import validate
from selection_profile import select
from conditional_reference import reference
from selective_reference import reference as selected_reference

VERSION='R5-A0.8-SOFT-DRIFT-CLOSURE-DEVELOPMENT'

def components(z,shape,kappa,profiles,ref,direction_shape,borrow):
    scale=np.max(np.abs(z),axis=(-1,-2),keepdims=True)
    if np.any(scale<=0):raise ArithmeticError('zero target block')
    z=z/scale;y=z@fc.contrasts(6);means=z.mean(-1)
    energy=np.einsum('gik,ij,gjk->g',y,np.linalg.inv(shape),y)
    if np.any(energy<=0):raise ArithmeticError('nonpositive target energy')
    den=np.sqrt(kappa*energy/20);result=np.zeros((len(z),2,2));directions=[]
    for d,sign in enumerate([1,-1]):
        marginal=ref.pvalues(sign*means/(den[:,None]*np.sqrt(np.diag(shape))),borrow)
        for subset in combinations(range(4),3):
            ids=list(subset);profile=profiles[d,ids]
            if not np.any(profile>0):profile=np.ones(3)
            a=fc.positive_direction(profile,direction_shape[np.ix_(ids,ids)])
            statistic=sign*(means[:,ids]@a)/(den*np.sqrt(a@shape[np.ix_(ids,ids)]@a))
            result[:,d,0]=np.maximum(result[:,d,0],np.minimum(1,3*marginal[:,ids].min(1)))
            result[:,d,1]=np.maximum(result[:,d,1],ref.pvalues(statistic,borrow))
            directions.append({'sign':sign,'subset':ids,'direction':a})
    return result,directions

def evaluate(z,cs,ct,*,bound,seed,reference_draws=4095,inner_draws=4095,audit=False,method='selective',calibration_law='radial',action_index=None,direction_calibration='as_defined'):
    z=np.asarray(z,float);cs=np.asarray(cs,float);ct=np.asarray(ct,float)
    validate(z,cs,ct,seed,reference_draws,1.,bound)
    if type(inner_draws) not in [int,np.int64,np.int32] or inner_draws<1:raise ValueError('positive inner draws')
    if method not in ['action_component','soft_closure','soft_bridge','soft_target','selective','closure','codesigned','target','source_bound','bridge','variance_bridge']:raise ValueError('unregistered conditional rule')
    if calibration_law not in ['radial','triangular']:raise ValueError('unregistered calibration law')
    if direction_calibration not in ['as_defined','target']:raise ValueError('unregistered direction calibration')
    if method=='action_component' and (calibration_law!='triangular' or action_index not in [0,1,2]):raise ValueError('action requires registered triangular setup')
    lawkw={};selectedkw={};estimator=model.geometric_kappa
    if calibration_law=='triangular':
        from triangular_calibration import from_bank as triangular_law,geometric_kappa as estimator
        lawkw={'law_factory':triangular_law};selectedkw={**lawkw,'estimator':estimator}
    version=VERSION if calibration_law=='radial' else 'R5-A0.13-TRIANGULAR-CALIBRATION-DEVELOPMENT'
    start=time.perf_counter();g=len(z);m=2*g
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            ks=None if method=='target' else estimator(cs)
            kt=None if method=='source_bound' else estimator(ct)
            if method=='action_component':
                from action_reference import reference as action_reference
                ref,receipt=action_reference(seed,g,cs,ct,bound,reference_draws,inner_draws,audit,action_index)
                version='R5-A0.14-ACTION-SELECTIVE-DEVELOPMENT'
            elif method in ['soft_bridge','soft_target']:ref,receipt=selected_reference(seed,g,cs,ct,bound,reference_draws,inner_draws,audit,soft_branch=method=='soft_bridge',**selectedkw)
            elif method=='selective':ref,receipt=selected_reference(seed,g,cs,ct,bound,reference_draws,inner_draws,audit,**selectedkw)
            else:ref,receipt=reference(seed,g,cs,ct,reference_draws,inner_draws,audit,method,**lawkw)
            ms,mt=receipt['centering']
            lks=None if ks is None else np.log(ks)-ms;lkt=None if kt is None else np.log(kt)-mt
            if method=='soft_closure':
                choice=select(lks,lkt,len(cs),len(ct),bound,ref.c,ref.a)
                ref.observed_u=choice['observed_slack']
                choice.update(log_scale=lkt,method=method,target_weight=1.,source_weight_in_score=ref.a,
                    scale_interpretation='target-normalized; borrowing happens inside full-rule score, not selected denominator')
            elif method in ['action_component','soft_bridge','soft_target']:choice=receipt['selection']
            elif method in ['selective','closure','codesigned']:choice=select(lks,lkt,len(cs),len(ct),bound,ref.c,ref.a)
            else:
                weight={'target':1.,'source_bound':0.,'bridge':len(ct)/(len(cs)+len(ct)),
                        'variance_bridge':1-ref.a}[method]
                choice={'borrow':method!='target','target_weight':weight,'method':method,
                        'log_scale':lkt if method=='target' else np.log(bound)+lks if method=='source_bound'
                            else (1-weight)*(np.log(bound)+lks)+weight*lkt}
            direction_bank=cs if method=='source_bound' and direction_calibration=='as_defined' else ct
            k=np.exp(choice['log_scale']);km=pb.kappa_estimator(direction_bank)
            if method=='selective' and choice['borrow']!=ref.borrow_observed:raise ArithmeticError('observed/reference branch mismatch')
            if not np.isfinite(k) or k<=0:raise ArithmeticError('selected scale invalid')
            ref_seconds=time.perf_counter()-start
            masks=[np.arange(g)%4==f for f in range(4)]
            hs=[fc.shape_fit(z[mask]@fc.contrasts(6),2) for mask in masks]
            dirs=[fc.learn_profiles(z[mask],h,km) for mask,h in zip(masks,hs)]
            pilots=[]
            for mask,h in zip(masks,hs):
                kp=pb.kappa_estimator(z[mask]);pp,ga,ok=fc.learn_profiles(z[mask],h,kp)
                pilots.append((fc.pc_components(z[mask],h,kp,1.,pp),ga,ok))
            evidence=np.empty((g,2));ordinary=np.empty((g,2));ps=np.empty((g,2,2));folds=[]
            for f in range(4):
                pilot=(f+1)%4;direction=(f+2)%4;shape=(f+3)%4;held=masks[f]
                p,aa=components(z[held],hs[shape],k,dirs[direction][0],ref,hs[direction],choice['borrow'])
                ps[held]=p;vals=[];calibrators=[];norms=[]
                for c in range(2):
                    _,rec=fc.calibrate(p[:,:,c],pilots[pilot][0][:,:,c],m,.05)
                    val,gr=grid.grid_calibrate(p[:,:,c],rec,m,reference_draws,c)
                    vals.append(val);calibrators.append(rec);norms.append(gr)
                gamma=pilots[pilot][1];ordinary[held]=vals[0]
                evidence[held]=(1-gamma)*vals[0]+gamma*vals[1]
                folds.append({'test_fold':f,'pilot_fold':pilot,'direction_fold':direction,'shape_fold':shape,
                    'directions':aa,'direction_profiles':dirs[direction][0],'inference_shape':hs[shape],
                    'gamma':gamma,'calibrators':calibrators,'grid':norms,
                    'direction_converged':dirs[direction][2],'pilot_converged':pilots[pilot][2]})
            return {'status':'completed','version':version,'calibration_law':calibration_law,'p':ps,'e':evidence,'ordinary_e':ordinary,
                'validity_scope':'CONDITIONAL_COMPONENT_ONLY' if method in ['action_component','soft_bridge','soft_target'] else 'M(D)_MODEL_METHOD',
                'decision_interpretation':'DIAGNOSTIC_ONLY_UNLESS_MATCHING_RANDOMIZED_SELECTION' if method in ['action_component','soft_bridge','soft_target'] else 'method_decision',
                'decision':fc.ebh(evidence,.05),'ordinary_decision':fc.ebh(ordinary,.05),
                'calibration':{**choice,'source_geometric':ks,'target_geometric':kt,'learning_median':km,
                    'external_bound':bound,'direction_calibration':direction_calibration},'folds':folds,'reference_seconds':ref_seconds,
                'seconds':time.perf_counter()-start,'reference_tuples':receipt,'profile_counts':ref.cache}
    except (ArithmeticError,np.linalg.LinAlgError,RuntimeError) as err:
        return {'status':'conservative_numerical_failure','version':version,'calibration_law':calibration_law,'error':repr(err),
                'failure_receipt':getattr(err,'receipt',None),
                'p':np.ones((g,2,2)),'e':np.zeros((g,2)),'ordinary_e':np.zeros((g,2)),
                'decision':np.zeros((g,2),bool),'ordinary_decision':np.zeros((g,2),bool),
                'folds':[],'calibration':{},'profile_counts':{},'reference_tuples':{},
                'reference_seconds':None,'seconds':time.perf_counter()-start}
