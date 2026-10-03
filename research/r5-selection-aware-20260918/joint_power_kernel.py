"""A0.9 direct PC-e joint moment budget. Experimental, not a released method."""
import time
from itertools import combinations
import numpy as np
from scipy.special import logsumexp
from ancillary_calibration import from_bank
from joint_power_reference import inverse_pair,positive_int
from r5_common import fc,pb,model
from r4_kernel import validate

VERSION='R5-A0.9.1-JOINT-MOMENT-AUDIT-REPAIR-DEVELOPMENT'

def fixed_meta(cs,ct,seed,n):
    ls=from_bank(cs);lt=from_bank(ct);streams=np.random.SeedSequence(seed).spawn(2)
    sm,sr=ls.draw(np.random.default_rng(streams[0]),n);tm,tr=lt.draw(np.random.default_rng(streams[1]),n)
    ms=float(sm.mean());mt=float(tm.mean());vs=float(sm.var(ddof=1));vt=float(tm.var(ddof=1))
    if min(vs,vt)<=0 or not np.isfinite([ms,mt,vs,vt]).all():raise ArithmeticError('invalid meta')
    a=vt/(vs+vt);c=np.sqrt(vs+vt);tau=c*np.sqrt(3)/np.pi
    # Arbitrary fixed-meta positive k: TWO-endpoint theorem covers estimation error.
    logk=float(logsumexp(-2*(tm-mt))-np.log(n)
        -(logsumexp(-2*a*(sm-ms))-np.log(n))
        -(logsumexp(-2*(1-a)*(tm-mt))-np.log(n)))
    return ls,lt,{'ms':ms,'mt':mt,'vs':vs,'vt':vt,'a':a,'c':float(c),'tau':float(tau),
                  'logk':logk,'draws':n,'source_sampler':sr,'target_sampler':tr}

def scalar_components(z,shape,kt,profiles,direction_shape):
    scale=np.max(np.abs(z),axis=(-1,-2),keepdims=True)
    if np.any(scale<=0):raise ArithmeticError('zero target block')
    z=z/scale;y=z@fc.contrasts(6);means=z.mean(-1)
    energy=np.einsum('gik,ij,gjk->g',y,np.linalg.inv(shape),y)
    if np.any(energy<=0):raise ArithmeticError('invalid test energy')
    den=np.sqrt(kt*energy/20);marginal=[];projected=[];directions=[]
    for d,sign in enumerate([1,-1]):
        marginal.append(sign*means/(den[:,None]*np.sqrt(np.diag(shape))))
        tt=[]
        for ids0 in combinations(range(4),3):
            ids=list(ids0);profile=profiles[d,ids]
            if not np.any(profile>0):profile=np.ones(3)
            a=fc.positive_direction(profile,direction_shape[np.ix_(ids,ids)])
            tt.append(sign*(means[:,ids]@a)/(den*np.sqrt(a@shape[np.ix_(ids,ids)]@a)))
            directions.append({'sign':sign,'subset':ids,'direction':a})
        projected.append(np.stack(tt,axis=-1))
    return np.stack(marginal,axis=1),np.stack(projected,axis=1),directions

def pc_e(marginal,projected,log_multiplier):
    def ev(x):
        values=np.zeros_like(x);pos=x>0
        values[pos]=np.exp(4*np.log(x[pos])+log_multiplier)
        if not np.isfinite(values).all():raise ArithmeticError('nonfinite power e evidence')
        return values
    em=ev(marginal);ep=ev(projected)
    ordinary=np.sort(em,axis=-1)[...,:3].mean(-1)
    projected_pc=ep.min(-1)
    return ordinary,projected_pc

def evaluate(z,cs,ct,*,bound,seed,successes=64,meta_draws=4095,numerator_draws=1024,proposal_cap=1000000):
    z=np.asarray(z,float);cs=np.asarray(cs,float);ct=np.asarray(ct,float)
    validate(z,cs,ct,seed,63,1.,bound)
    successes=positive_int(successes,'successes');meta_draws=positive_int(meta_draws,'meta_draws',2)
    numerator_draws=positive_int(numerator_draws,'numerator_draws',2);proposal_cap=positive_int(proposal_cap,'proposal_cap',successes)
    start=time.perf_counter();g=len(z)
    streams=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(2)]
    stage='meta'
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            ls,lt,meta=fixed_meta(cs,ct,streams[0],meta_draws)
            stage='reference'
            logr,receipt=inverse_pair(ls.w,lt.w,meta['ms'],meta['mt'],meta['a'],meta['c'],meta['tau'],meta['logk'],g,streams[1],
                successes=successes,numerator_draws=numerator_draws,cap=proposal_cap)
            stage='observed_scale_and_folds'
            lks=np.log(model.geometric_kappa(cs))-meta['ms'];lkt=np.log(model.geometric_kappa(ct))-meta['mt']
            u=np.log(bound)+lks-lkt;zz=(u-meta['c'])/meta['tau']
            logg=-np.logaddexp(0.,zz);lognot=-np.logaddexp(0.,-zz)
            lograw=float(np.logaddexp(lognot,meta['logk']-2*meta['a']*u+logg))
            logmult=lograw+logr;kt=np.exp(lkt)
            masks=[np.arange(g)%4==f for f in range(4)];km=pb.kappa_estimator(ct)
            hs=[fc.shape_fit(z[m]@fc.contrasts(6),2) for m in masks]
            dirs=[fc.learn_profiles(z[m],h,km) for m,h in zip(masks,hs)]
            pilots=[fc.learn_profiles(z[m],h,pb.kappa_estimator(z[m])) for m,h in zip(masks,hs)]
            evidence=np.empty((g,2));pc=np.empty((g,2,2));raw_e_power=np.empty((g,2));folds=[]
            for f in range(4):
                stage='test_fold_'+str(f)
                pi=(f+1)%4;di=(f+2)%4;sh=(f+3)%4;held=masks[f]
                marginal,projected,aa=scalar_components(z[held],hs[sh],kt,dirs[di][0],hs[di])
                eo,ep=pc_e(marginal,projected,logmult);gamma=pilots[pi][1]
                raw_o,raw_p=pc_e(marginal,projected,0.)
                raw_e_power[held]=(1-gamma)*raw_o+gamma*raw_p
                evidence[held]=(1-gamma)*eo+gamma*ep;pc[held]=np.stack([eo,ep],axis=-1)
                folds.append({'test':f,'pilot':pi,'direction':di,'shape':sh,'gamma':gamma,'directions':aa,
                    'inference_shape':hs[sh],'direction_profiles':dirs[di][0],
                    'direction_converged':dirs[di][2],'pilot_converged':pilots[pi][2]})
            return {'status':'completed','version':VERSION,'e':evidence,'decision':fc.ebh(evidence,.05),'pc_e':pc,'raw_e_power':raw_e_power,
                'p_kind':'NO_P_VALUES_DIRECT_E_BUDGET_NOT_RANK_CALIBRATION','meta':meta,'reference':receipt,
                'calibration':{'centered_u':float(u),'bound':bound,'log_kt':float(lkt),'log_raw_multiplier':lograw,
                    'log_total_multiplier':float(logmult),'log_borrow_probability':float(logg)},
                'folds':folds,'seconds':time.perf_counter()-start,'streams':streams,
                'acceptance':'DEVELOPMENT_ONLY_ALL_GATES_OPEN'}
    except (ArithmeticError,RuntimeError,np.linalg.LinAlgError) as e:
        return {'status':'conservative_numerical_failure','version':VERSION,'error':repr(e),'failure_receipt':{'stage':stage,'detail':getattr(e,'receipt',None)},
            'e':np.zeros((g,2)),'decision':np.zeros((g,2),bool),'seconds':time.perf_counter()-start,
            'failure_policy':'WHOLE_FAMILY_ZERO_NO_REFERENCE_RETRY','acceptance':'NOT_ACCEPTED'}
