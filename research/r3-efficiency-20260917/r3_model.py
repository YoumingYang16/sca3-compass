"""R3 prototype: finite-law geometric scale + independent orientation.

Theorem is model/ideal arithmetic, not a universal float certificate.
Never accepts true nuisance parameters or truth labels.
"""
import time
import sys
sys.dont_write_bytecode=True
from r3_common import R2
import numpy as np
from itertools import combinations
from predictive_bridge import rank_tail, kappa_estimator, F42_MEDIAN
from finite_calibration import (shape_fit, contrasts, learn_profiles,
                               pc_components, calibrate, positive_direction, ebh)
from grid_calibration import grid_calibrate

VERSION='R3-GSR-1.0.0'
LOG_F42_CENTER=1-np.log(2.)

def geometric_kappa(cal):
    scale=np.max(np.abs(cal),axis=(-1,-2),keepdims=True)
    if np.any(scale<=0): raise ArithmeticError('zero calibration block')
    c=cal/scale; x=c.mean(-1); y=c@contrasts(6)
    u,s,_=np.linalg.svd(y,full_matrices=False)
    if not np.isfinite(s).all() or np.any(s<=0): raise ArithmeticError('invalid calibration SVD')
    coord=np.einsum('nsi,ns->ni',u,x)
    a=.5*np.sum((coord/s)**2,axis=1)
    if not np.isfinite(a).all() or np.any(a<=0): raise ArithmeticError('nonpositive calibration pivot')
    value=float(np.exp(np.mean(np.log(a))-LOG_F42_CENTER))
    if not np.isfinite(value) or value<=0: raise ArithmeticError('invalid geometric calibration')
    return value

def references(seed,g,n,draws):
    rng=np.random.default_rng(seed)
    result={k:[] for k in ['R3_main','R3_scale','R3_median','R2_replay']}
    for start in range(0,draws,64):
        count=min(64,draws-start)
        gaussian=rng.normal(size=(count,g//2,4,5))
        full=shape_fit(gaussian,2); half=shape_fit(gaussian[:,:g//4],2)
        ev=np.linalg.eigvalsh(full); eh=np.linalg.eigvalsh(half)
        x=rng.f(4,2,size=(count,n))
        bm=np.median(x,axis=1)/F42_MEDIAN
        bg=np.exp(np.mean(np.log(x),axis=1)-LOG_F42_CENTER)
        u=rng.chisquare(5,size=(count,4)); normal=rng.normal(size=count)
        q=np.sum(u/ev,axis=1); qh=np.sum(u/eh,axis=1)
        den={
            'R3_main':bg*half[:,0,0]*qh/20,
            'R3_median':bm*half[:,0,0]*qh/20,
            'R3_scale':bg*ev[:,0]*q/20,
            'R2_replay':bm*ev[:,0]*q/20}
        for k,v in den.items():
            if not np.isfinite(v).all() or np.any(v<=0): raise ArithmeticError('invalid reference denominator')
            result[k].append(normal/np.sqrt(v))
    output={k:np.sort(np.concatenate(v)) for k,v in result.items()}
    if not all(np.isfinite(v).all() for v in output.values()): raise ArithmeticError('invalid reference draw')
    return output

def components(z,shape,kappa,profiles,reference,direction_shape):
    """Direction uses DIR shape, NEVER inference SHAPE fit."""
    scale=np.max(np.abs(z),axis=(-1,-2),keepdims=True)
    if np.any(scale<=0): raise ArithmeticError('zero target block')
    z=z/scale; y=z@contrasts(6); means=z.mean(-1)
    energy=np.einsum('gik,ij,gjk->g',y,np.linalg.inv(shape),y)
    if np.any(energy<=0): raise ArithmeticError('nonpositive target energy')
    den=np.sqrt(kappa*energy/20); result=np.zeros((len(z),2,2)); directions=[]
    for d,sign in enumerate([1,-1]):
        marginal=rank_tail(sign*means/(den[:,None]*np.sqrt(np.diag(shape))),reference)
        for subset in combinations(range(4),3):
            ids=list(subset); profile=profiles[d,ids]
            if not np.any(profile>0): profile=np.ones(3)
            a=positive_direction(profile,direction_shape[np.ix_(ids,ids)])
            statistic=sign*(means[:,ids]@a)/(den*np.sqrt(a@shape[np.ix_(ids,ids)]@a))
            result[:,d,0]=np.maximum(result[:,d,0],np.minimum(1,3*marginal[:,ids].min(1)))
            result[:,d,1]=np.maximum(result[:,d,1],rank_tail(statistic,reference))
            directions.append({'sign':sign,'subset':ids,'direction':a})
    return result,directions

def _evaluate(z,cal,seed,reference_draws,mismatch_bound):
    start=time.perf_counter(); g=len(z); m=2*g
    kg=geometric_kappa(cal); km=kappa_estimator(cal)
    refs=references(seed,g,len(cal),reference_draws)
    reference_seconds=time.perf_counter()-start
    masks=[np.arange(g)%4==f for f in range(4)]
    hs=[shape_fit(z[mask]@contrasts(6),2) for mask in masks]
    dirs=[learn_profiles(z[mask],h,km) for mask,h in zip(masks,hs)]
    pilots=[]
    for mask,h in zip(masks,hs):
        kp=kappa_estimator(z[mask]); pp,ga,ok=learn_profiles(z[mask],h,kp)
        pilots.append((pc_components(z[mask],h,kp,1.,pp),ga,ok))
    evidence={k:np.empty((g,2)) for k in ['R3_main','R3_scale','R3_ordinary','R3_median']}
    ps={k:np.empty((g,2,2)) for k in ['R3_main','R3_scale','R3_median']}; receipts=[]
    for f in range(4):
        pilot=(f+1)%4; direction=(f+2)%4; shape=(f+3)%4
        held=masks[f]; learn=~held & ~masks[pilot]
        # Primary signal family/learner unchanged; direction and inference fits
        # have separate data to justify exact orientation, not lambda_min.
        p,aa=components(z[held],hs[shape],kg*mismatch_bound,dirs[direction][0],refs['R3_main'],hs[direction])
        median_p,_=components(z[held],hs[shape],km*mismatch_bound,dirs[direction][0],refs['R3_median'],hs[direction])
        full=shape_fit(z[learn]@contrasts(6),2)
        pp,_,okfull=learn_profiles(z[learn],full,km)
        scale_p,_=components(z[held],full,kg*mismatch_bound,pp,refs['R3_scale'],full)
        gamma=pilots[pilot][1]; selections=[];norms=[]
        for tag,pvalue in [('R3_main',p),('R3_scale',scale_p),('R3_median',median_p)]:
            ps[tag][held]=pvalue; ev=[]
            for c in range(2):
                _,receipt=calibrate(pvalue[:,:,c],pilots[pilot][0][:,:,c],m,.05)
                val,grid_receipt=grid_calibrate(pvalue[:,:,c],receipt,m,reference_draws,c)
                ev.append(val)
                if tag=='R3_main': selections.append(receipt);norms.append(grid_receipt)
            evidence[tag][held]=(1-gamma)*ev[0]+gamma*ev[1]
            if tag=='R3_main': evidence['R3_ordinary'][held]=ev[0]
        receipts.append({'test_fold':f,'pilot_fold':pilot,'direction_fold':direction,'shape_fold':shape,
                         'directions':aa,'direction_profiles':dirs[direction][0],
                         'inference_shape':hs[shape],'gamma':gamma,'calibrators':selections,'grid':norms,
                         'direction_converged':dirs[direction][2],'pilot_converged':pilots[pilot][2],
                         'scale_ablation_train_converged':okfull})
    return {'version':VERSION,'status':'completed','evidence':evidence,'p':ps,'references':refs,
            'decisions':{k:ebh(v,.05) for k,v in evidence.items()},'folds':receipts,
            'kappa_geometric':kg,'kappa_median_learning_only':km,'reference_seconds':reference_seconds,
            'seconds':time.perf_counter()-start,'mismatch_bound':mismatch_bound}

def evaluate(z,cal,*,seed,reference_draws=4095,mismatch_bound=1.):
    z=np.asarray(z,float);cal=np.asarray(cal,float)
    if z.ndim!=3 or z.shape[1:]!=(4,6) or len(z)<16 or len(z)%4 or not np.isfinite(z).all():
        raise ValueError('finite Gx4x6 with G>=16 divisible by4 required')
    if cal.ndim!=3 or cal.shape[1:]!=(4,6) or len(cal)<4 or not np.isfinite(cal).all():
        raise ValueError('finite Nx4x6 independent centered calibration, N>=4 required')
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) for x in [seed,reference_draws]):
        raise ValueError('integer seed and reference count required')
    n=reference_draws+1
    if seed<0 or reference_draws<1 or n&(n-1) or n>65536: raise ValueError('invalid randomization design')
    if not np.isfinite(mismatch_bound) or mismatch_bound<1: raise ValueError('external mismatch bound>=1 required')
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            return _evaluate(z,cal,seed,reference_draws,mismatch_bound)
    except (ArithmeticError,np.linalg.LinAlgError,RuntimeError) as err:
        return {'version':VERSION,'status':'conservative_numerical_failure','error':repr(err),
                'evidence':{k:np.zeros((len(z),2)) for k in ['R3_main','R3_scale','R3_ordinary','R3_median']},
                'decisions':{k:np.zeros((len(z),2),bool) for k in ['R3_main','R3_scale','R3_ordinary','R3_median']},
                'p':{},'references':{},'folds':[],'seconds':None}
