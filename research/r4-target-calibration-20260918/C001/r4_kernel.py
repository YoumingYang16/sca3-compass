"""Primary-only R3/bridge kernel; exact endpoint regression is mandatory."""
import time
import numpy as np
from r4_common import model,fc,pb,grid
from two_bank_reference import reference
from target_calibration import scale

def validate(z,cs,ct,seed,draws,weight,bound):
    if z.ndim!=3 or z.shape[1:]!=(4,6) or len(z)<16 or len(z)%4 or not np.isfinite(z).all():
        raise ValueError('finite G>=16 divisible by4 target blocks required')
    for cal in [cs]+([] if ct is None else [ct]):
        if cal.ndim!=3 or cal.shape[1:]!=(4,6) or len(cal)<4 or not np.isfinite(cal).all():
            raise ValueError('finite N>=4 calibration blocks required')
    if weight!=0 and ct is None: raise ValueError('target calibration required')
    if any(type(v) not in [int,np.int64,np.int32] for v in [seed,draws]) or seed<0 or draws<1:
        raise ValueError('integer nonnegative seed/positive draw count required')
    n=draws+1
    if n&(n-1) or n>65536: raise ValueError('unsupported reference rank grid')
    if not np.isfinite(bound) or bound<1: raise ValueError('external D>=1 required')

def evaluate_kernel(z,cs,ct,*,seed,reference_draws=4095,weight,bound=1.,audit=False):
    z=np.asarray(z,float);cs=np.asarray(cs,float);ct=None if ct is None else np.asarray(ct,float)
    validate(z,cs,ct,seed,reference_draws,weight,bound)
    start=time.perf_counter();g=len(z);m=2*g
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            k,km,cal=scale(cs,ct,weight,bound)
            ref,tuples=reference(seed,g,len(cs),0 if ct is None else len(ct),weight,reference_draws,audit)
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
                p,aa=model.components(z[held],hs[shape],k,dirs[direction][0],ref,hs[direction])
                ps[held]=p;vals=[];calibrators=[];norms=[]
                for c in range(2):
                    _,receipt=fc.calibrate(p[:,:,c],pilots[pilot][0][:,:,c],m,.05)
                    val,gr=grid.grid_calibrate(p[:,:,c],receipt,m,reference_draws,c)
                    vals.append(val);calibrators.append(receipt);norms.append(gr)
                gamma=pilots[pilot][1]
                ordinary[held]=vals[0];evidence[held]=(1-gamma)*vals[0]+gamma*vals[1]
                folds.append({'test_fold':f,'pilot_fold':pilot,'direction_fold':direction,'shape_fold':shape,
                    'directions':aa,'direction_profiles':dirs[direction][0],'inference_shape':hs[shape],
                    'gamma':gamma,'calibrators':calibrators,'grid':norms,
                    'direction_converged':dirs[direction][2],'pilot_converged':pilots[pilot][2]})
            return {'status':'completed','p':ps,'e':evidence,'ordinary_e':ordinary,'reference':ref,
                'decision':fc.ebh(evidence,.05),'ordinary_decision':fc.ebh(ordinary,.05),
                'calibration':cal,'folds':folds,'reference_seconds':ref_seconds,
                'seconds':time.perf_counter()-start,'reference_tuples':tuples}
    except (ArithmeticError,np.linalg.LinAlgError,RuntimeError) as err:
        return {'status':'conservative_numerical_failure','error':repr(err),'p':np.ones((g,2,2)),
                'e':np.zeros((g,2)),'ordinary_e':np.zeros((g,2)),
                'decision':np.zeros((g,2),bool),'ordinary_decision':np.zeros((g,2),bool),
                'reference':np.array([]),'folds':[],'calibration':{},'reference_seconds':None,
                'seconds':time.perf_counter()-start,'reference_tuples':[]}
