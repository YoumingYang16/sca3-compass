"""A0.10 continuum-protected focused joint MC-e; development only."""
import time
import numpy as np
from scipy.special import expit
from r5_common import fc,pb,model
from r4_kernel import validate
from joint_power_kernel import scalar_components
from focused_reference import make_reference,mc_e

VERSION='R5-A0.10.1-FOCUSED-JOINT-MC-AUDIT-REPAIR'
METHODS=['focused_joint','focused_target','focused_source_bound','focused_count_bridge',
         'focused_variance_bridge','focused_fixed_mix','focused_variance_mix','focused_fixed_joint_score']

def indicators(t,u,a,threshold):
    values=np.asarray(t,float);out=np.zeros(values.shape,bool);positive=values>0
    shifts=np.broadcast_to(np.asarray(u,float),values.shape)
    out[positive]=np.log(values[positive])-a*shifts[positive]/2>np.log(threshold)
    return out.astype(float)

def scalar_e(t,u,ref):
    n=ref['outer_draws'];q=ref['thresholds'];m=ref['meta'];g=float(expit((m['c']-u)/m['tau']))
    scores={name:indicators(t,u,a,q[name]) for name,a in ref['weights'].items()}
    result={'focused_'+name:mc_e(scores[name],ref['totals'][name],n) for name in scores}
    selected=g*scores['variance_bridge']+(1-g)*scores['target']
    result['focused_joint']=mc_e(selected,ref['Q'],n)
    result['focused_fixed_joint_score']=mc_e(.5*scores['target']+.5*scores['count_bridge'],
                                            .5*(ref['totals']['target']+ref['totals']['count_bridge']),n)
    return result

def evaluate(z,cs,ct,*,bound,seed,outer_draws=8191,meta_draws=4095):
    z=np.asarray(z,float);cs=np.asarray(cs,float);ct=np.asarray(ct,float)
    validate(z,cs,ct,seed,outer_draws,1.,bound);start=time.perf_counter();stage='reference';g=len(z)
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            ref=make_reference(cs,ct,g,seed,outer_draws=outer_draws,meta_draws=meta_draws)
            stage='observed_scales'
            m=ref['meta'];lks=np.log(model.geometric_kappa(cs))-m['ms'];lkt=np.log(model.geometric_kappa(ct))-m['mt']
            u=float(np.log(bound)+lks-lkt);kt=float(np.exp(lkt))
            masks=[np.arange(g)%4==f for f in range(4)];km=pb.kappa_estimator(ct)
            hs=[fc.shape_fit(z[mask]@fc.contrasts(6),2) for mask in masks]
            ds=[fc.learn_profiles(z[mask],h,km) for mask,h in zip(masks,hs)]
            ps=[fc.learn_profiles(z[mask],h,pb.kappa_estimator(z[mask])) for mask,h in zip(masks,hs)]
            evidence={name:np.empty((g,2)) for name in METHODS if name not in ['focused_fixed_mix','focused_variance_mix']}
            all_marginal=np.empty((g,2,4));all_projected=np.empty((g,2,4));folds=[]
            for f in range(4):
                stage='fold_'+str(f);pi=(f+1)%4;di=(f+2)%4;sh=(f+3)%4;held=masks[f];gamma=ps[pi][1]
                marg,proj,dirs=scalar_components(z[held],hs[sh],kt,ds[di][0],hs[di])
                all_marginal[held]=marg;all_projected[held]=proj
                em=scalar_e(marg,u,ref);ep=scalar_e(proj,u,ref)
                for name in evidence:
                    ordinary=np.sort(em[name],axis=-1)[...,:3].mean(-1);projected=ep[name].min(-1)
                    evidence[name][held]=(1-gamma)*ordinary+gamma*projected
                folds.append({'test':f,'pilot':pi,'direction':di,'shape':sh,'gamma':gamma,'directions':dirs,
                              'inference_shape':hs[sh],'direction_profiles':ds[di][0],
                              'direction_converged':ds[di][2],'pilot_converged':ps[pi][2]})
            evidence['focused_fixed_mix']=.5*evidence['focused_target']+.5*evidence['focused_count_bridge']
            evidence['focused_variance_mix']=.5*evidence['focused_target']+.5*evidence['focused_variance_bridge']
            if not all(np.isfinite(e).all() for e in evidence.values()):raise ArithmeticError('nonfinite focused evidence')
            return {'status':'completed','version':VERSION,'evidence':evidence,
                'decisions':{k:fc.ebh(e,.05) for k,e in evidence.items()},'reference':ref,
                'statistics':{'marginal':all_marginal,'projected':all_projected},'folds':folds,
                'calibration':{'observed_u':u,'log_target_kappa':float(lkt),'bound':bound,
                               'borrow_probability':float(expit((m['c']-u)/m['tau']))},
                'seconds':time.perf_counter()-start,'acceptance':'ALL_GATES_OPEN_NO_NOVELTY_OR_UTILITY_ACCEPTANCE'}
    except (ArithmeticError,RuntimeError,ValueError,np.linalg.LinAlgError) as err:
        return {'status':'conservative_failure','version':VERSION,'stage':stage,'error':repr(err),
                'receipt':getattr(err,'receipt',None),'seconds':time.perf_counter()-start,
                'evidence':{name:np.zeros((g,2)) for name in METHODS},
                'decisions':{name:np.zeros((g,2),bool) for name in METHODS},
                'failure_policy':'WHOLE_FAMILY_ZERO_ALL_ENDPOINTS_NO_RETRY'}
