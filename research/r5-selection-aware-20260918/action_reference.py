"""Finite three-action selective reference, K1, DEVELOPMENT ONLY."""
import numpy as np
from scipy.special import softmax
from r5_common import fc
from triangular_calibration import from_bank,geometric_kappa
from conditional_reference import EndpointProfile
from selective_reference import SelectedPairCapError


def probabilities(u,c,tau):
    if not np.isfinite([c,tau]).all() or c<=0 or tau<=0:raise ValueError('fixed positive gate scales')
    u=np.asarray(u,float)
    # Target, variance bridge, source. Log weights are all concave in u.
    return softmax(np.stack([(u-c)/tau,np.zeros_like(u),(-u-c)/tau],axis=-1),axis=-1)


def selected_errors(ls,lt,ms,mt,c,tau,j,rngs,size,cap=None):
    if j not in [0,1,2]:raise ValueError('finite registered action')
    cap=max(10000,100*size) if cap is None else int(cap)
    ss=[];ts=[];used=accepted=0;receipts=[]
    while accepted<size and used<cap:
        n=min(4096,max(64,2*(size-accepted)),cap-used)
        x,rx=ls.draw(rngs[0],n);y,ry=lt.draw(rngs[1],n);x-=ms;y-=mt
        keep=rngs[2].random(n)<probabilities(x-y,c,tau)[:,j]
        ss.append(x[keep]);ts.append(y[keep]);used+=n;accepted+=int(keep.sum())
        receipts.append({'source':rx,'target':ry})
    rec={'attempted_pairs':used,'accepted_before_truncation':accepted,'requested':size,
         'retained':min(accepted,size),'returned':size if accepted>=size else 0,
         'cap':cap,'action':j,'samplers':receipts,'rate':accepted/used if used else 0}
    if accepted<size:raise SelectedPairCapError(rec)
    return np.concatenate(ss)[:size],np.concatenate(ts)[:size],rec


def reference(seed,g,cs,ct,bound,outer_draws=4095,meta_draws=4095,audit=False,action_index=0):
    if action_index not in [0,1,2]:raise ValueError('action index')
    ls=from_bank(cs);lt=from_bank(ct)
    seeds=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(4)]
    meta_seeds=np.random.SeedSequence(seeds[3]).spawn(2)
    sm,rs=ls.draw(np.random.default_rng(meta_seeds[0]),meta_draws)
    tm,rt=lt.draw(np.random.default_rng(meta_seeds[1]),meta_draws)
    vs=float(sm.var(ddof=1));vt=float(tm.var(ddof=1));ms=float(sm.mean());mt=float(tm.mean())
    if not np.isfinite([vs,vt,ms,mt]).all() or min(vs,vt)<=0:raise ArithmeticError('invalid meta')
    a=vt/(vs+vt);c=float(np.sqrt(vs+vt));tau=c*np.sqrt(3)/np.pi
    weights=[0.,a,1.];lam=weights[action_index]
    lks=np.log(geometric_kappa(cs))-ms;lkt=np.log(geometric_kappa(ct))-mt
    u=np.log(bound)+lks-lkt;probs=probabilities(u,c,tau)
    choice={'observed_slack':float(u),'borrow':action_index!=0,'target_weight':1-lam,
            'source_weight':lam,'action_index':action_index,'log_scale':float(lkt+lam*u),
            'method':'action_component','switch_threshold':c,'action_probabilities':probs.tolist()}
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seeds[2]).spawn(4)]
    es,et,rec=selected_errors(ls,lt,ms,mt,c,tau,action_index,[streams[0],streams[1],streams[3]],outer_draws)
    rng=streams[2];bases=[];extra=[]
    for start in range(0,outer_draws,64):
        n=min(64,outer_draws-start);h=fc.shape_fit(rng.normal(size=(n,g//4,4,5)),2)
        ev=np.linalg.eigvalsh(h);u2=rng.chisquare(5,size=(n,4));z=rng.normal(size=n)
        d=h[:,0,0]*np.sum(u2/ev,axis=1)/20
        if np.any(d<=0) or not np.isfinite(d).all():raise ArithmeticError('shape reference failure')
        bases.append(z/np.sqrt(d))
        if audit:extra.append({'H':h,'eigen':ev,'U':u2,'normal':z,'denominator2':d})
    base=np.concatenate(bases);profile=EndpointProfile(es,et,base,lam,'bridge');profile.c=c
    return profile,{'streams':seeds,'outer_draws':outer_draws,'es':es,'et':et,'base':base,
        'inner_target':np.array([]),'inner_bridge':np.array([]),'outer_shape_tuples':extra,
        'ancillary_source':ls.w,'ancillary_target':lt.w,'angles_source':ls.angles,'angles_target':lt.angles,
        'centering':[ms,mt],'selection':choice,'action_probabilities':probs,'action_weights':weights,
        'action_index':action_index,'soft_tau':tau,'conditional_pair_sampling':rec,
        'meta':{'source_variance_MC':vs,'target_variance_MC':vt,'source_mean_MC':ms,'target_mean_MC':mt,
                'source_weight':a,'draws_each':meta_draws,'sampler_source':rs,'sampler_target':rt},
        'interpretation':'entire pairs reselected using K1 action kernel at boundary s0; component-only'}
