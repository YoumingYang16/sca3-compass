"""Branch-conditional joint reference; full pairs reselected, not old fixed law."""
import numpy as np
from scipy.special import expit
from ancillary_calibration import from_bank
from conditional_reference import EndpointProfile
from selection_profile import select
from r5_common import fc,model

class SelectedPairCapError(RuntimeError):
    def __init__(self,receipt):
        super().__init__('selected-pair cap: fail closed, never use unconditional substitute')
        self.receipt=receipt

def selected_errors(ls,lt,ms,mt,c,borrow,rngs,size,cap=None,soft_tau=None):
    cap=max(10000,100*size) if cap is None else cap
    ss=[];ts=[];used=accepted=0;receipts=[]
    while accepted<size and used<cap:
        n=min(4096,max(64,2*(size-accepted)),cap-used)
        es,rs=ls.draw(rngs[0],n);et,rt=lt.draw(rngs[1],n);es=es-ms;et=et-mt
        if soft_tau is None:chosen=es-et<=c
        else:
            if not np.isfinite(soft_tau) or soft_tau<=0:raise ValueError('positive fixed smoothing scale')
            chosen=rngs[2].random(n)<expit((c-es+et)/soft_tau)
        keep=chosen==borrow
        ss.append(es[keep]);ts.append(et[keep]);accepted+=int(keep.sum());used+=n
        receipts.append({'source':rs,'target':rt})
    if accepted<size:raise SelectedPairCapError({'attempted_pairs':used,'accepted_before_truncation':accepted,
        'required':size,'cap':cap,'branch':bool(borrow),'soft_tau':soft_tau,'samplers':receipts})
    return np.concatenate(ss)[:size],np.concatenate(ts)[:size],{
        'attempted_pairs':used,'accepted_before_truncation':accepted,'retained':size,
        'branch':bool(borrow),'rate':accepted/used,'samplers':receipts,'soft_tau':soft_tau}

def reference(seed,g,cs,ct,bound,outer_draws=4095,meta_draws=4095,audit=False,soft_branch=None,
              law_factory=from_bank,estimator=model.geometric_kappa):
    ls=law_factory(cs);lt=law_factory(ct)
    seeds=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(4)]
    # Preserve the former meta stream for an interpretable paired revision.
    ssm=np.random.SeedSequence(seeds[3]).spawn(2)
    sm,smr=ls.draw(np.random.default_rng(ssm[0]),meta_draws)
    tm,tmr=lt.draw(np.random.default_rng(ssm[1]),meta_draws)
    vs=float(sm.var(ddof=1));vt=float(tm.var(ddof=1));ms=float(sm.mean());mt=float(tm.mean())
    if not np.isfinite([vs,vt,ms,mt]).all() or min(vs,vt)<=0:raise ArithmeticError('invalid meta moments')
    a=vt/(vs+vt);c=float(np.sqrt(vs+vt))
    choice=select(np.log(estimator(cs))-ms,np.log(estimator(ct))-mt,
                  len(cs),len(ct),bound,c,a)
    soft=soft_branch is not None;tau=c*np.sqrt(3)/np.pi if soft else None
    probability=float(expit((c-choice['observed_slack'])/tau)) if soft else None
    if soft:
        choice['borrow']=bool(soft_branch);choice['target_weight']=1-a if soft_branch else 1.
        lkt=np.log(estimator(ct))-mt
        choice['log_scale']=lkt+a*choice['observed_slack'] if soft_branch else lkt
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seeds[2]).spawn(4 if soft else 3)]
    pair_rngs=streams[:2]+([streams[3]] if soft else [])
    es,et,rec=selected_errors(ls,lt,ms,mt,c,choice['borrow'],pair_rngs,outer_draws,soft_tau=tau)
    rng=streams[2];bases=[];extra=[]
    for start in range(0,outer_draws,64):
        n=min(64,outer_draws-start);h=fc.shape_fit(rng.normal(size=(n,g//4,4,5)),2)
        ev=np.linalg.eigvalsh(h);u=rng.chisquare(5,size=(n,4));z=rng.normal(size=n)
        d=h[:,0,0]*np.sum(u/ev,axis=1)/20
        if np.any(d<=0) or not np.isfinite(d).all():raise ArithmeticError('selective shape tuple failure')
        bases.append(z/np.sqrt(d))
        if audit:extra.append({'H':h,'eigen':ev,'U':u,'normal':z,'denominator2':d})
    base=np.concatenate(bases)
    profile=EndpointProfile(es,et,base,a,'bridge' if choice['borrow'] else 'target')
    profile.c=c;profile.borrow_observed=choice['borrow']
    meta={'used':True,'source_variance_MC':vs,'target_variance_MC':vt,'source_weight':a,
          'source_mean_MC':ms,'target_mean_MC':mt,'draws_each':meta_draws,
          'sampler_source':smr,'sampler_target':tmr}
    return profile,{'streams':seeds,'unused_streams':seeds[:2],'inner_target':np.array([]),'inner_bridge':np.array([]),
        'independent_inner_draws_each':0,'outer_draws':outer_draws,'es':es,'et':et,'base':base,
        'outer_shape_tuples':extra,'ancillary_source':ls.w,'ancillary_target':lt.w,'samplers':[rec],
        'angles_source':getattr(ls,'angles',None),'angles_target':getattr(lt,'angles',None),
        'method':'selective_branch','meta':meta,'centering':[ms,mt],'selection':choice,
        'borrow_probability':probability,'soft_tau':tau,
        'conditional_pair_sampling':rec,'interpretation':'conditional on W, independent meta, and observed branch; all outer pairs reselected at s=0'}
