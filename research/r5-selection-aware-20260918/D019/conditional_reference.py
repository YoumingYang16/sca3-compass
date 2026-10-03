"""A0.3 conditional full-reference prototype: ancillary information, not truth."""
import numpy as np
from r5_common import fc
from ancillary_calibration import from_bank
from codesigned_profile import CodesignedProfile
from switch_closure import SwitchClosure
from soft_drift_closure import SoftDriftClosure

class EndpointProfile:
    def __init__(self,es,et,base,a,method):
        self.method=method;self.m=len(base);self.cache={};self.c=0.;self.a=a
        logbase=np.full(len(base),-np.inf);pos=base>0;logbase[pos]=np.log(base[pos])
        logs={'target':logbase-et/2,'bridge':logbase-(a*es+(1-a)*et)/2,'source_bound':logbase-es/2}
        self.endpoint_scores=np.sort(logs['bridge' if method=='variance_bridge' else method])

    def pvalues(self,stat,borrow):
        w=np.asarray(stat,float);out=np.ones_like(w);pos=w>0
        counts=self.m-np.searchsorted(self.endpoint_scores,np.log(w[pos]),side='left')
        out[pos]=(1+counts)/(self.m+1)
        return out

def reference(seed,g,cs,ct,outer_draws=4095,inner_draws=4095,audit=False,method='closure',law_factory=from_bank):
    if method not in ['soft_closure','closure','codesigned','target','source_bound','bridge','variance_bridge']:raise ValueError('unregistered conditional rule')
    ls=None if method=='target' else law_factory(cs)
    lt=None if method=='source_bound' else law_factory(ct)
    ns=len(cs);nt=len(ct);a=ns/(ns+nt)
    seeds=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(4)]
    ms=mt=0.;meta={'used':False}
    if method in ['soft_closure','closure','codesigned','variance_bridge']:
        ss=np.random.SeedSequence(seeds[3]).spawn(2)
        sm,smr=ls.draw(np.random.default_rng(ss[0]),inner_draws)
        tm,tmr=lt.draw(np.random.default_rng(ss[1]),inner_draws)
        vs=float(sm.var(ddof=1));vt=float(tm.var(ddof=1))
        if not np.isfinite([vs,vt]).all() or min(vs,vt)<=0:raise ArithmeticError('nonpositive conditional variance estimate')
        a=vt/(vs+vt);ms=float(sm.mean());mt=float(tm.mean())
        meta={'used':True,'source_variance_MC':vs,'target_variance_MC':vt,'source_weight':a,
              'source_mean_MC':ms,'target_mean_MC':mt,'draws_each':inner_draws,'sampler_source':smr,'sampler_target':tmr}
    def tuples(seed,draws):
        randoms=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
        es,rs=(np.zeros(draws),{'unused':True}) if ls is None else ls.draw(randoms[0],draws)
        et,rt=(np.zeros(draws),{'unused':True}) if lt is None else lt.draw(randoms[1],draws)
        rng=randoms[2]
        bases=[];extra=[]
        for start in range(0,draws,64):
            n=min(64,draws-start);h=fc.shape_fit(rng.normal(size=(n,g//4,4,5)),2)
            ev=np.linalg.eigvalsh(h);u=rng.chisquare(5,size=(n,4));z=rng.normal(size=n)
            d=h[:,0,0]*np.sum(u/ev,axis=1)/20
            if np.any(d<=0) or not np.isfinite(d).all():raise ArithmeticError('conditional reference shape failure')
            bases.append(z/np.sqrt(d))
            if audit:extra.append({'H':h,'eigen':ev,'U':u,'normal':z,'denominator2':d})
        return es-ms,et-mt,np.concatenate(bases),{'source_sampler':rs,'target_sampler':rt,'shape':extra}
    if method in ['soft_closure','closure','codesigned']:
        s0,t0,v0,r0=tuples(seeds[0],inner_draws)
        s1,t1,v1,r1=tuples(seeds[1],inner_draws)
        it=np.sort(v0*np.exp(-t0/2));ib=np.sort(v1*np.exp(-(a*s1+(1-a)*t1)/2))
    else:it=ib=np.array([]);r0=r1={'unused':True}
    es,et,v,r=tuples(seeds[2],outer_draws)
    args=(es,et,v,ns,nt,it,ib)
    if method=='soft_closure':profile=SoftDriftClosure(es,et,v,a,np.sqrt(vs+vt),it,ib)
    elif method=='closure':profile=SwitchClosure(*args,source_weight=a,switch_threshold=np.sqrt(vs+vt))
    elif method=='codesigned':profile=CodesignedProfile(*args,source_weight=a)
    else:profile=EndpointProfile(es,et,v,a,method)
    return profile,{'streams':seeds,'independent_inner_draws_each':inner_draws,'outer_draws':outer_draws,
        'es':es,'et':et,'base':v,'inner_target':it,'inner_bridge':ib,
        'outer_shape_tuples':r['shape'],'ancillary_source':None if ls is None else ls.w,'ancillary_target':None if lt is None else lt.w,
        'angles_source':getattr(ls,'angles',None),'angles_target':getattr(lt,'angles',None),
        'samplers':[r0,r1,r],'method':method,'meta':meta,'centering':[ms,mt],
        'normalizer_fallback':getattr(profile,'normalizer_fallback',{'unused':True}),
        'map_fallback':getattr(profile,'map_fallback',{'unused':True}),
        'interpretation':'conditional on observed centered log-radius residuals, angles IF PROVIDED, and independent MC meta; never exact observed meanlogradius'}
