"""Independent inner normalizers and complete outer selection-reference tuples."""
import numpy as np
from r5_common import fc,fixed_reference
from selection_profile import Profile,draw_errors
from continuous_profile import ContinuousProfile
from codesigned_profile import CodesignedProfile

def reference(seed,g,ns,nt,outer_draws=4095,inner_draws=4095,audit=False,normalization='codesigned'):
    if normalization not in ['rank','continuous','codesigned']:raise ValueError('unregistered inner map')
    streams=np.random.SeedSequence(seed).spawn(3)
    seeds=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in streams]
    it,_=fixed_reference(seeds[0],g,ns,nt,1.,inner_draws,False)
    ib,_=fixed_reference(seeds[1],g,ns,nt,nt/(ns+nt),inner_draws,False)
    rng=np.random.default_rng(seeds[2]);es=[];et=[];base=[];receipts=[]
    for start in range(0,outer_draws,64):
        n=min(64,outer_draws-start)
        gaussian=rng.normal(size=(n,g//4,4,5))
        h=fc.shape_fit(gaussian,2);eigen=np.linalg.eigvalsh(h)
        xs=draw_errors(rng,ns,n);xt=draw_errors(rng,nt,n)
        u=rng.chisquare(5,size=(n,4));normal=rng.normal(size=n)
        den=h[:,0,0]*np.sum(u/eigen,axis=1)/20
        if not np.isfinite(den).all() or np.any(den<=0):raise ArithmeticError('outer shape reference failure')
        v=normal/np.sqrt(den);es.append(xs);et.append(xt);base.append(v)
        if audit:receipts.append({'H':h,'eigen':eigen,'U':u,'normal':normal,'denominator2':den})
    cls={'rank':Profile,'continuous':ContinuousProfile,'codesigned':CodesignedProfile}[normalization]
    profile=cls(np.concatenate(es),np.concatenate(et),np.concatenate(base),ns,nt,it,ib)
    return profile,{'streams':seeds,'independent_inner_draws_each':inner_draws,'outer_draws':outer_draws,
        'es':profile.es,'et':profile.et,'base':profile.base,'inner_target':it,'inner_bridge':ib,
        'outer_shape_tuples':receipts}
