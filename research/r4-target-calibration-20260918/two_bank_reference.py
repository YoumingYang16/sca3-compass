"""Exact R3 main-reference stream, with unused ablation shape fits removed.

The full Gaussian array is drawn to preserve frozen R3 RNG consumption.
Only H fitted on its first G/4 blocks enters the primary. Removing unused
full-H computations changes neither its inputs nor any random draw.
"""
import numpy as np
from r4_common import fc, model

def reference(seed, g, ns, nt, weight, draws=4095, audit=False):
    if weight==0: n=ns
    elif weight==1: n=nt
    elif weight==nt/(ns+nt): n=ns+nt
    else: raise ValueError('only two endpoints and prespecified sample-count bridge')
    if n<4: raise ValueError('at least four calibration blocks at active endpoint')
    rng=np.random.default_rng(seed); values=[]; receipts=[]
    for start in range(0,draws,64):
        count=min(64,draws-start)
        gaussian=rng.normal(size=(count,g//2,4,5))
        h=fc.shape_fit(gaussian[:,:g//4],2)
        eigen=np.linalg.eigvalsh(h)
        x=rng.f(4,2,size=(count,n))
        b=np.exp(np.mean(np.log(x),axis=1)-model.LOG_F42_CENTER)
        u=rng.chisquare(5,size=(count,4)); normal=rng.normal(size=count)
        den=b*h[:,0,0]*np.sum(u/eigen,axis=1)/20
        if np.any(den<=0) or not np.isfinite(den).all():
            raise ArithmeticError('nonpositive/nonfinite reference denominator')
        values.append(normal/np.sqrt(den))
        if audit:
            receipts.append({'H':h,'eigen':eigen,'B':b,'U':u,'normal':normal,'denominator2':den})
    value=np.sort(np.concatenate(values))
    if not np.isfinite(value).all(): raise ArithmeticError('invalid reference; no draws dropped')
    return value,receipts
