"""Conditional calibration information diagnostic; NOT a certified R5 method.

Y=mean(logF42) given centered within-bank log residuals has an exact nuisance-
free, log-concave density. Three-tangent accept/reject samples it without MCMC.
Conditioning/ARS are classical; independent new content remains unresolved.
"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit,logsumexp
from r5_common import fc
from selection_profile import LOG_CENTER

def log_pivots(cal):
    cal=np.asarray(cal,float)
    if cal.ndim!=3 or cal.shape[1:]!=(4,6) or len(cal)<4 or not np.isfinite(cal).all():
        raise ValueError('finite bank N>=4 required')
    mx=np.max(np.abs(cal),axis=(-1,-2),keepdims=True)
    if np.any(mx<=0):raise ArithmeticError('zero calibration block')
    c=cal/mx;x=c.mean(-1);y=c@fc.contrasts(6)
    u,s,_=np.linalg.svd(y,full_matrices=False)
    if np.any(s<=0):raise ArithmeticError('singular calibration')
    coord=np.einsum('nsi,ns->ni',u,x);A=.5*np.sum((coord/s)**2,axis=1)
    if np.any(A<=0) or not np.isfinite(A).all():raise ArithmeticError('bad pivot')
    return np.log(A)

class ConditionalLogF:
    def __init__(self,residuals):
        w=np.asarray(residuals,float)
        if w.ndim!=1 or len(w)<4 or not np.isfinite(w).all():raise ValueError('finite centered residuals N>=4')
        if abs(w.mean())>1e-10*(1+np.max(np.abs(w))):raise ValueError('residuals must be centered, not nuisance truths')
        self.w=w;self.n=len(w)
        lo=-float(w.max())-4;hi=-float(w.min())+4
        self.mode=float(brentq(self.derivative,lo,hi,xtol=1e-13))
        curvature=3*np.sum(expit(self.mode+w+np.log(2))*(1-expit(self.mode+w+np.log(2))))
        step=1/np.sqrt(max(curvature,1.))
        for _ in range(64):
            if self.derivative(self.mode-step)>=.5 and self.derivative(self.mode+step)<=-.5:break
            step*=2
        else:raise ArithmeticError('cannot bracket integrable conditional envelope')
        self.left=self.mode-step;self.right=self.mode+step
        self.sl=float(self.derivative(self.left));self.sr=float(self.derivative(self.right))
        if not self.sl>0>self.sr:raise ArithmeticError('nonintegrable rejection envelope')
        self.shift=float(self.log_density(self.mode))
        bl=float(self.log_density(self.left))-self.shift-self.sl*self.left
        br=float(self.log_density(self.right))-self.shift-self.sr*self.right
        # Third tangent prevents exponential inefficiency on broad plateaus.
        # Any interior tangent is valid: no exact numerical mode is assumed.
        self.sm=float(self.derivative(self.mode));bm=-self.sm*self.mode
        if not self.sl>self.sm>self.sr:raise ArithmeticError('degenerate tangent slopes')
        self.cl=(bm-bl)/(self.sl-self.sm);self.cr=(br-bm)/(self.sm-self.sr)
        if not np.isfinite([self.cl,self.cr]).all() or self.cr<=self.cl:raise ArithmeticError('bad tangent intersections')
        self.bl=bl;self.bm=bm;self.br=br
        hl=self.sm*self.cl+bm;hr=self.sm*self.cr+bm;width=self.cr-self.cl;t=self.sm*width
        if self.sm==0:lm=hl+np.log(width)
        elif t>50:lm=hl+t+np.log1p(-np.exp(-t))-np.log(abs(self.sm))
        else:lm=hl+np.log(abs(np.expm1(t)))-np.log(abs(self.sm))
        lw=np.array([hl-np.log(self.sl),lm,hr-np.log(-self.sr)])
        self.region_prob=np.exp(lw-logsumexp(lw))
        # A common vertical shift does not change the proposal distribution.
        # This guard is not an interval-arithmetic certificate. Every observed
        # positive excess still fails closed, without acceptance clipping.
        self.envelope_pad=64*np.finfo(float).eps*(1+abs(self.shift)+abs(bl)+abs(bm)+abs(br)+np.max(np.abs(w)))

    def regions(self,u):
        # The last interval is the remainder, not a fourth uninitialized region
        # when a floating-point cumulative probability is just below one.
        t0=float(self.region_prob[0]);t1=min(1.,float(self.region_prob[:2].sum()))
        return np.where(u<t0,0,np.where(u<t1,1,2))

    def log_density(self,y):
        x=np.asarray(y)[...,None]+self.w
        # g(log F42)=8 exp(2x)/(1+2exp x)^3. Constant retained for quadrature.
        return np.sum(np.log(8.)+2*x-3*np.logaddexp(0.,x+np.log(2.)),axis=-1)

    def derivative(self,y):
        return np.sum(2-3*expit(np.asarray(y)[...,None]+self.w+np.log(2.)),axis=-1)

    def log_envelope(self,y):
        y=np.asarray(y)
        return np.minimum(np.minimum(self.sl*y+self.bl,self.sm*y+self.bm),self.sr*y+self.br)+self.envelope_pad

    def draw(self,rng,size,proposal_cap=None):
        if not isinstance(size,int) or size<1:raise ValueError('positive sample size')
        cap=max(10000,100*size) if proposal_cap is None else int(proposal_cap)
        pieces=[];used=accepted=0;max_excess=-np.inf
        while accepted<size and used<cap:
            n=min(max(64,2*(size-accepted)),4096,cap-used)
            region=self.regions(rng.random(n));u=rng.random(n)
            if np.any(u<=0):raise ArithmeticError('uniform zero: conservative failure')
            y=np.empty(n);left=region==0;mid=region==1;right=region==2
            y[left]=self.cl+np.log(u[left])/self.sl
            y[right]=self.cr+np.log(u[right])/self.sr
            t=self.sm*(self.cr-self.cl)
            if self.sm==0:y[mid]=self.cl+u[mid]*(self.cr-self.cl)
            elif abs(t)<50:y[mid]=self.cl+np.log1p(u[mid]*np.expm1(t))/self.sm
            else:y[mid]=self.cl+np.logaddexp(np.log1p(-u[mid]),np.log(u[mid])+t)/self.sm
            excess=self.log_density(y)-self.shift-self.log_envelope(y)
            max_excess=max(max_excess,float(excess.max()))
            if not np.isfinite(excess).all() or np.any(excess>0):
                raise ArithmeticError('numerical envelope violation: no clipping into success')
            mask=np.log(rng.random(n))<=excess
            got=y[mask];pieces.append(got);accepted+=len(got);used+=n
        if accepted<size:raise RuntimeError('conditional rejection sampler cap: fail closed, no MCMC substitute')
        return np.concatenate(pieces)[:size]-LOG_CENTER,{'proposed':used,'accepted_before_truncation':accepted,
            'draws':size,'acceptance_fraction':accepted/used,'max_log_envelope_excess':max_excess,
            'mode':self.mode,'left':self.left,'right':self.right,'slopes':[self.sl,self.sr],
            'envelope_pad':float(self.envelope_pad),'region_rule':'last interval is remainder',
            'floating_point_certificate':False}

def from_bank(cal):
    logs=log_pivots(cal);return ConditionalLogF(logs-logs.mean())
