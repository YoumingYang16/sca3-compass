"""Full-row lower-triangular calibration pivot; R5 development, not certified.

T1-T3: TRIANGULAR_CALIBRATION.md. No truth/TEST/row selection input.
Inherits only the three-tangent rejection mechanics, NOT the radial density.
"""
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from ancillary_calibration import ConditionalLogF
from selection_profile import LOG_CENTER
from r5_common import fc


def summaries(cal):
    cal=np.asarray(cal,float)
    if cal.ndim!=3 or cal.shape[1:]!=(4,6) or len(cal)<4 or not np.isfinite(cal).all():
        raise ValueError('finite calibration bank N>=4 required')
    mx=np.max(np.abs(cal),axis=(1,2),keepdims=True)
    if np.any(mx<=0):raise ArithmeticError('zero calibration block')
    c=cal/mx;x=c.mean(-1);y=c@fc.contrasts(6)
    # QR(y.T), positive R diagonal: R.T equals lower chol(y@y.T).
    _,r=np.linalg.qr(y.swapaxes(1,2),mode='reduced')
    signs=np.sign(np.diagonal(r,axis1=1,axis2=2))
    if np.any(signs==0):raise ArithmeticError('singular triangular calibration')
    r=r*signs[:,:,None]
    t=np.stack([solve_triangular(ri.T,xi,lower=True) for ri,xi in zip(r,x)])
    q=np.cumsum(t*t,axis=1)
    if not np.isfinite(q).all() or np.any(q[:,0]<=0):raise ArithmeticError('bad triangular radius')
    logq=np.log(q[:,-1]);angles=q[:,:3]/q[:,-1,None]
    if np.any(np.diff(np.c_[np.zeros(len(q)),angles,np.ones(len(q))],axis=1)<=0):
        raise ArithmeticError('angular precision/singularity failure: no clipping')
    return logq,angles


class ConditionalTriangular(ConditionalLogF):
    def __init__(self,residuals,angles):
        w=np.asarray(residuals,float);a=np.asarray(angles,float)
        if w.ndim!=1 or len(w)<4 or a.shape!=(len(w),3) or not np.isfinite(w).all() or not np.isfinite(a).all():
            raise ValueError('finite centered residuals and Nx3 angles required')
        if abs(w.mean())>1e-10*(1+np.max(np.abs(w))):raise ValueError('residuals must be centered')
        if np.any(np.diff(np.c_[np.zeros(len(w)),a,np.ones(len(w))],axis=1)<=0):
            raise ValueError('strict interior ordered angles required; no clipping')
        self.w=w.copy();self.n=len(w);self.angles=a.copy();self.log_angles=np.log(a)
        lo=-float(w.max())-10;hi=-float(w.min())-float(self.log_angles.min())+10
        self.mode=float(brentq(self.derivative,lo,hi,xtol=1e-13))
        z=self.mode+w;v=expit(z[:,None]+self.log_angles);v4=expit(z)
        curvature=np.sum(v*(1-v))+1.5*np.sum(v4*(1-v4))
        step=1/np.sqrt(max(curvature,1.))
        for _ in range(64):
            if self.derivative(self.mode-step)>=.5 and self.derivative(self.mode+step)<=-.5:break
            step*=2
        else:raise ArithmeticError('cannot bracket triangular envelope')
        self.left=self.mode-step;self.right=self.mode+step
        self.sl=float(self.derivative(self.left));self.sr=float(self.derivative(self.right))
        self.shift=float(self.log_density(self.mode));self.sm=float(self.derivative(self.mode))
        if not self.sl>self.sm>self.sr or not self.sl>0>self.sr:raise ArithmeticError('invalid tangent slopes')
        self.bl=float(self.log_density(self.left))-self.shift-self.sl*self.left
        self.br=float(self.log_density(self.right))-self.shift-self.sr*self.right
        self.bm=-self.sm*self.mode
        self.cl=(self.bm-self.bl)/(self.sl-self.sm);self.cr=(self.br-self.bm)/(self.sm-self.sr)
        if not np.isfinite([self.cl,self.cr]).all() or self.cr<=self.cl:raise ArithmeticError('bad tangent intersections')
        hl=self.sm*self.cl+self.bm;hr=self.sm*self.cr+self.bm;width=self.cr-self.cl;z=self.sm*width
        if self.sm==0:lm=hl+np.log(width)
        elif z>50:lm=hl+z+np.log1p(-np.exp(-z))-np.log(abs(self.sm))
        else:lm=hl+np.log(abs(np.expm1(z)))-np.log(abs(self.sm))
        lw=np.array([hl-np.log(self.sl),lm,hr-np.log(-self.sr)])
        self.region_prob=np.exp(lw-logsumexp(lw))
        self.envelope_pad=128*np.finfo(float).eps*(1+abs(self.shift)+abs(self.bl)+abs(self.bm)+abs(self.br)
                            +np.max(np.abs(w))+np.max(np.abs(self.log_angles)))

    def log_density(self,y):
        z=np.asarray(y)[...,None]+self.w
        return np.sum(2*z-np.sum(np.logaddexp(0.,z[...,None]+self.log_angles),axis=-1)
                      -1.5*np.logaddexp(0.,z),axis=-1)

    def derivative(self,y):
        z=np.asarray(y)[...,None]+self.w
        return np.sum(2-np.sum(expit(z[...,None]+self.log_angles),axis=-1)-1.5*expit(z),axis=-1)

    def draw(self,rng,size,proposal_cap=None):
        draws,rec=super().draw(rng,size,proposal_cap)
        # Inherited mechanical sampler subtracts radial LOG_CENTER. Undo it,
        # then subtract ONE for mean(logQ4)-1, the actual same old estimate.
        return draws+LOG_CENTER-1.,{**rec,'law':'full_triangular_conditional',
            'error_center_subtracted':1.,'conditioned_on':'angles AND centered logQ4 residuals',
            'signs_and_W_conditioned_on':False}


def from_bank(cal):
    logq,a=summaries(cal)
    return ConditionalTriangular(logq-logq.mean(),a)


def geometric_kappa(cal):
    logq,_=summaries(cal)
    return float(np.exp(logq.mean()-1.))
