"""Minimal upward switch correction. Classical building blocks; novelty OPEN."""
import numpy as np
from selection_profile import Profile
from continuous_profile import TailMap

class LogTailMap:
    def __init__(self,reference):
        r=np.asarray(reference,float)
        if not np.isfinite(r).all():raise ArithmeticError('nonfinite inner map')
        self.fallback=len(np.unique(r[r>0]))<2
        self.map=None if self.fallback else TailMap(r)
    def forward_log(self,x):
        x=np.asarray(x,float)
        if np.isnan(x).any() or np.isposinf(x).any():raise ArithmeticError('nonfinite log score')
        if self.fallback:return np.logaddexp(0.,x)
        t=self.map;y=np.interp(x,t.x,t.y)
        y=np.where(x<t.x[0],t.y[0]+t.low*(x-t.x[0]),y)
        y=np.where(x>t.x[-1],t.y[-1]+t.high*(x-t.x[-1]),y)
        return np.maximum(y,0.)

class SwitchClosure(Profile):
    def __init__(self,*args,source_weight,switch_threshold):
        super().__init__(*args)
        self.a=float(source_weight);self.c=float(switch_threshold)
        if not 0<self.a<=1 or not np.isfinite(self.c):raise ArithmeticError('invalid closure rule')
        self.logbridge=self.logbase-(self.a*self.es+(1-self.a)*self.et)/2
        self.mt=LogTailMap(self.it);self.mb=LogTailMap(self.ib)
        self.map_fallback={'target':self.mt.fallback,'bridge':self.mb.fallback}
        self.boundary_scores=np.sort(self.draw_scores(0.))
    def bridge_score(self,logw):
        return np.maximum(self.mb.forward_log(logw),self.mt.forward_log(logw+self.a*self.c/2))
    def draw_scores(self,s):
        if not np.isfinite(s) or s<0:raise ValueError('finite nonnegative legal slack')
        active=s+self.es-self.et<=self.c
        return np.where(active,self.bridge_score(self.logbridge-self.a*s/2),self.mt.forward_log(self.logtarget))
    def pvalues(self,stat,borrow):
        w=np.asarray(stat,float);p=np.ones_like(w);pos=w>0
        if not np.isfinite(w).all():raise ArithmeticError('nonfinite observed statistic')
        logw=np.log(w[pos]);score=self.bridge_score(logw) if borrow else self.mt.forward_log(logw)
        count=self.m-np.searchsorted(self.boundary_scores,score,side='left')
        p[pos]=(1+count)/(self.m+1)
        return p
