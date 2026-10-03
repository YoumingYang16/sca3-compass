"""Drift-monotone co-design; classical max-score calibration, novelty OPEN."""
import numpy as np
from selection_profile import Profile

class CodesignedProfile(Profile):
    def __init__(self,*args,source_weight=None):
        super().__init__(*args)
        if source_weight is not None:
            self.a=float(source_weight)
            if not 0<self.a<=1:raise ArithmeticError('conditional source weight outside(0,1]')
            self.logbridge=self.logbase-(self.a*self.es+(1-self.a)*self.et)/2
        # Independent reference normalization only, no calibrated-p claim.
        self.qt=float(np.quantile(self.it,.99,method='higher'))
        self.qb=float(np.quantile(self.ib,.99,method='higher'))
        if not np.isfinite([self.qt,self.qb]).all():
            raise ArithmeticError('invalid co-design normalizer')
        # Any strictly positive independent normalizer is valid. An all-negative
        # finite inner sample must not create a permanent convergence exception.
        # This changes the score only; it does not substitute target discoveries.
        self.normalizer_fallback={'target':self.qt<=0,'bridge':self.qb<=0}
        if self.qt<=0:self.qt=1.
        if self.qb<=0:self.qb=1.
        self.lqt=np.log(self.qt);self.lqb=np.log(self.qb)
        self.c=2*(self.lqt-self.lqb)/self.a
        self.boundary_scores=np.sort(np.maximum(self.logtarget-self.lqt,self.logbridge-self.lqb))

    def pvalues(self,stat,borrow):
        w=np.asarray(stat,float);p=np.ones_like(w);pos=w>0
        score=np.log(w[pos])-(self.lqb if borrow else self.lqt)
        count=self.m-np.searchsorted(self.boundary_scores,score,side='left')
        p[pos]=(1+count)/(self.m+1)
        return p

    def draw_scores(self,s):
        if s<0:raise ValueError('legal slack nonnegative')
        return np.maximum(self.logtarget-self.lqt,self.logbridge-self.lqb-self.a*s/2)
