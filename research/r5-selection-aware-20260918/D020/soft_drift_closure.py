"""A0.8 complete-rule soft-score closure; classical order/MC, novelty OPEN."""
import numpy as np

class SoftDriftClosure:
    def __init__(self,es,et,base,a,c,inner_target,inner_bridge):
        self.es=np.asarray(es,float);self.et=np.asarray(et,float);self.base=np.asarray(base,float)
        self.a=float(a);self.c=float(c);self.tau=self.c*np.sqrt(3)/np.pi
        if not 0<self.a<1 or not np.isfinite([self.a,self.c]).all() or self.c<=0:raise ArithmeticError('invalid soft closure meta')
        if self.es.shape!=self.et.shape or self.es.shape!=self.base.shape or not np.isfinite([self.es,self.et,self.base]).all():raise ArithmeticError('invalid tuples')
        self.m=len(base);self.cache={};self.observed_u=None;self.normalizer_fallback={}
        qq=[]
        for name,x in [('target',inner_target),('bridge',inner_bridge)]:
            x=np.asarray(x,float)
            if not len(x) or not np.isfinite(x).all():raise ArithmeticError('invalid independent inner reference')
            q=float(np.quantile(x,.99));bad=q<=0
            self.normalizer_fallback[name]=bad;qq.append(0. if bad else np.log(q))
        self.logqt,self.logqb=qq
        self.logtarget=np.full(self.m,-np.inf);pos=self.base>0
        self.logtarget[pos]=np.log(self.base[pos])-self.et[pos]/2
        self.boundary_scores=np.sort(self.draw_scores(0.))

    def score(self,logt,u):
        logt,u=np.broadcast_arrays(np.asarray(logt,float),np.asarray(u,float))
        if np.isnan(logt).any() or np.isposinf(logt).any() or not np.isfinite(u).all():raise ArithmeticError('invalid soft score')
        ratio=self.logqt-self.logqb-self.a*u/2
        gate=(u-self.c)/self.tau
        # max(target, raw weighted score), in log space; no exp overflow.
        gain=np.logaddexp(-np.logaddexp(0.,-gate),-np.logaddexp(0.,gate)+ratio)
        return logt-self.logqt+np.where(ratio>0,gain,0.)

    def draw_scores(self,s):
        if not np.isfinite(s) or s<0:raise ValueError('legal nonnegative drift slack')
        return self.score(self.logtarget,self.es-self.et+s)

    def pvalues(self,stat,borrow):
        if self.observed_u is None:raise ValueError('observed target/source contrast required')
        w=np.asarray(stat,float)
        if not np.isfinite(w).all():raise ArithmeticError('nonfinite statistic')
        p=np.ones_like(w);pos=w>0
        q=self.score(np.log(w[pos]),self.observed_u)
        p[pos]=(1+self.m-np.searchsorted(self.boundary_scores,q,side='left'))/(self.m+1)
        return p
