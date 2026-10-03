"""A0.1: remove inner-rank saturation, NOT an assumed tail-law estimator.

Any independent fixed nondecreasing map can define the statistic; ONLY complete
outer calibration provides validity. Linear log-tail interpolation and explicit
secant extrapolation define a continuous map and exact algebraic inverse.
"""
import numpy as np
from selection_profile import Profile

class TailMap:
    def __init__(self,reference):
        r=np.sort(np.asarray(reference,float));L=len(r)
        ids=np.flatnonzero(r>0);values=r[ids]
        x,ix=np.unique(np.log(values),return_index=True)
        y=-np.log((L+1-ids[ix])/(L+1))
        if len(x)<2 or not np.isfinite(x).all() or not np.all(np.diff(y)>0):
            raise ArithmeticError('insufficient distinct positive inner reference')
        self.x=x;self.y=y;k=min(16,len(x)-1)
        self.low=(y[k]-y[0])/(x[k]-x[0])
        self.high=(y[-1]-y[-1-k])/(x[-1]-x[-1-k])
        if not np.isfinite([self.low,self.high]).all() or min(self.low,self.high)<=0:
            raise ArithmeticError('noninvertible inner map')

    def forward(self,stat):
        w=np.asarray(stat,float);out=np.zeros_like(w);pos=w>0;x=np.log(w[pos])
        y=np.interp(x,self.x,self.y)
        y=np.where(x<self.x[0],self.y[0]+self.low*(x-self.x[0]),y)
        y=np.where(x>self.x[-1],self.y[-1]+self.high*(x-self.x[-1]),y)
        out[pos]=np.maximum(y,0.)
        return out

    def inverse_log(self,score):
        if not np.isfinite(score) or score<=0:raise ValueError('positive score inverse')
        if score<self.y[0]:return self.x[0]+(score-self.y[0])/self.low
        if score>self.y[-1]:return self.x[-1]+(score-self.y[-1])/self.high
        return float(np.interp(score,self.y,self.x))

class ContinuousProfile(Profile):
    def __init__(self,*args):
        super().__init__(*args);self.mt=TailMap(self.it);self.mb=TailMap(self.ib)

    def count_score(self,x):
        x=float(x)
        if x<=0:return self.m
        if x in self.cache:return self.cache[x]['maximum']
        lt=self.mt.inverse_log(x);lb=self.mb.inverse_log(x)
        # Non-strict outer >= comparison; closed borrowed tau endpoint. At an
        # event point ties can ADD isolated mass: evaluate closed-prefix endings
        # BEFORE removing them, while suffix starts strictly after the switch.
        suffix=self.positive & (self.logtarget>=lt)
        tau=np.full(self.m,-np.inf);tau[self.positive]=2*(self.logbridge[self.positive]-lb)/self.a
        prefix=self.positive & (self.switch>=0) & (tau>=0)
        ends=np.minimum(self.switch[prefix],tau[prefix])
        starts=self.switch[suffix & (self.switch>=0)]
        initial=int(prefix.sum()+np.count_nonzero(suffix & (self.switch<0)))
        points=np.r_[ends,starts];changes=np.r_[-np.ones(len(ends),int),np.ones(len(starts),int)]
        if len(points):
            order=np.argsort(points,kind='stable');points=points[order];changes=changes[order]
            idx=np.r_[0,1+np.flatnonzero(points[1:]!=points[:-1])]
            counts=initial+np.cumsum(np.add.reduceat(changes,idx))
            maximum=max(initial,int(counts.max()));where=0. if initial>=counts.max() else float(points[idx[counts.argmax()]])
        else:maximum=initial;where=0.
        end=int(suffix.sum())
        if not 0<=initial<=self.m or not end<=maximum<=self.m:raise ArithmeticError('continuous profile invariant')
        self.cache[x]={'maximum':maximum,'at_zero':initial,'at_infinity':end,
                       'attainment_or_right_limit':where,'event_count':len(points)}
        return maximum

    def pvalues(self,stat,borrow):
        score=(self.mb if borrow else self.mt).forward(stat)
        keys,inv=np.unique(score,return_inverse=True)
        counts=np.array([self.count_score(x) for x in keys])
        return ((1+counts[inv])/(self.m+1)).reshape(score.shape)

    def direct_score_count(self,x,s,right=False):
        if x<=0:return self.m
        active=self.switch>s if right else self.switch>=s
        lt=self.mt.inverse_log(x);lb=self.mb.inverse_log(x)
        return int(np.count_nonzero(self.positive & np.where(active,self.logbridge-self.a*s/2>=lb,self.logtarget>=lt)))
