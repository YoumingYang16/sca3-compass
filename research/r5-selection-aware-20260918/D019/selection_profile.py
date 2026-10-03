"""Exact one-switch empirical nuisance profile; development, not certified.

Inner fixed-reference ranks standardize the target/bridge branches. The outer
profile includes the WHOLE selector for every s=log(D/delta)>=0. No theta grid,
estimated delta, target truth or extra observational pilot is used.
"""
import numpy as np

LOG_CENTER=1.-np.log(2.)
LOG_VARIANCE=np.pi**2/3.-1.

def threshold(ns,nt):
    if ns<4 or nt<4:raise ValueError('original bank minimum remains four')
    return float(np.sqrt(LOG_VARIANCE*(1/ns+1/nt)))

def select(log_ks,log_kt,ns,nt,bound,switch_threshold=None,source_weight=None):
    if not np.isfinite(bound) or bound<1:raise ValueError('external D>=1 required')
    u=np.log(bound)+log_ks-log_kt;c=threshold(ns,nt) if switch_threshold is None else float(switch_threshold)
    if not np.isfinite(c):raise ValueError('finite independent selector threshold required')
    borrow=bool(u<=c)
    a=ns/(ns+nt) if source_weight is None else float(source_weight)
    if not 0<a<=1:raise ValueError('source weight in(0,1]')
    target_weight=nt/(ns+nt) if source_weight is None else 1-a
    return {'borrow':borrow,'target_weight':target_weight if borrow else 1.,
            'observed_slack':float(u),'switch_threshold':c,
            'log_scale':float(log_kt+a*u if borrow else log_kt)}

def inner_rank(stat,reference):
    stat=np.asarray(stat,float);m=len(reference)
    return np.where(stat>0,1+m-np.searchsorted(reference,stat,side='left'),m+1)

class Profile:
    """A shared outer empirical profile. Its envelope entries are NOT iid.

    positive_base is sign-allowed standard projected shape/radial pivot V, NOT
    raw target radii. Es/Et are independently generated log calibration errors.
    These constructor inputs are reference randomization only, not deployable
    observed nuisance truths. All numerical failure must be handled fail-closed
    by the calling research kernel.
    """
    def __init__(self,es,et,base,ns,nt,inner_target,inner_bridge):
        self.es=np.asarray(es,float);self.et=np.asarray(et,float);self.base=np.asarray(base,float)
        if self.es.ndim!=1 or self.es.shape!=self.et.shape or self.et.shape!=self.base.shape:
            raise ValueError('complete independent outer tuple arrays required')
        if not all(np.isfinite(x).all() for x in [self.es,self.et,self.base]):raise ArithmeticError('nonfinite outer tuple')
        self.ns=ns;self.nt=nt;self.a=ns/(ns+nt);self.c=threshold(ns,nt);self.m=len(self.es)
        self.it=np.sort(np.asarray(inner_target,float));self.ib=np.sort(np.asarray(inner_bridge,float))
        if len(self.it)!=len(self.ib) or not len(self.it) or not np.isfinite(self.it).all() or not np.isfinite(self.ib).all():
            raise ValueError('finite equal-size inner references')
        self.ninner=len(self.it)+1
        self.switch=self.c-self.es+self.et
        self.positive=self.base>0
        self.logbase=np.full(self.m,-np.inf)
        self.logbase[self.positive]=np.log(self.base[self.positive])
        self.logtarget=self.logbase-self.et/2
        self.logbridge=self.logbase-(self.a*self.es+(1-self.a)*self.et)/2
        self.cache={}

    def cutoff(self,k,branch):
        if not 1<=k<self.ninner:raise ValueError('nontrivial inner rank required')
        r=self.it if branch=='target' else self.ib
        return max(0.,float(r[len(r)-k]))

    def count(self,k):
        """Exact max overlap of prefixes[0,min(switch,tau)) and suffixes.

        At switch, bridge is used; immediately after it target is used. Group
        all equal endpoints: never count artificial overlaps of open/closed
        branches. Comparing left and right limits covers the exact point too.
        Prefix endpoint is closed if switch<tau, else open; this can only
        lower the count AT the endpoint versus its left limit. Domain s0 is
        checked separately. Strict tail inversion matches inner >= tie rule.
        """
        k=int(k)
        if k==self.ninner:return self.m
        if k in self.cache:return self.cache[k]['maximum']
        ct=self.cutoff(k,'target');cb=self.cutoff(k,'bridge')
        lt=-np.inf if ct==0 else np.log(ct);lb=-np.inf if cb==0 else np.log(cb)
        suffix=self.positive & (self.logtarget>lt)
        tau=np.full(self.m,-np.inf)
        tau[self.positive]=np.inf if cb==0 else 2*(self.logbridge[self.positive]-lb)/self.a
        prefix=self.positive & (self.switch>=0) & (tau>0)
        ends=np.minimum(self.switch[prefix],tau[prefix])
        starts=self.switch[suffix & (self.switch>=0)]
        initial=int(prefix.sum()+np.count_nonzero(suffix & (self.switch<0)))
        points=np.r_[ends,starts];change=np.r_[-np.ones(len(ends),int),np.ones(len(starts),int)]
        if len(points):
            order=np.argsort(points,kind='stable');x=points[order];d=change[order]
            idx=np.r_[0,1+np.flatnonzero(x[1:]!=x[:-1])]
            counts=initial+np.cumsum(np.add.reduceat(d,idx))
            maximum=max(initial,int(counts.max()))
            where=0. if initial>=counts.max() else float(x[idx[np.argmax(counts)]])
        else:maximum=initial;where=0.
        end=int(suffix.sum())
        if not 0<=initial<=self.m or not end<=maximum<=self.m:raise ArithmeticError('profile overlap invariant')
        self.cache[k]={'maximum':maximum,'at_zero':initial,'at_infinity':end,
                       'attainment_or_right_limit':where,'event_count':len(points)}
        return maximum

    def calibrate_rank(self,k):
        a=np.asarray(k)
        if np.any((a<1)|(a>self.ninner)|(a!=np.floor(a))):raise ValueError('inner integer rank grid')
        keys,inv=np.unique(a,return_inverse=True)
        counts=np.array([self.count(int(x)) for x in keys])
        return ((1+counts[inv])/(self.m+1)).reshape(a.shape)

    def pvalues(self,stat,borrow):
        return self.calibrate_rank(inner_rank(stat,self.ib if borrow else self.it))

    def direct_count(self,k,s,force_right=False):
        """Independent finite-parameter check; NOT the algorithm's optimizer."""
        if k==self.ninner:return self.m
        active=self.switch>s if force_right else self.switch>=s
        ct=self.cutoff(k,'target');cb=self.cutoff(k,'bridge')
        lt=-np.inf if ct==0 else np.log(ct);lb=-np.inf if cb==0 else np.log(cb)
        return int(np.count_nonzero(self.positive & np.where(active,self.logbridge-self.a*s/2>lb,self.logtarget>lt)))

def draw_errors(rng,n,size):
    return np.mean(np.log(rng.f(4,2,size=(size,n))),axis=1)-LOG_CENTER
