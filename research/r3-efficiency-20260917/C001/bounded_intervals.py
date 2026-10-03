"""Fixed-grid test-martingale inversion for independent bounded family metrics.

Classical betting principle, not a new inference method or R3 mechanism.
No fitted bets, asymptotic t interval, optional sample extension or gene units.
"""
import numpy as np
from scipy.special import logsumexp

FRACTIONS=np.array([.01,.02,.04,.08,.16,.32,.5,.7,.85,.95,.99,.999])

def log_wealth(x,mu):
    x=np.asarray(x,float)
    if mu==0: return np.inf if np.any(x>0) else float(logsumexp(len(x)*np.log1p(-FRACTIONS))-np.log(len(FRACTIONS)))
    if not 0<mu<=1: raise ValueError('mean in[0,1]')
    factors=1+FRACTIONS[:,None]*(x[None,:]/mu-1)
    if np.any(factors<=0):raise ArithmeticError('nonpositive betting factor')
    return float(logsumexp(np.log(factors).sum(1))-np.log(len(FRACTIONS)))

def lower_bound(x,tail_alpha):
    if not np.any(x>0):return 0.
    lo=0.;hi=float(np.mean(x)); threshold=-np.log(tail_alpha)
    for _ in range(60):
        mid=(lo+hi)/2
        if log_wealth(x,mid)>threshold:lo=mid
        else:hi=mid
    return max(0.,lo-1e-12)  # outward guard, not a universal float certificate

def interval(values,low=0.,high=1.,cap=384,alpha=.025):
    v=np.asarray(values,float)
    if len(v)<2 or not np.isfinite(v).all() or np.any(v<low) or np.any(v>high) or high<=low:
        raise ValueError('independent observations must fit declared finite bounds')
    x=(v-low)/(high-low);tail=alpha/(2*cap)
    a=lower_bound(x,tail);b=1-lower_bound(1-x,tail)
    return {'mean':float(v.mean()),'simultaneous_interval':[low+(high-low)*a,low+(high-low)*b],
            'mcse':float(v.std(ddof=1)/np.sqrt(len(v))),'n':len(v),
            'p90':float(np.quantile(v,.9)),'p99':float(np.quantile(v,.99))}
