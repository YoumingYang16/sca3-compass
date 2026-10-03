"""Nuisance-indexed joint reference profile; DEVELOPMENT, not accepted."""
import heapq
import numpy as np
from scipy.special import logsumexp
from focused_reference import components,mc_e

def log_cauchy(v,scale):
    v=np.asarray(v,float)
    if not np.isfinite(v).all() or not np.isfinite(scale) or scale<=0:raise ValueError('finite positive proxy parameters')
    with np.errstate(divide='ignore'):
        return -np.logaddexp(0.,2*(np.log(np.abs(v))-np.log(scale)))

def log_cauchy_max(lo,hi,scale):
    lo,hi=np.broadcast_arrays(lo,hi)
    if np.any(lo>hi):raise ValueError('reversed proxy interval')
    point=np.where(lo>0,lo,np.where(hi<0,hi,0.))
    return log_cauchy(point,scale)

def profile(t,u,observed_u,*,a,c,tau,qt,qb,bound,sigma,max_cells=256,tolerance=.01):
    t=np.asarray(t,float);u=np.asarray(u,float)
    if t.ndim!=1 or t.shape!=u.shape or not np.isfinite([*t,*u,observed_u,a,c,tau,qt,qb,bound,sigma]).all():raise ValueError('finite paired tuples')
    if min(qt,qb,tau,sigma)<=0 or not 0<a<=1 or bound<1 or max_cells<1:raise ValueError('invalid profile design')
    fscale=2*sigma;rscale=sigma+np.log(bound)
    lf=log_cauchy(u,fscale);lr_obs=float(log_cauchy(observed_u,rscale))
    stop=max(0.,float(observed_u),float(-u.min()),float(c-u.min()))+8*(fscale+rscale)
    cache={};lower=0.
    def at(s):
        nonlocal lower
        if s not in cache:
            b,aa=components(t,u,s,a,c,tau,qt,qb)
            with np.errstate(divide='ignore'):
                ell=np.log(b+aa)+log_cauchy(u+s,rscale)-lf
            value=float(np.exp(log_cauchy(observed_u-s,fscale)+logsumexp(ell)-lr_obs))
            if not np.isfinite(value):raise ArithmeticError('nonfinite localized score')
            cache[s]=(b,aa,value);lower=max(lower,value)
        return cache[s]
    def cell(lo,hi):
        bl,_,_=at(lo);_,ar,_=at(hi)
        with np.errstate(divide='ignore'):
            ell=np.log(bl+ar)+log_cauchy_max(u+lo,u+hi,rscale)-lf
        value=float(np.exp(log_cauchy_max(observed_u-hi,observed_u-lo,fscale)+logsumexp(ell)-lr_obs))
        if not np.isfinite(value):raise ArithmeticError('nonfinite localized cell bound')
        return value
    bS,_,_=at(stop)
    with np.errstate(divide='ignore'):
        ell=np.log((t>qt).astype(float)+bS)+log_cauchy(u+stop,rscale)-lf
    tail=float(np.exp(log_cauchy(observed_u-stop,fscale)+logsumexp(ell)-lr_obs))
    if not np.isfinite(tail):raise ArithmeticError('nonfinite localized tail')
    heap=[(-cell(0.,stop),0.,stop)];created=1
    while created<max_cells:
        maximum=max(-heap[0][0],tail)
        if maximum<=(1+tolerance)*lower+1e-12:break
        if tail>=-heap[0][0]:break # tail bound is valid; cannot fix it by splitting finite cells
        neg,lo,hi=heapq.heappop(heap);mid=(lo+hi)/2
        if not lo<mid<hi:
            heapq.heappush(heap,(neg,lo,hi));break
        heapq.heappush(heap,(-cell(lo,mid),lo,mid));heapq.heappush(heap,(-cell(mid,hi),mid,hi));created+=1
    Q=max(-heap[0][0],tail)
    if Q+1e-10<lower:raise ArithmeticError('localized upper below evaluated lower')
    return Q,{'kind':'NUISANCE_INDEXED_MC_WITH_CERTIFIED_IDEAL_ARITHMETIC_COVER',
        'proxy_f':'unnormalizedCauchy','fscale':float(fscale),'rscale':float(rscale),
        'S':stop,'cells':created,'max_cells':max_cells,'relative_target':tolerance,
        'upper_sum':Q,'evaluated_lower_sum':lower,'tail_bound':tail,
        'relative_gap':None if lower==0 else Q/lower-1,
        'active_interval':[heap[0][1],heap[0][2]],'evaluation_nodes':len(cache),
        'not_claimed':'machine interval certificate, new theorem priority or utility acceptance'}

def observed_e(t,observed_u,ref,Q):
    from focused_kernel import indicators
    from scipy.special import expit
    m=ref['meta'];q=ref['thresholds'];g=float(expit((m['c']-observed_u)/m['tau']))
    score=g*indicators(t,observed_u,m['a'],q['variance_bridge'])+(1-g)*indicators(t,observed_u,0.,q['target'])
    return mc_e(score,Q,ref['outer_draws'])
