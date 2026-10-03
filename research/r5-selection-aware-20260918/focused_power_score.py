"""Fixed fourth-power-above-focus score; no parameter search or novelty claim."""
import heapq
import numpy as np
from scipy.special import expit
from focused_reference import mc_e

def power_score(t,shift,q,focused=True):
    t=np.asarray(t,float);shift=np.broadcast_to(shift,t.shape)
    out=np.zeros(t.shape);pos=t>0
    logs=np.full(t.shape,-np.inf);logs[pos]=np.log(t[pos])-shift[pos]-np.log(q)
    keep=(logs>0) if focused else pos
    out[keep]=np.exp(4*logs[keep])
    if not np.isfinite(out).all():raise ArithmeticError('nonfinite focused power')
    return out

def components(t,u,s,a,c,tau,qt,qb,focused=True):
    gate=expit((c-u-s)/tau)
    return gate*power_score(t,a*(u+s)/2,qb,focused),(1-gate)*power_score(t,0.,qt,focused)

def cover(t,u,*,a,c,tau,qt,qb,max_cells=256,tolerance=.01,focused=True):
    t=np.asarray(t,float);u=np.asarray(u,float)
    if t.shape!=u.shape or t.ndim!=1 or not np.isfinite([*t,*u,a,c,tau,qt,qb]).all():raise ValueError('finite paired reference')
    if not 0<a<=1 or min(qt,qb,tau)<=0:raise ValueError('invalid fixed rule')
    # At S the borrow scale is extremely attenuated; actual remainder, not0, retained.
    stop=max(0.,float(c-u.min()+tau*np.log(max(len(t)/.01-1,1.))))+8*tau
    cache={};lower=0.
    def at(s):
        nonlocal lower
        if s not in cache:
            b,aa=components(t,u,s,a,c,tau,qt,qb,focused)
            cache[s]=(float(b.sum()),float(aa.sum()));lower=max(lower,sum(cache[s]))
        return cache[s]
    def interval(lo,hi):return at(lo)[0]+at(hi)[1]
    tail=float(power_score(t,0.,qt,focused).sum())+at(stop)[0]
    lower=max(lower,float(power_score(t,0.,qt,focused).sum()))
    heap=[(-interval(0.,stop),0.,stop)];cells=1
    while cells<max_cells:
        if max(-heap[0][0],tail)<=(1+tolerance)*lower+1e-12:break
        if tail>=-heap[0][0]:break
        n,lo,hi=heapq.heappop(heap);mid=(lo+hi)/2
        if not lo<mid<hi:heapq.heappush(heap,(n,lo,hi));break
        heapq.heappush(heap,(-interval(lo,mid),lo,mid));heapq.heappush(heap,(-interval(mid,hi),mid,hi));cells+=1
    Q=max(-heap[0][0],tail)
    if not np.isfinite(Q) or Q+1e-10<lower:raise ArithmeticError('invalid power cover')
    return Q,{'kind':'CONTINUOUS_MONOTONE_COVER','focused':focused,'power':4,'cells':cells,'S':stop,
              'upper':Q,'lower':lower,'tail':tail,'relative_gap':None if lower==0 else Q/lower-1,'max_cells':max_cells}

def prepare_scores(target,u,ref):
    m=ref['meta'];q=ref['thresholds']
    Q,receipt=cover(target,u,a=m['a'],c=m['c'],tau=m['tau'],qt=q['target'],qb=q['variance_bridge'])
    QU,pure=cover(target,u,a=m['a'],c=m['c'],tau=m['tau'],qt=q['target'],qb=q['variance_bridge'],focused=False)
    endpoint={name:power_score(target,a*u/2,q[name]) for name,a in ref['weights'].items()}
    totals={k:float(v.sum()) for k,v in endpoint.items()}
    return {'Q':Q,'unfocused_Q':QU,'cover':receipt,'unfocused_cover':pure,'totals':totals,
            'max_total':float(np.maximum(endpoint['target'],endpoint['variance_bridge']).sum()),
            'fixed_joint_total':float(.5*(endpoint['target']+endpoint['count_bridge']).sum())}

def scalar_values(t,u,ref,p):
    m=ref['meta'];q=ref['thresholds'];n=ref['outer_draws'];gate=float(expit((m['c']-u)/m['tau']))
    scores={name:power_score(t,a*u/2,q[name]) for name,a in ref['weights'].items()}
    values={'powerfocus_'+k:mc_e(v,p['totals'][k],n) for k,v in scores.items()}
    joint=gate*scores['variance_bridge']+(1-gate)*scores['target']
    values['powerfocus_joint']=mc_e(joint,p['Q'],n)
    values['powerfocus_max']=mc_e(np.maximum(scores['target'],scores['variance_bridge']),p['max_total'],n)
    values['powerfocus_fixed_joint_score']=mc_e(.5*(scores['target']+scores['count_bridge']),p['fixed_joint_total'],n)
    raw=gate*power_score(t,m['a']*u/2,q['variance_bridge'],False)+(1-gate)*power_score(t,0.,q['target'],False)
    values['unfocused_joint_MC']=mc_e(raw,p['unfocused_Q'],n)
    return values
