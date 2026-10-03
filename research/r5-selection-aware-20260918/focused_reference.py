"""A0.10 finite MC e-normalization with pointwise continuous-drift cover.

Development only. Classical exchangeability and monotone bounds, not novelty.
No test labels, unknown delta, kappa or true shape are used.
"""
import numpy as np
from scipy.special import expit
from joint_power_reference import positive_int
from joint_power_kernel import fixed_meta
from r5_common import fc

class FocusedSamplerFailure(RuntimeError):
    def __init__(self,message,receipt):super().__init__(message);self.receipt=receipt

def draw_with_receipt(law,rng,n,stage):
    try:return law.draw(rng,n)
    except (ArithmeticError,RuntimeError,ValueError) as err:
        raise FocusedSamplerFailure('conditional reference draw failed',{
            'stage':stage,'required':n,'bank_n':law.n,'w':law.w,
            'default_proposal_cap':max(10000,100*n),'error':repr(err),
            'actual_attempted_and_accepted':'not exposed by inherited sampler; not invented'}) from err

def pivot_sample(rng,n,g,identity_shape=False):
    parts=[]
    for first in range(0,n,64):
        size=min(64,n-first)
        h=np.broadcast_to(np.eye(4),(size,4,4)) if identity_shape else fc.shape_fit(rng.normal(size=(size,g//4,4,5)),2)
        ev=np.linalg.eigvalsh(h);chi=rng.chisquare(5,size=(size,4));z=rng.normal(size=size)
        den=h[:,0,0]*np.sum(chi/ev,axis=1)/20
        if not np.isfinite(den).all() or np.any(den<=0):raise ArithmeticError('invalid reference pivot')
        parts.append(z/np.sqrt(den))
    return np.concatenate(parts)

def mc_e(score,total,n):
    score=np.asarray(score,float);total=np.asarray(total,float)
    if not np.isfinite(score).all() or not np.isfinite(total).all() or np.any(score<0) or np.any(total<0):
        raise ArithmeticError('invalid nonnegative MC scores')
    den=score+total
    return np.divide((n+1)*score,den,out=np.zeros(np.broadcast_shapes(score.shape,total.shape)),where=den>0)

def components(t,u,s,a,c,tau,qt,qb):
    us=u+s;gate=expit((c-us)/tau)
    target=(t>qt).astype(float)
    # Thresholds compared on log scale to avoid an unnecessary exponential.
    positive=t>0;logt=np.full_like(t,-np.inf);logt[positive]=np.log(t[positive])
    borrowed=(logt-a*us/2>np.log(qb)).astype(float)
    return gate*borrowed,(1-gate)*target

def cover_sum(t,u,*,a,c,tau,qt,qb,max_cells=1024):
    t=np.asarray(t,float);u=np.asarray(u,float)
    if t.ndim!=1 or t.shape!=u.shape or not np.isfinite([*t,*u,a,c,tau,qt,qb]).all():raise ValueError('finite scalar reference tuples')
    if not 0<a<=1 or tau<=0 or min(qt,qb)<=0 or len(t)==0:raise ValueError('invalid reference cover parameters')
    max_cells=positive_int(max_cells,'max_cells')
    # Geometry only depends on complete reference tuples; F2b is pointwise.
    stop=max(0.,float(c-u.min()+tau*np.log(max(len(t)/.01-1,1.))))
    cells=min(max_cells,max(1,int(np.ceil(stop/(.1*tau)))))
    endpoints=np.linspace(0.,stop,cells+1)
    b0,a0=components(t,u,0.,a,c,tau,qt,qb);last_b=float(b0.sum())
    lower=float((b0+a0).sum());upper=0.;maxcell=None
    for i,s in enumerate(endpoints[1:]):
        b,aa=components(t,u,float(s),a,c,tau,qt,qb)
        candidate=last_b+float(aa.sum())
        if candidate>upper:upper=candidate;maxcell=i
        lower=max(lower,float((b+aa).sum()));last_b=float(b.sum())
    tail=float((t>qt).sum())+last_b
    upper=max(upper,tail);lower=max(lower,float((t>qt).sum()))
    if not np.isfinite([upper,lower]).all() or upper<lower-1e-10:raise ArithmeticError('cover bound inconsistency')
    return upper,{'kind':'POINTWISE_MONOTONE_CONTINUUM_UPPER_SUM_NOT_GRID_ONLY',
        'cells':cells,'S':stop,'nominal_width_over_tau':.1,'actual_width_over_tau':stop/cells/tau,
        'maximum_upper_cell':maxcell,'upper_sum':upper,'evaluated_lower_sum':lower,
        'relative_cover_gap':None if lower==0 else upper/lower-1,'tail_sum':tail,
        'tail_borrow_part':last_b,'max_cells':max_cells}

def make_reference(cs,ct,g,seed,*,outer_draws=8191,meta_draws=4095):
    outer_draws=positive_int(outer_draws,'outer_draws',63);meta_draws=positive_int(meta_draws,'meta_draws',63)
    streams=[int(s.generate_state(1,dtype=np.uint64)[0]) for s in np.random.SeedSequence(seed).spawn(9)]
    ls,lt,meta=fixed_meta(cs,ct,streams[0],meta_draws)
    # Fresh quantile-meta sample independent of gate meta and evaluation reference.
    sx,srec=draw_with_receipt(ls,np.random.default_rng(streams[1]),meta_draws,'quantile_source')
    tx,trec=draw_with_receipt(lt,np.random.default_rng(streams[2]),meta_draws,'quantile_target')
    sx-=meta['ms'];tx-=meta['mt'];v=pivot_sample(np.random.default_rng(streams[3]),meta_draws,g)
    weights={'target':0.,'source_bound':1.,'count_bridge':len(cs)/(len(cs)+len(ct)),'variance_bridge':meta['a']}
    thresholds={}
    for name,a in weights.items():
        value=v*np.exp(-.5*(a*sx+(1-a)*tx))
        thresholds[name]=float(np.quantile(value,1-1/1024,method='higher'))
        if not np.isfinite(thresholds[name]) or thresholds[name]<=0:raise ArithmeticError('invalid independently chosen focus threshold')
    ex,er=draw_with_receipt(ls,np.random.default_rng(streams[4]),outer_draws,'evaluation_source')
    ey,yr=draw_with_receipt(lt,np.random.default_rng(streams[5]),outer_draws,'evaluation_target')
    ex-=meta['ms'];ey-=meta['mt'];base=pivot_sample(np.random.default_rng(streams[6]),outer_draws,g)
    target=base*np.exp(-ey/2);u=ex-ey
    Q,cover=cover_sum(target,u,a=meta['a'],c=meta['c'],tau=meta['tau'],qt=thresholds['target'],qb=thresholds['variance_bridge'])
    totals={name:float(np.sum(base*np.exp(-.5*(a*ex+(1-a)*ey))>thresholds[name])) for name,a in weights.items()}
    return {'meta':meta,'thresholds':thresholds,'weights':weights,'totals':totals,'Q':Q,'cover':cover,
        'outer_draws':outer_draws,'streams':streams,'unused_streams':streams[7:],
        'quantile_level':1-1/1024,'quantile_meta_draws':meta_draws,
        'samplers':{'quantile_source':srec,'quantile_target':trec,'reference_source':er,'reference_target':yr},
        'reference_arrays':{'error_source':ex,'error_target':ey,'pivot':base},
        'ancillary_source':ls.w,'ancillary_target':lt.w}
