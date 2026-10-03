"""A0.9 joint moment-budget prototype; exact-model proof, gates still OPEN.

No inverse plug-in means: tangent envelope importance numerators times
negative-binomial unbiased reciprocals. All failures are explicit.
"""
import time
import numpy as np
from scipy.special import logsumexp,gammaln
from ancillary_calibration import ConditionalLogF,from_bank
from selection_profile import LOG_CENTER
from r5_common import fc

class JointReferenceFailure(RuntimeError):
    def __init__(self,message,receipt):super().__init__(message);self.receipt=receipt

def positive_int(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(name+' must be a finite integer >= '+str(minimum))
    return int(value)

class TiltEnvelope(ConditionalLogF):
    def __init__(self,w,tilt=0.,center=0.):
        self.tilt=float(tilt);self.center=float(center)
        # First prototype alpha4 => all needed tilts<=2; no unverified root range.
        if not 0<=self.tilt<=2 or not np.isfinite(self.center):raise ValueError('prototype requires tilt in[0,2]')
        super().__init__(w)
        hl=self.sm*self.cl+self.bm;hr=self.sm*self.cr+self.bm
        width=self.cr-self.cl;t=self.sm*width
        if self.sm==0:lm=hl+np.log(width)
        elif t>50:lm=hl+t+np.log1p(-np.exp(-t))-np.log(abs(self.sm))
        else:lm=hl+np.log(abs(np.expm1(t)))-np.log(abs(self.sm))
        self.logC=float(self.shift+self.envelope_pad+logsumexp([hl-np.log(self.sl),lm,hr-np.log(-self.sr)]))

    def log_density(self,y):return super().log_density(y)-self.tilt*(np.asarray(y)-self.center)
    def derivative(self,y):return super().derivative(y)-self.tilt
    def log_bound(self,y):return self.shift+self.log_envelope(y)

    def proposal(self,rng,n):
        region=self.regions(rng.random(n));u=rng.random(n)
        if np.any(u<=0):raise ArithmeticError('uniform endpoint in envelope proposal')
        y=np.empty(n);left=region==0;middle=region==1;right=region==2
        y[left]=self.cl+np.log(u[left])/self.sl;y[right]=self.cr+np.log(u[right])/self.sr
        t=self.sm*(self.cr-self.cl)
        if self.sm==0:y[middle]=self.cl+u[middle]*(self.cr-self.cl)
        elif abs(t)<50:y[middle]=self.cl+np.log1p(u[middle]*np.expm1(t))/self.sm
        else:y[middle]=self.cl+np.logaddexp(np.log1p(-u[middle]),np.log(u[middle])+t)/self.sm
        if not np.isfinite(y).all():raise ArithmeticError('nonfinite envelope proposal')
        return y

    def normalizer_ratio(self,rng,n):
        x=self.proposal(rng,n);ell=self.log_density(x)-self.log_bound(x)
        if not np.isfinite(ell).all() or np.any(ell>0):raise ArithmeticError('base envelope violation')
        weights=np.exp(ell);mean=float(weights.mean())
        if not 0<mean<=1:raise ArithmeticError('nonpositive numerator normalizer')
        return mean,{'draws':n,'mean_weight':mean,'SE_descriptive':float(weights.std(ddof=1)/np.sqrt(n)),
                     'max_log_excess':float(ell.max()),'logC':self.logC}

def log_dominating_moment(alpha):
    if not 0<alpha<5:raise ValueError('positive pivot moment must be below5')
    return float(alpha/2*np.log(20)+gammaln((alpha+1)/2)+gammaln((5-alpha)/2)
                 -np.log(2)-.5*np.log(np.pi)-gammaln(2.5))

def shape_sizebias_log_accept(rng,n,g,alpha,identity_shape=False):
    # identity_shape is ONLY a known-parameter test fixture, never deployed API.
    h=np.broadcast_to(np.eye(4),(n,4,4)) if identity_shape else fc.shape_fit(rng.normal(size=(n,g//4,4,5)),2)
    y=rng.normal(size=(n,4,5));direction=y[:,0,:];norm=np.linalg.norm(direction,axis=1)
    if np.any(norm<=0):raise ArithmeticError('zero angular proposal')
    radius2=rng.chisquare(5-alpha,size=n)
    y[:,0,:]=direction/norm[:,None]*np.sqrt(radius2[:,None])
    energy=np.sum(y*np.linalg.solve(h,y),axis=(1,2));den=h[:,0,0]*energy
    if np.any(radius2<=0) or np.any(den<=0):raise ArithmeticError('invalid size-biased shape')
    result=alpha/2*(np.log(radius2)-np.log(den))
    if not np.isfinite(result).all() or np.any(result>0):raise ArithmeticError('Cauchy dominance numeric violation')
    return result

def negative_binomial_trials(log_accept_batch,rng,successes,cap,batch=64):
    successes=positive_int(successes,'successes');cap=positive_int(cap,'cap',successes);batch=positive_int(batch,'batch')
    used=accepted=0;max_excess=-np.inf
    while used<cap:
        n=min(batch,cap-used);la=np.asarray(log_accept_batch(n),float)
        if la.shape!=(n,) or not np.isfinite(la).all() or np.any(la>0):
            raise JointReferenceFailure('invalid proposal acceptance',{'proposed':used,'accepted':accepted})
        max_excess=max(max_excess,float(la.max()));u=rng.random(n)
        if np.any(u<=0):raise JointReferenceFailure('uniform endpoint',{'proposed':used,'accepted':accepted})
        ids=np.flatnonzero(np.log(u)<=la);needed=successes-accepted
        if len(ids)>=needed:
            exact=used+int(ids[needed-1])+1
            return exact,{'statistical_trials_through_Lth_success':exact,'generated_proposals':used+n,
                'required_successes':successes,'accepted_including_unused':accepted+len(ids),
                'max_log_acceptance':max_excess,'cap':cap,'status':'completed'}
        used+=n;accepted+=len(ids)
    raise JointReferenceFailure('negative-binomial cap; zero family',{'proposed':used,'accepted':accepted,'required':successes,'cap':cap})

def inverse_pair(w_source,w_target,ms,mt,a,c,tau,logk,g,seed,*,alpha=4.,successes=64,
                 numerator_draws=1024,cap=1000000,identity_shape=False):
    if alpha!=4.:raise ValueError('first prototype fixesalpha4; no silent parameter search')
    if not 0<a<1 or c<=0 or tau<=0 or not np.isfinite([a,c,tau,logk]).all():raise ValueError('invalid fixed meta')
    successes=positive_int(successes,'successes');numerator_draws=positive_int(numerator_draws,'numerator_draws',2);cap=positive_int(cap,'cap',successes)
    start=time.perf_counter();r=alpha/2;b=r*a
    s0=TiltEnvelope(w_source,0,LOG_CENTER+ms);sb=TiltEnvelope(w_source,b,LOG_CENTER+ms)
    t0=TiltEnvelope(w_target,0,LOG_CENTER+mt);tr=TiltEnvelope(w_target,r,LOG_CENTER+mt)
    tb=TiltEnvelope(w_target,r*(1-a),LOG_CENTER+mt)
    rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(9)]
    zs,zsr=s0.normalizer_ratio(rng[0],numerator_draws);zt,ztr=t0.normalizer_ratio(rng[1],numerator_draws)
    logB=log_dominating_moment(alpha)
    logterms=np.array([s0.logC+tr.logC,logk+sb.logC+tb.logC]);logmix=float(logsumexp(logterms))
    prob_first=float(np.exp(logterms[0]-logmix))
    def boundary_accept(n):
        first=rng[2].random(n)<prob_first;x=np.empty(n);y=np.empty(n)
        for choose,sl,tl in [(first,s0,tr),(~first,sb,tb)]:
            nn=int(choose.sum());x[choose]=sl.proposal(rng[3],nn);y[choose]=tl.proposal(rng[4],nn)
        u=(x-s0.center)-(y-t0.center);z=(u-c)/tau
        logg=-np.logaddexp(0.,z);lognot=-np.logaddexp(0.,-z)
        target=s0.log_density(x)+tr.log_density(y)+lognot
        bridge=logk+sb.log_density(x)+tb.log_density(y)+logg
        numerator=np.logaddexp(target,bridge)
        denominator=np.logaddexp(s0.log_bound(x)+tr.log_bound(y),logk+sb.log_bound(x)+tb.log_bound(y))
        return numerator-denominator+shape_sizebias_log_accept(rng[5],n,g,alpha,identity_shape)
    partial={'stage':'boundary','numerator_source':zsr,'numerator_target':ztr,'alpha':alpha,'successes':successes}
    try:n0,rec0=negative_binomial_trials(boundary_accept,rng[6],successes,cap)
    except (ArithmeticError,RuntimeError,np.linalg.LinAlgError) as err:
        raise JointReferenceFailure('boundary normalizer failed',dict(partial,seconds=time.perf_counter()-start,error=repr(err),detail=getattr(err,'receipt',None))) from err
    def infinity_accept(n):
        y=tr.proposal(rng[7],n)
        return tr.log_density(y)-tr.log_bound(y)+shape_sizebias_log_accept(rng[8],n,g,alpha,identity_shape)
    # Separate acceptance RNG from proposal RNG; streams are independent of data.
    accept_inf=np.random.default_rng(np.random.SeedSequence(seed).spawn(10)[9])
    partial.update(stage='infinity',boundary=rec0)
    try:ni,reci=negative_binomial_trials(infinity_accept,accept_inf,successes,cap)
    except (ArithmeticError,RuntimeError,np.linalg.LinAlgError) as err:
        raise JointReferenceFailure('infinity normalizer failed',dict(partial,seconds=time.perf_counter()-start,error=repr(err),detail=getattr(err,'receipt',None))) from err
    logR0=np.log(zs)+np.log(zt)+s0.logC+t0.logC-logB-logmix+np.log(n0/successes)
    logRi=np.log(zt)+t0.logC-logB-tr.logC+np.log(ni/successes)
    logr=min(logR0,logRi)
    if not np.isfinite([logR0,logRi,logr]).all():raise ArithmeticError('nonfinite inverse budget')
    return logr,{'method':'INDEPENDENT_UNBIASED_INVERSES_THEN_MIN_NOT_PLUGIN',
        'log_inverse_boundary':float(logR0),'log_inverse_infinity':float(logRi),'log_inverse_used':float(logr),
        'numerator_source':zsr,'numerator_target':ztr,'boundary':rec0,'infinity':reci,
        'log_dominating_V_moment':logB,'log_boundary_envelope':logB+logmix,
        'log_infinity_envelope':logB+tr.logC,'seconds':time.perf_counter()-start,
        'identity_shape_DIAGNOSTIC_ONLY':identity_shape,'alpha':alpha,'successes':successes,
        'not_claimed':'finite floating-point interval certificate or statistical acceptance'}

def endpoint_inverse(w_source,w_target,ms,mt,a,g,seed,*,successes=64,numerator_draws=1024,cap=1000000):
    """Same direct-e reference for target(a0), source(a1), or FIXED bridge."""
    successes=positive_int(successes,'successes');numerator_draws=positive_int(numerator_draws,'numerator_draws',2);cap=positive_int(cap,'cap',successes)
    if not np.isfinite(a) or not 0<=a<=1:raise ValueError('fixed source weight in[0,1]')
    start=time.perf_counter();r=2.;rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    active=[];receipts=[];logpref=0.
    for j,(w,center,weight) in enumerate([(w_source,LOG_CENTER+ms,a),(w_target,LOG_CENTER+mt,1-a)]):
        if weight==0:continue
        zero=TiltEnvelope(w,0,center);tilted=TiltEnvelope(w,r*weight,center)
        ratio,rec=zero.normalizer_ratio(rng[j],numerator_draws)
        logpref+=np.log(ratio)+zero.logC-tilted.logC
        active.append((j,tilted));receipts.append({'bank':j,'normalizer':rec,'log_tilt_envelope':tilted.logC})
    def accept(n):
        v=shape_sizebias_log_accept(rng[4],n,g,4.)
        for j,t in active:
            y=t.proposal(rng[j+2],n);v=v+t.log_density(y)-t.log_bound(y)
        return v
    try:nn,rec=negative_binomial_trials(accept,rng[5],successes,cap)
    except (ArithmeticError,RuntimeError,np.linalg.LinAlgError) as err:
        raise JointReferenceFailure('endpoint reference failed',{'stage':'endpoint','source_weight':a,'numerators':receipts,
            'seconds':time.perf_counter()-start,'error':repr(err),'detail':getattr(err,'receipt',None)}) from err
    logr=float(logpref-log_dominating_moment(4.)+np.log(nn/successes))
    return logr,{'method':'UNBIASED_ENDPOINT_INVERSE','source_weight':a,'numerators':receipts,'trials':rec,
                 'log_inverse':logr,'seconds':time.perf_counter()-start}
