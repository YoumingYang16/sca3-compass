import numpy as np
import pytest
from soft_drift_closure import SoftDriftClosure

def fixture():
    r=np.random.default_rng(370)
    return SoftDriftClosure(r.normal(size=1023),r.normal(size=1023),r.normal(size=1023),.8,.7,
                           r.standard_t(5,4095),r.standard_t(20,4095))

def test_legal_drift_path_and_target_limit():
    p=fixture();old=p.draw_scores(0)
    for s in np.r_[np.linspace(0,20,301),1e6]:
        new=p.draw_scores(s);assert np.all(new<=old+2e-14);old=new
    assert np.array_equal(old,p.logtarget-p.logqt)

def test_exact_maximum_and_minimal_majorant_identity():
    p=fixture();u=np.linspace(-20,20,151);t=1.3
    zt=t/np.exp(p.logqt);zb=t*np.exp(-p.a*u/2-p.logqb)
    from scipy.special import expit
    raw=zt+expit((p.c-u)/p.tau)*(zb-zt)
    assert np.allclose(np.exp(p.score(np.log(t),u)),np.maximum(zt,raw),rtol=5e-15)
    # Function limit is part of the proof; finite grid merely checks majorization.
    for i in range(len(u)):
        required=max(zt,float(raw[i:].max()))
        assert np.exp(p.score(np.log(t),u[i]))>=required-1e-14*max(1.,required)

def test_rank_has_full_rule_not_selected_reference():
    p=fixture();p.observed_u=.4;w=np.array([-1.,0.,.1,1.,3.,10.])
    expected=np.ones_like(w);sc=p.score(np.log(w[2:]),.4)
    expected[2:]=(1+np.sum(p.draw_scores(0)[:,None]>=sc[None,:],axis=0))/(p.m+1)
    assert np.array_equal(p.pvalues(w,False),expected)
    assert np.array_equal(p.pvalues(w,True),expected)
    assert np.all(np.diff(expected)<=0)

def test_guard_and_inner_fallback():
    p=SoftDriftClosure(np.zeros(7),np.zeros(7),np.arange(-3.,4.),.5,1.,-np.ones(31),-np.ones(31))
    assert all(p.normalizer_fallback.values())
    with pytest.raises(ValueError):p.pvalues(np.ones(2),False)
    p.observed_u=1e5;assert p.pvalues(np.ones(2),False).shape==(2,)
