import numpy as np
from sca3_compass.robustness_calibrators import by_calibrator, focused_calibrator
from sca3_compass.molecular_methods import ebh, fdr_adjust


def test_by_step_integrates_one_and_reproduces_by():
    rng=np.random.default_rng(135927)
    for m in [2,10,512]:
        h=np.sum(1/np.arange(1,m+1))
        cut=.05*np.arange(1,m+1)/(m*h)
        midpoint=(np.r_[0,cut[:-1]]+cut)/2
        assert abs(np.sum(np.diff(np.r_[0,cut])*by_calibrator(midpoint,m))-1)<1e-14
        for _ in range(30):
            p=rng.uniform(size=m)**4
            np.testing.assert_array_equal(ebh(by_calibrator(p,m),.05),fdr_adjust(p)<=.05)


def test_focused_integrates_one_on_exact_breakpoints():
    m=512
    for tau,lam in [(.0003,.5),(.001,.5),(.001,.8),(.003,.5)]:
        h=np.sum(1/np.arange(1,m+1))
        cut=np.unique(np.r_[0,.05*np.arange(1,m+1)/(m*h),tau,1])
        mid=(cut[1:]+cut[:-1])/2
        e=focused_calibrator(mid,m,tau,lam)
        assert abs(np.sum(np.diff(cut)*e)-1)<1e-14
        assert np.all(np.diff(e)<=0)


def test_capped_reshaping_integrates_one_and_preserves_sparse_by():
    m=512
    rng=np.random.default_rng(29172)
    for cap in [16,64,128]:
        h=np.sum(1/np.arange(1,cap+1))
        cut=np.unique(np.r_[0,.05*np.arange(1,cap+1)/(m*h),.001,1])
        mid=(cut[1:]+cut[:-1])/2
        for fraction in [0.,.2,.5]:
            e=focused_calibrator(mid,m,.001,fraction,cap=cap)
            assert abs(np.sum(np.diff(cut)*e)-1)<1e-14
        for _ in range(100):
            p=rng.uniform(size=m)
            p[:5]*=1e-4
            old=fdr_adjust(p)<=.05
            new=ebh(by_calibrator(p,m,cap=cap),.05)
            if old.sum()<=cap:
                assert np.all(new[old])
