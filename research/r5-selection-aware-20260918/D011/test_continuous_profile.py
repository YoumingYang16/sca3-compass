import numpy as np
import pytest
from continuous_profile import TailMap,ContinuousProfile

def test_inverse_and_unsaturated_tail():
    r=np.sort(np.random.default_rng(3).normal(size=4095));f=TailMap(r)
    w=np.geomspace(.01,1e5,200);y=f.forward(w);pos=y>0
    assert np.all(np.diff(y)>=0)
    assert np.allclose(np.array([f.inverse_log(x) for x in y[pos]]),np.log(w[pos]))
    assert y[-1]>y[-2]>y[-3]

@pytest.mark.parametrize('seed',range(8))
def test_exact_continuous_event_scan(seed):
    r=np.random.default_rng(seed);m=47
    p=ContinuousProfile(r.normal(size=m)/3,r.normal(size=m),r.normal(size=m),32,4,
                        r.normal(size=127),r.normal(size=127))
    for x in [.01,.3,1.,2.,4.,8.,12.]:
        h=p.switch[p.switch>=0]
        tau=2*(p.logbridge-p.mb.inverse_log(x))/p.a
        pts=np.unique(np.r_[0,h,tau[(tau>=0)&np.isfinite(tau)]])
        # Exact at every switch/crossing; evaluate one point in each open cell.
        ss=np.r_[pts,(pts[:-1]+pts[1:])/2,pts[-1]+2]
        exact=max(p.direct_score_count(x,s) for s in ss)
        assert p.count_score(x)==exact

def test_perfect_tie_switch_count():
    r=np.linspace(-2,2,129);p=ContinuousProfile(np.zeros(6),np.zeros(6),np.ones(6),4,4,r,r)
    for x in [.1,.5,1.]:
        assert p.count_score(x)==max(p.direct_score_count(x,s) for s in [0,p.c,p.c+1,1e5])
