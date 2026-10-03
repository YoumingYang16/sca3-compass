import numpy as np
import pytest
from selection_profile import Profile,select,inner_rank

@pytest.mark.parametrize('seed',range(8))
def test_exact_profile_vs_all_event_limits(seed):
    r=np.random.default_rng(seed);m=37
    p=Profile(r.normal(size=m),r.normal(size=m),r.normal(size=m),32,4,
              r.normal(size=63),r.normal(size=63))
    for k in range(1,65):
        # A brute sample-wise reference rule evaluated at every switch. Include
        # both limits; profiles only decrease between switches, no grid needed.
        points=np.r_[0,p.switch[p.switch>=0]]
        brute=max([p.direct_count(k,float(s)) for s in points]+
                  [p.direct_count(k,float(s),True) for s in points]+[p.direct_count(k,np.inf)])
        assert p.count(k)==brute,(seed,k)
        for s in r.uniform(0,20,size=10):assert p.direct_count(k,s)<=p.count(k)

def test_selector_only_observed_values():
    a=select(0.,0.,32,32,5.)
    assert not a['borrow'] and a['target_weight']==1.
    b=select(0.,np.log(4.),32,4,4.)
    assert b['borrow'] and b['target_weight']==4/36

def test_rank_ties_and_nonpositive():
    assert np.array_equal(inner_rank(np.array([-1.,0.,1.,2.,3.]),np.array([0.,1.,1.])),[4,4,3,1,1])

def test_simultaneous_switches_no_artificial_overlap():
    p=Profile(np.array([0.,0.,1.,1.]),np.zeros(4),np.array([1.,2.,1.,2.]),4,4,
              np.array([.1,1.,2.]),np.array([.1,.5,1.]))
    for k in range(1,5):
        all_s=np.r_[0,p.switch[p.switch>=0],10.]
        assert p.count(k)==max([p.direct_count(k,s,side) for s in all_s for side in [False,True]])

def test_rank_support_and_monotonicity():
    r=np.random.default_rng(500)
    p=Profile(r.normal(size=255),r.normal(size=255),r.normal(size=255),32,4,r.normal(size=255),r.normal(size=255))
    v=p.calibrate_rank(np.arange(1,257))
    assert np.all(np.diff(v)>=0) and np.all(v*256==np.floor(v*256)) and v[-1]==1.
