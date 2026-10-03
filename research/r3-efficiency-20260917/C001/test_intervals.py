from itertools import product
import numpy as np
from bounded_intervals import interval,log_wealth,FRACTIONS

def test_exact_bernoulli_expected_wealth():
    for mu in [.05,.3,.7,.95]:
        expectation=0.
        for bits in product([0.,1.],repeat=6):
            k=sum(bits);prob=mu**k*(1-mu)**(6-k)
            expectation+=prob*np.exp(log_wealth(np.array(bits),mu))
        assert abs(expectation-1)<1e-12

def test_inversion_monotonic_and_mirrored():
    x=np.linspace(.01,.2,40)
    wealth=[log_wealth(x,m) for m in [.01,.02,.05,.1,.2,.5]]
    assert np.all(np.diff(wealth)<0)
    a=interval(x);b=interval(1-x)
    assert np.allclose(a['simultaneous_interval'],1-np.array(b['simultaneous_interval'])[::-1])
    assert a['simultaneous_interval'][0]<=x.mean()<=a['simultaneous_interval'][1]

def test_zero_one_and_paired_bounds():
    a=interval(np.zeros(1024));b=interval(np.ones(1024))
    assert a['simultaneous_interval'][0]==0 and a['simultaneous_interval'][1]<.02
    assert b['simultaneous_interval'][1]==1
    c=interval(np.full(1024,.08),-1,1)
    assert c['simultaneous_interval'][0]>0
