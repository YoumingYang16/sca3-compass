from itertools import product
import numpy as np
import pytest
from scipy.stats import binom
from sca3_compass.robustness_confirmation_bounds import binary_kl,kl_interval,envelope_interval

@pytest.mark.parametrize('n',[10,50,200])
def test_binomial_exact_coverage(n):
    intervals=[kl_interval(k/n,n,.01) for k in range(n+1)]
    for p in [.001,.01,.05,.2,.5,.8,.99]:
        probs=binom.pmf(np.arange(n+1),n,p)
        assert sum(probs[k] for k,(lo,hi) in enumerate(intervals) if p<lo)<=.01+1e-12
        assert sum(probs[k] for k,(lo,hi) in enumerate(intervals) if p>hi)<=.01+1e-12

def test_zero_fdp_upper_exact_chernoff():
    lo,hi=kl_interval(0,1000,.00001)
    assert lo==0;assert hi==pytest.approx(1-.00001**.001)

def test_fractional_fdp_not_binomial_counts():
    vals=[0,.25,1];probs=[.5,.3,.2];mu=np.dot(vals,probs);below=above=0.
    for ids in product(range(3),repeat=5):
        prob=np.prod([probs[i] for i in ids]);lo,hi=kl_interval(np.mean([vals[i] for i in ids]),5,.1)
        below+=prob*(mu<lo);above+=prob*(mu>hi)
    assert below<=.1;assert above<=.1

def test_envelope_order_and_pairing():
    xs=[np.array([[.1,.4],[.3,.2],[.2,.3]])]*2
    r=envelope_interval(xs,[1,1],.01,.01)
    assert r['sample_envelope_difference']==pytest.approx(.2)
    assert r['fixed_development_comparator_difference']==pytest.approx(.3)
    assert r['sum_weight_squared_over_n']==pytest.approx(1/6)
    assert r['lower']<=.2<=r['upper']

def test_envelope_concavity_exhaustive():
    # Equal-probability independent family draws, two correlated baseline gaps.
    atoms=np.array([[-.5,.5],[.5,-.5]])
    estimates=[atoms[list(ids)].mean(0).min() for ids in product(range(2),repeat=4)]
    assert np.mean(estimates)<=atoms.mean(0).min()

def test_invalid_inputs():
    with pytest.raises(ValueError):envelope_interval([np.array([[2.],[0.]])],[0],.01,.01)
