import numpy as np
import pytest
from scipy.special import digamma,polygamma
from scipy.integrate import quad
from ancillary_calibration import ConditionalLogF,log_pivots
from r5_common import model
from selection_profile import LOG_CENTER

def test_observed_pivot_matches_frozen_geometric_scale():
    r=np.random.default_rng(23);a=r.normal(size=(12,4,6))
    assert np.isclose(np.exp(log_pivots(a).mean()-LOG_CENTER),model.geometric_kappa(a),rtol=1e-13)

@pytest.mark.parametrize('n',[4,12,32])
def test_equal_residuals_exact_beta_prime_identity(n):
    p=ConditionalLogF(np.zeros(n));v,receipt=p.draw(np.random.default_rng(4100+n),40000)
    exact=digamma(2*n)-digamma(n)-np.log(2.)-LOG_CENTER
    variance=polygamma(1,2*n)+polygamma(1,n)
    assert abs(v.mean()-exact)<6*np.sqrt(variance/len(v))
    assert abs(v.var()-variance)<.04*variance
    assert receipt['acceptance_fraction']>.3

@pytest.mark.parametrize('w',[[-3,-1,1,3],[-.2,-.1,0,.3],[-12,-8,-2,22]])
def test_nonnormalized_quadrature_and_envelope(w):
    p=ConditionalLogF(w);lo=p.mode-80;hi=p.mode+80
    f=lambda x:np.exp(p.log_density(x)-p.shift)
    z=quad(f,lo,hi,epsabs=1e-10)[0]
    mean=quad(lambda x:x*f(x),lo,hi,epsabs=1e-10)[0]/z
    var=quad(lambda x:(x-mean)**2*f(x),lo,hi,epsabs=1e-10)[0]/z
    values,rec=p.draw(np.random.default_rng(78),40000)
    assert abs(values.mean()+LOG_CENTER-mean)<6*np.sqrt(var/len(values))
    grid=np.linspace(lo,hi,2001)
    assert np.max(p.log_density(grid)-p.shift-p.log_envelope(grid))<1e-9

def test_failure_cap_is_explicit():
    with pytest.raises(RuntimeError):ConditionalLogF(np.zeros(4)).draw(np.random.default_rng(1),100,proposal_cap=1)

def test_broad_conditional_plateau_not_hidden_as_sampler_failure():
    p=ConditionalLogF([-80.]*4+[40.]*8)
    x,receipt=p.draw(np.random.default_rng(2),40000)
    assert receipt['acceptance_fraction']>.1
    assert abs(x.var()-1070.413722)<.04*1070.413722

def test_last_region_rounding_has_no_uninitialized_fourth_region():
    p=ConditionalLogF(np.zeros(4))
    p.region_prob=np.array([.2,.3,np.nextafter(.5,0)])
    u=np.array([0.,.199,.2,.499,.5,np.nextafter(1.,0)])
    assert np.array_equal(p.regions(u),[0,0,1,1,2,2])

def test_any_positive_envelope_excess_fails_closed(monkeypatch):
    p=ConditionalLogF(np.zeros(4))
    monkeypatch.setattr(p,'log_envelope',lambda y:p.log_density(y)-p.shift-1e-12)
    with pytest.raises(ArithmeticError,match='envelope violation'):
        p.draw(np.random.default_rng(21),20)
