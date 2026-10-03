import numpy as np
import pytest
from scipy.integrate import quad
from triangular_calibration import summaries,ConditionalTriangular
from ancillary_calibration import log_pivots


def test_same_radius_and_triangular_invariance():
    rng=np.random.default_rng(7654);cal=rng.normal(size=(16,4,6))
    lq,a=summaries(cal)
    np.testing.assert_allclose(lq,np.log(2)+log_pivots(cal),atol=3e-13,rtol=3e-13)
    mat=np.array([[2,0,0,0],[.7,1,0,0],[-.3,.2,.5,0],[.2,-.8,.5,3.]])
    lq2,a2=summaries(np.einsum('ij,njk->nik',mat,cal)*np.exp(np.linspace(-20,20,16))[:,None,None])
    np.testing.assert_allclose(lq,lq2,atol=3e-13,rtol=3e-13)
    np.testing.assert_allclose(a,a2,atol=3e-13,rtol=3e-13)


def test_density_derivative_and_logconcavity():
    law=ConditionalTriangular(np.array([-1.,-.1,.2,.9]),np.tile([.04,.3,.65],(4,1)))
    y=np.linspace(-12,12,101);h=1e-5
    numeric=(law.log_density(y+h)-law.log_density(y-h))/(2*h)
    np.testing.assert_allclose(law.derivative(y),numeric,atol=2e-8,rtol=1e-7)
    assert np.all(np.diff(law.derivative(y))<0)
    assert np.all(law.log_density(y)-law.shift<=law.log_envelope(y))
    assert abs(law.derivative(-100)-8)<1e-10
    assert abs(law.derivative(100)+10)<1e-10


def test_sampler_center_and_failure_policy():
    law=ConditionalTriangular(np.zeros(4),np.tile([.04,.3,.65],(4,1)))
    f=lambda y:np.exp(law.log_density(y)-law.shift)
    norm=quad(f,-40,40,epsabs=1e-11)[0]
    mean=quad(lambda y:y*f(y),-40,40,epsabs=1e-11)[0]/norm-1
    x,r=law.draw(np.random.default_rng(223),8192)
    assert abs(x.mean()-mean)<6*x.std(ddof=1)/np.sqrt(len(x))
    assert r['error_center_subtracted']==1 and r['max_log_envelope_excess']<=0
    with pytest.raises(RuntimeError):law.draw(np.random.default_rng(224),100,proposal_cap=1)
    with pytest.raises(ValueError):ConditionalTriangular(np.zeros(4),np.tile([0,.3,.65],(4,1)))
