import numpy as np
from scipy.integrate import quad
from scipy.stats import t
from scipy.special import ndtr

from sca3_compass.robustness_conjunction import diagonal_joint_tail, hunter_partial_conjunction


def test_gaussian_independent_pair_is_product():
    h=np.array([1.,2.,4.,6.])
    np.testing.assert_allclose(diagonal_joint_tail(h,0,np.inf,96),ndtr(-h)**2,rtol=1e-5,atol=1e-15)


def test_student_pair_matches_independent_adaptive_quadrature():
    for df in [3,25,80]:
        for rho in [-.3,.65,.95]:
            for h in [1.,3.,5.]:
                integrand=lambda x: t.pdf(x,df)*t.sf((h-rho*x)/np.sqrt((df+x*x)*(1-rho*rho)/(df+1)),df+1)
                expected=quad(integrand,h,np.inf,epsabs=1e-13,epsrel=1e-9)[0]
                actual=diagonal_joint_tail(h,rho,df,96)
                np.testing.assert_allclose(actual,expected,rtol=8e-5,atol=1e-12)


def test_hunter_is_between_fixed_max_and_bonferroni():
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    rng=np.random.default_rng(271)
    z=np.abs(rng.normal(size=(16,2,4))*3)+2
    actual=hunter_partial_conjunction(z,[shape,shape],25)
    h=np.sort(z,axis=-1)[:,:,2]
    p=t.sf(h,25)
    assert np.all(actual>=p-1e-12) and np.all(actual<=3*p+1e-12)
