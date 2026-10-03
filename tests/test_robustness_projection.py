import numpy as np
from scipy.stats import norm
from sca3_compass.robustness_projection import positive_direction,projection_pc


def test_direction_nonnegative_normalized_and_identity_optimum():
    profile=np.array([2.,0.,1.])
    direction=positive_direction(profile,np.eye(3))
    np.testing.assert_allclose(direction,profile/np.linalg.norm(profile),atol=1e-14)
    shape=.65*np.ones((3,3))+.35*np.eye(3)
    direction=positive_direction(profile,shape)
    assert np.min(direction)>=0
    np.testing.assert_allclose(direction@shape@direction,1.,atol=1e-14)


def test_direction_kkt_certificate_on_deterministic_profiles():
    # KKT certifies the complete nonnegative linear class, not a random grid.
    for rho in [-.25, 0., .65, .95]:
        shape=rho*np.ones((3,3))+(1-rho)*np.eye(3)
        for profile in [np.array([1.,0.,0.]),np.array([2.,0.,1.]),np.array([1.,1.,1.])]:
            direction=positive_direction(profile,shape)
            optimum=direction*(profile@direction)
            gradient=shape@optimum-profile
            assert np.min(gradient)>-2e-12
            assert np.min(optimum)>=0
            np.testing.assert_allclose(optimum@gradient,0.,atol=2e-12)
            np.testing.assert_allclose(direction@shape@direction,1.,atol=2e-12)


def test_unconstrained_gls_can_reverse_a_composite_null_mean():
    shape=.8*np.ones((3,3))+.2*np.eye(3)
    profile=np.array([1.,0.,0.])
    null_mean=np.array([0.,-5.,-5.])
    unrestricted=np.linalg.solve(shape,profile)
    unrestricted/=np.sqrt(unrestricted@shape@unrestricted)
    restricted=positive_direction(profile,shape)
    assert unrestricted@null_mean>8.
    assert restricted@null_mean<=0.
    # This is a deterministic mean/geometry check, not a simulated FDR claim.


def test_two_study_profile_reduces_to_max_of_two_one_sided_p():
    x=np.random.default_rng(88).normal(size=(80,4))
    x[:20]+=4
    m=np.stack((x,-x),axis=1)
    profile=np.tile([1.,1.,0.,0.],(2,1))
    value=projection_pc(m,np.ones(len(m)),np.eye(4),np.inf,profile)
    # The subset containing both positive coordinates cannot be limiting
    # when both are positive: normalized sum >= each smaller coordinate.
    expected=np.maximum(norm.sf(m[:,:,0]),norm.sf(m[:,:,1]))
    expected=np.where(np.minimum(m[:,:,0],m[:,:,1])>0,expected,1.)
    np.testing.assert_allclose(value,expected,atol=1e-14)


def test_composite_null_size_and_monotonicity_known_student():
    rng=np.random.default_rng(663)
    n=60000
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    x=rng.normal(size=(n,4))@np.linalg.cholesky(shape).T
    x/=np.sqrt(rng.chisquare(23,n)/23)[:,None]
    x[:,0]+=100
    m=np.stack((x,-x),axis=1)
    profiles=np.array([[1.,1.,0.,0.],[1.,1.,1.,1.]])
    for options in [{},{'bonferroni':True},{'support_simes':True}]:
        value=projection_pc(m,np.ones(n),shape,23,profiles,**options)
        assert (value[:,0]<=.05).mean()<.0535
        lower=m.copy();lower[:,0,1:]-=1
        reduced=projection_pc(lower,np.ones(n),shape,23,profiles,**options)
        assert np.all(reduced[:,0]>=value[:,0]-1e-14)
