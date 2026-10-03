import numpy as np
from scipy.stats import t

from sca3_compass.robustness_methods import RadialFit, contrasts, student_tail, tyler_shape, mixture_tail


def test_helmert_annihilates_all_study_effects():
    h=contrasts(6)
    np.testing.assert_allclose(h.T@h,np.eye(5),atol=1e-14)
    np.testing.assert_allclose(np.ones(6)@h,0,atol=1e-14)
    means=np.array([0,2,-3,5])
    np.testing.assert_allclose((means[:,None]*np.ones((1,6)))@h,0,atol=1e-14)


def test_conditional_student_formula_independent_scalar_calculation():
    fit=RadialFit(.8,.6,5,0,True,1,False)
    means=np.array([1.,2.,3.])
    q=np.array([2.,8.,30.])
    expected=t.sf(means/np.sqrt(.8*(5*.6+q)/25),25)
    np.testing.assert_allclose(student_tail(means,.8,fit,q,20),expected,atol=1e-14)


def test_tyler_is_invariant_to_individual_positive_radial_rescaling():
    rng=np.random.default_rng(924)
    x=rng.normal(size=(300,4))
    a,da=tyler_shape(x)
    b,db=tyler_shape(x*np.exp(rng.normal(size=(300,1))*3))
    assert da["converged"] and db["converged"]
    np.testing.assert_allclose(a,b,atol=1e-12)


def test_gaussian_conditional_does_not_use_residual_scale():
    fit=RadialFit(.8,1,1e8,0,True,1,True)
    np.testing.assert_array_equal(student_tail(np.array([2.]),1,fit,np.array([3e20]),20),
                                  student_tail(np.array([2.]),1,fit))


def test_nonpositive_student_tail_conservative_extension():
    fit=RadialFit(.8,.6,5,0,True,1,False)
    assert np.all(student_tail(np.array([-4.,0.]),.8,fit,np.array([2.,80.]),20)==1.)


def test_calibration_can_fit_infinite_variance_finite_mean_student():
    from sca3_compass.robustness_methods import fit_calibration
    rng=np.random.default_rng(192813)
    x=rng.normal(size=(8000,6))*np.sqrt(1.5/rng.chisquare(1.5,size=(8000,1)))
    fit=fit_calibration(x)
    assert fit.converged and not fit.gaussian_bic_selected and 1.3<fit.df<1.7


def test_mixture_tail_convex_probability_and_sign_symmetry():
    rng=np.random.default_rng(4871)
    means=rng.normal(size=(256,2,4))*20
    grid=np.geomspace(.001,1000,48)
    weights=rng.dirichlet(np.ones(48)*.1)
    q=np.exp(rng.normal(size=256)*4)
    p=mixture_tail(means,q,20,.8,grid,weights)
    minus=mixture_tail(-means,q,20,.8,grid,weights)
    assert p.shape==means.shape and np.all((p>=0)&(p<=1))
    np.testing.assert_allclose(p+minus,1,atol=2e-15)


def test_mixture_point_mass_is_gaussian_regardless_of_conditioning():
    from scipy.special import ndtr
    means=np.array([[[1.,-2.]],[[3.,-4.]]])
    q=np.array([.01,10000.])
    p=mixture_tail(means,q,20,.8,np.array([2.]),np.array([1.]))
    np.testing.assert_allclose(p,ndtr(-means/np.sqrt(1.6)),atol=1e-14)
