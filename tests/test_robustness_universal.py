import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_t

from sca3_compass.robustness_universal import orthant_distance, pc_null_distance, alternatives, universal_pc_e


def test_orthant_projection_against_generic_constrained_optimizer():
    rng=np.random.default_rng(493)
    for dimension in [2,3,4]:
        a=rng.normal(size=(dimension,dimension))
        covariance=a@a.T+np.eye(dimension)
        inverse=np.linalg.inv(covariance)
        for x in rng.normal(size=(20,dimension))*4:
            optimum=minimize(lambda mu:((x-mu)@inverse@(x-mu),-2*inverse@(x-mu)),np.minimum(x,0),
                jac=True,bounds=[(None,0)]*dimension,method="L-BFGS-B",options={"ftol":1e-13,"gtol":1e-9})
            np.testing.assert_allclose(orthant_distance(x,covariance),optimum.fun,atol=1e-8,rtol=1e-8)


def test_partial_null_distance_zero_for_one_positive_coordinate():
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    assert pc_null_distance(np.array([8.,-1.,0.,-4.]),shape)==0
    assert pc_null_distance(np.ones(4)*3,shape)>0


def test_universal_density_ratio_against_scipy_full_density():
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    means=np.array([[[2,1,-1,3],[-2,-1,1,-3]],[[4,3,2,1],[-4,-3,-2,-1]]],float)
    variance=np.array([.8,2.])
    output=universal_pc_e(means,variance,[shape,shape],25)
    locations,weights=alternatives()
    for g in range(2):
        for sign in range(2):
            x=means[g,sign]
            numerator=sum(w*multivariate_t.pdf(x,loc=loc,shape=variance[g]*shape,df=25) for w,loc in zip(weights,locations))
            distance=pc_null_distance(x,shape)/variance[g]
            center_density=multivariate_t.pdf(np.zeros(4),shape=variance[g]*shape,df=25)
            denominator=center_density*(1+distance/25)**(-29/2)
            np.testing.assert_allclose(output[g,sign],numerator/denominator,rtol=1e-12)
