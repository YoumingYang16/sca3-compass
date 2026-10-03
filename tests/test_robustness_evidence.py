import numpy as np
from scipy.integrate import quad
from scipy.stats import t, norm

from sca3_compass.robustness_evidence import directional_likelihood_e, partial_conjunction_e


def test_directional_e_expectation_is_one_at_boundary_and_lower_at_negative_location():
    for df in [3.,25.,np.inf]:
        for scale in [.4,1.,3.]:
            for null_mean in [0.,-1.]:
                distribution=norm if np.isinf(df) else t(df)
                def integrand(x):
                    evidence=float(directional_likelihood_e(x,scale,df))
                    return evidence*distribution.pdf((x-null_mean)/scale)/scale
                expectation=quad(integrand,0,30,epsabs=1e-9)[0]+quad(integrand,30,np.inf,epsabs=1e-9)[0]
                if null_mean==0:
                    np.testing.assert_allclose(expectation,1,atol=1e-7)
                else:
                    assert 0<=expectation<1


def test_pc_e_not_more_than_average_of_any_three_null_studies():
    rng=np.random.default_rng(281)
    e=np.exp(rng.normal(size=(20,2,4))*3)
    result=partial_conjunction_e(e,2)
    for omitted in range(4):
        assert np.all(result<=np.delete(e,omitted,axis=-1).mean(axis=-1)+1e-12)


def test_directional_e_shape_sign_and_mixture():
    mean=np.array([[[2.,-2.]],[[1.,-1.]]])
    scale=np.array([.5,2.]).reshape(2,1,1)
    both=directional_likelihood_e(mean,scale,25,effects=(2,4))
    singles=(directional_likelihood_e(mean,scale,25,effects=(2,))+directional_likelihood_e(mean,scale,25,effects=(4,)))/2
    assert both.shape==mean.shape and np.all(both[:,:,1]==0)
    np.testing.assert_allclose(both,singles,rtol=1e-14)
