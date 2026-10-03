import numpy as np
from sca3_compass.robustness_cone import cone_weights3, cone_tail, cone_partial_conjunction
from sca3_compass.robustness_universal import orthant_distance


def test_independent_orthant_binomial_weights():
    np.testing.assert_allclose(cone_weights3(np.eye(3)),np.array([1,3,3,1])/8,atol=1e-15)


def test_cone_tail_against_monte_carlo_projection():
    rng=np.random.default_rng(802)
    shape=.65*np.ones((3,3))+.35*np.eye(3)
    x=rng.normal(size=(200000,3))@np.linalg.cholesky(shape).T
    q=orthant_distance(x,shape)
    for df in [np.inf,25.]:
        observed=q if np.isinf(df) else q/(rng.chisquare(df,size=len(x))/df)
        for threshold in [2,5,10]:
            empirical=np.mean(observed>threshold)
            theoretical=float(cone_tail(threshold,shape,df))
            assert abs(empirical-theoretical)<5*np.sqrt(theoretical*(1-theoretical)/len(x))+.00005


def test_cone_partial_conjunction_monotone_positive_shift():
    rng=np.random.default_rng(941)
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    x=rng.normal(size=(30,2,4))*3
    old=cone_partial_conjunction(x,np.ones(30),[shape,shape],25)
    new=cone_partial_conjunction(x+1,np.ones(30),[shape,shape],25)
    assert np.all(old>=new-1e-12)
