import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import chi2, ks_2samp

from sca3_compass.molecular_envelope import (
    correlation_envelope,
    efilter,
    efilter_adjusted,
    envelope_pvalues,
    equicorrelated_bank,
    gaussian_bank,
    rank_tail,
)
from sca3_compass.molecular_methods import equicorrelation


def test_pair_pivot_matches_hand_calculation():
    x=np.array([[1.,0],[0,1],[2,-1],[1,1]])
    d=x[:,0]-x[:,1]
    expected=1-np.sum((d-d.mean())**2)/(2*chi2.ppf(.01,3))
    result=correlation_envelope(x,.01)
    assert result.rho_lower == pytest.approx(max(-1,expected))
    assert result.fallback


def test_identical_pipelines_have_degenerate_envelope():
    x=np.tile(np.arange(12)[:,None],(1,6)).astype(float)
    env=correlation_envelope(x,.001)
    assert env.rho_lower == 1
    assert not env.fallback


def test_negative_lower_bound_uses_bonferroni_not_clamped_gaussian():
    rng=np.random.default_rng(11)
    env=correlation_envelope(rng.normal(size=(8,4)),.001)
    z=np.array([[1.,2.,3.,4.]])
    np.testing.assert_allclose(envelope_pvalues(z,env,1023,rng),4*ndtr(-4))


@pytest.mark.parametrize("rho",[0,.4,.95,1])
def test_fast_max_sampler_matches_full_gaussian(rho):
    rng=np.random.default_rng(771)
    a=equicorrelated_bank(6,rho,50000,rng)
    b=gaussian_bank(equicorrelation(6,rho),50000,rng)
    assert ks_2samp(a,b).statistic < .015


def test_rank_ties_and_no_zero_pvalues():
    np.testing.assert_allclose(rank_tail(np.array([0.,1.,1.,2.]),np.array([-2.,1.,3.])),[1,.8,.2])


def test_confidence_coverage_in_independent_gaussian_calibration():
    rng=np.random.default_rng(631)
    covered=[]
    cov=equicorrelation(4,.8)
    for _ in range(2000):
        env=correlation_envelope(rng.multivariate_normal(np.zeros(4),cov,size=32),.05)
        covered.append(np.all(env.lower_matrix <= cov+1e-12))
    assert np.mean(covered)>.94


def test_efilter_adjustment_against_quadratic_reference():
    s=np.array([1.,30,50,30,.5])
    f=np.array([3.,100,50,30,200])
    order=np.argsort(-s,kind="stable")
    vals=[]
    for j in range(len(s)):
        vals.append(max((h+1)*s[order[h]]/np.sum(f>=s[order[h]]) for h in range(j,len(s))))
    expected=np.empty_like(s)
    expected[order]=vals
    np.testing.assert_allclose(efilter_adjusted(s,f),expected)


def test_efilter_no_signal_and_obvious_signal():
    assert not efilter(np.ones((100,4)),2,.025).any()
    assert efilter(np.full((100,4),1e-12),2,.025).all()


@pytest.mark.parametrize("x,delta",[(np.ones((1,3)),.01),(np.ones((5,1)),.01),(np.ones((5,3)),0),(np.array([[0,np.nan],[1,1]]),.1)])
def test_bad_calibration_rejected(x,delta):
    with pytest.raises(ValueError):
        correlation_envelope(x,delta)
