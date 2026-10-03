import numpy as np
import pytest
from scipy.stats import binom, t, f
from sca3_compass.robustness_domain import scale_from_absolute_scores,domain_variance,central_f_scale


def test_upper_order_statistic_has_exact_binomial_coverage():
    for n in [32,128,300]:
        for p in [.1,.25,.5]:
            _,_,info=scale_from_absolute_scores(np.arange(1,n+1),20,p,.005)
            k=info['upper_order_statistic_rank']
            assert binom.sf(k-1,n,p)<=.005
            assert k==1 or binom.sf(k-2,n,p)>.005


def test_radial_invariance_and_effect_inflation():
    rng=np.random.default_rng(3799)
    n=200000
    numerator=rng.normal(size=n)
    denominator=np.sqrt(rng.chisquare(20,size=n)/20)
    radial=np.exp(rng.normal(size=n))
    scores=np.abs(2*radial*numerator)/(radial*denominator)
    point,upper,_=scale_from_absolute_scores(scores,20)
    assert point==pytest.approx(4,rel=.04)
    np.testing.assert_allclose(scores,np.abs(2*numerator/denominator),rtol=1e-14)
    shifted=np.abs(3+2*numerator)/denominator
    assert scale_from_absolute_scores(shifted,20)[0]>point


def test_guard_exposes_gate_and_uncertainty_price():
    rng=np.random.default_rng(448)
    q=rng.chisquare(20,4000)
    m=rng.normal(size=(4000,4))
    v,info=domain_variance(m,q,20,1.,'gated_upper')
    assert v==1 and not info['trigger']
    point,point_info=domain_variance(m*10,q,20,1.,'gated_point')
    upper,upper_info=domain_variance(m*10,q,20,1.,'gated_upper')
    assert upper>=point>60 and point_info['trigger'] and upper_info['trigger']


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        scale_from_absolute_scores(np.array([1.,np.nan]*8),20)


def test_central_f_selection_normalization_recovers_null_scale():
    rng=np.random.default_rng(7751)
    x=7*rng.f(4,20,100000)
    value,info=central_f_scale(x,4,20)
    assert value==pytest.approx(7,rel=.05)
    assert info['selection_corrected_likelihood']
    # Huge nonnull effects only affect nuisance fitting, not later testing.
    contaminated=np.r_[x,np.full(len(x)//2,100000.)]
    value,_=central_f_scale(contaminated,4,20)
    assert value==pytest.approx(7,rel=.05)


def test_bidirectional_transport_is_explicit_and_can_reduce_wrong_base():
    rng=np.random.default_rng(291)
    n=8000
    q=rng.chisquare(20,n)
    means=rng.normal(size=(n,4))*.2
    fixed,info=domain_variance(means,q,20,1.,'transport_fcentral',np.eye(4))
    assert .025<fixed<.065 and info['allows_variance_decrease']
    guarded,_=domain_variance(means,q,20,1.,'always_fcentral',np.eye(4))
    assert guarded==1


def test_pooled_marginal_scale_is_exact_composite_ratio_fit():
    rng=np.random.default_rng(205399)
    m=rng.normal(size=(500,4))*np.array([1,2,3,4])
    q=rng.chisquare(20,500)
    shape=np.diag([1,4,9,16])
    ratios=((m*m/np.diag(shape))/(q[:,None]/20)).reshape(-1)
    expected,_=central_f_scale(ratios,1,20)
    actual,info=domain_variance(m,q,20,1.,'transport_pooled_fcentral',shape)
    assert actual==expected
    details=info['central_f']
    assert details['training_genes']==500 and details['training_coordinates']==2000
    assert not details['independent_coordinate_claim']
    assert details['selection_corrected_composite_likelihood']
    order=[3,1,0,2]
    same,_=domain_variance(m[:,order],q,20,1.,'transport_pooled_fcentral',shape[np.ix_(order,order)])
    assert actual==pytest.approx(same,rel=1e-6)


@pytest.mark.parametrize('multiplier',[.04,1.,9.])
def test_central_scale_anchor_is_explicit_same_region_comparison(multiplier):
    rng=np.random.default_rng(59611)
    n=1000
    q=rng.chisquare(20,n)
    m=np.sqrt(multiplier)*rng.normal(size=(n,4))
    raw,raw_info=domain_variance(m,q,20,1.,'transport_fcentral',np.eye(4))
    value,info=domain_variance(m,q,20,1.,'anchored_fcentral',np.eye(4))
    details=info['central_f']
    ratio=(m*m).sum(1)/4/(q/20)
    region=ratio[ratio<=details['threshold']]
    fixed=-np.mean(f.logpdf(region,4,20)-f.logcdf(details['threshold'],4,20))
    statistic=max(0.,2*len(region)*(fixed-raw_info['central_f']['objective']))
    assert details['likelihood_ratio_statistic']==pytest.approx(statistic)
    assert details['complexity_penalty']==np.log(len(region))
    assert value==(raw if statistic>np.log(len(region)) else 1.)
    assert info['allows_variance_decrease']
    if multiplier!=1:
        assert details['free_scale_adopted']


def test_central_scale_anchor_keeps_fixed_model_on_matched_quantile_grid():
    # A deterministic density-calibration fixture, not evidence of main-run
    # power or stochastic coverage. No simulated truth is an API argument.
    n=1000
    q=np.full(n,20.)
    radii=np.sqrt(4*f.ppf((np.arange(n)+.5)/n,4,20))
    m=np.column_stack([radii,np.zeros((n,3))])
    value,info=domain_variance(m,q,20,1.,'anchored_fcentral',np.eye(4))
    assert value==1.
    assert not info['central_f']['free_scale_adopted']


def test_pooled_marginal_scale_null_recovery_with_dependent_studies():
    rng=np.random.default_rng(205400)
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    m=np.sqrt(3)*rng.normal(size=(12000,4))@np.linalg.cholesky(shape).T
    q=rng.chisquare(20,len(m))
    value,info=domain_variance(m,q,20,1.,'transport_pooled_fcentral',shape)
    assert value==pytest.approx(3,rel=.15)
    assert info['allows_variance_decrease'] and info['central_f']['converged']
