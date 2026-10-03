import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_convergence_audit import factorial,stratified,bounded
from robustness_convergence_auxiliary import covariance_equivalence,compound_scale_reference


def test_factorial_identity_and_pairing():
    a=np.array([.1,.2,.3]);b=a+.2;c=a+.1;d=a+.4
    f=factorial(a,b,c,d)
    assert f['package_simple_B-A']['mean']==pytest.approx(.2)
    assert f['projection_candidate_D-B']['mean']==pytest.approx(.2)
    assert f['interaction_D-C-B+A']['mean']==pytest.approx(.1)
    assert f['interaction_D-C-B+A']['mean']==pytest.approx(
        f['package_complex_D-C']['mean']-f['package_simple_B-A']['mean'])


def test_stratified_mean_se_and_heterogeneity_distinct():
    x=np.array([[.1,.2,.3,.4],[.5,.6,.7,.8]])
    r=stratified(x)
    assert r['mean']==pytest.approx(x.mean())
    assert r['mc_se']==pytest.approx(np.sqrt(np.sum(x.var(axis=1,ddof=1)/4))/2)
    assert r['scenario_mean_sd']>r['mc_se']
    assert r['supplemental_bernstein']['n']==8


def test_zero_observed_variance_does_not_prove_zero_uncertainty():
    r=stratified(np.zeros((2,20)))
    assert r['mc_se']==0
    assert r['supplemental_bernstein']['lower']<0<r['supplemental_bernstein']['upper']


@pytest.mark.parametrize('x',[[],[np.nan],[np.inf],[-.1],[1.1],[[.1,.2]]])
def test_bad_repetitions_fail_closed(x):
    with pytest.raises(ValueError):bounded(x)


def test_no_truncation():
    with pytest.raises(ValueError):factorial([.1,.2],[.2],[.3,.4],[.2,.4])


def test_full_observation_equivalence_not_only_means():
    result=covariance_equivalence()
    assert result['full_observed_covariance_max_error']<1e-14
    assert min(result['positive_definite_noise_min_eigenvalues'])>0
    assert result['effect_covariance_after_pipeline_centering_max']<1e-14


def test_compound_scale_reference_and_boundaries():
    assert compound_scale_reference(.8,.8)==pytest.approx(1)
    assert compound_scale_reference(.5,.3)==pytest.approx(1.96)
    assert compound_scale_reference(.95,.1)==pytest.approx(69)
    assert compound_scale_reference(.8,.1)==pytest.approx(15)
    with pytest.raises(ValueError):compound_scale_reference(1,.1)
