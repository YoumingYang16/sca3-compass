"""Dimension extension preserves historical draws, truth and information budget."""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_screen import data,one,simulation_equicorrelation


@pytest.mark.parametrize('case,expected',[
    ({'distribution':'t5','rho':.8,'n':64,'effect':3.5},
     ['21d3da547971c85aab86c4f7f920f7b06e2196010b7a93f9bab7a7b3e8db1486',
      '55d551caaa2904d48a29151be6fe10e29325a78d08e484aaf1a0d80d4072f7a7',
      '92e354640320dcaeb0eda9302d00b923dbd7b3e45418d66e2060e1263eee2e0a']),
    ({'distribution':'normal','rho':.95,'n':16,'effect':8,'truth':'single_study_only',
      'null_pipeline_shift':1,'pipeline_heterogeneity':True},
     ['60a9eec1889e1af0ef4190c7c73a8be533b436a6352d04ed795c9b4c8a3c7908',
      '309ea2f47fad1e0c77f5e586adc0007d7634c54e1d33f5d427c4ff9ddd803598',
      '8f9bf5c1f44445ba8acf0868c17b1bd47f0332205821438947d866399b36eb51']),
    ({'distribution':'lognormal','rho':.1,'n':64,'effect':3.5,'effect_layout':'random_gene',
      'effect_jitter':.7,'pipeline_heterogeneity':'study_random'},
     ['8e8b253981fea173b879ec884e2bd6b410f68e56fb668f5ee9140e62c0305cde',
      '91b6dc0ee1ccf8858d14b07a354f9b8f07289b39b5fd6cddabdf165eabfdbf0b',
      '92e354640320dcaeb0eda9302d00b923dbd7b3e45418d66e2060e1263eee2e0a']),
])
def test_historical_default_draws_bitwise_unchanged(case,expected):
    # Digests captured before modifying the runner, seed5423911, NumPy2.5.3.
    assert [hashlib.sha256(x.tobytes()).hexdigest() for x in data(np.random.default_rng(5423911),case)]==expected


@pytest.mark.parametrize('g,k',[(32,3),(65,4),(128,10),(512,12)])
@pytest.mark.parametrize('heterogeneity',[False,True,'gene_random','study_random'])
def test_dimensions_keep_signed_truth(g,k,heterogeneity):
    case={'distribution':'t5','rho':.8,'n':24,'effect':3.5,'pipeline_heterogeneity':heterogeneity}
    z,x,truth=data(np.random.default_rng(41903),case,g=g,k=k)
    assert z.shape==(g,4,k) and x.shape==(4,24,k) and truth.shape==(g,2)
    assert int(truth.sum())==int(g*.2)
    assert np.isfinite(z).all() and np.isfinite(x).all()


@pytest.mark.parametrize('k',[3,4,10,12])
def test_shifted_composite_truth_not_latent_label(k):
    case={'distribution':'normal','rho':.8,'n':16,'effect':3.5,
          'truth':'global_null','null_pipeline_shift':1}
    _,_,truth=data(np.random.default_rng(41904),case,g=32,k=k)
    assert not truth[:,0].any() and truth[:,1].all()


@pytest.mark.parametrize('g,k',[(32,3),(65,4),(96,10)])
def test_complete_candidate_and_fair_baselines_other_dimensions(g,k):
    result=one({'name':'dimension_test','distribution':'t5','rho':.8,'n':16,'effect':3.5,
                'genes':g,'pipelines':k},0,1,41905,profile='frontier',patterns=True,pilot_patterns=True,
               loading_patterns=True,loading=True)
    assert result['dimensions']=={'genes':g,'studies':4,'pipelines':k,'signed_family':2*g}
    assert result['replicated_signed_truths']==int(g*.2)
    names={r['method'] for r in result['rows']}
    assert 'geometry_joint_pilot80_pattern_projection_gatedmix_eBH' in names
    assert 'geometry_joint_pilot80_pattern_support_simes_eBH' in names
    assert 'capped_loading_limma_robust_f0.0_eBH' in names
    for row in result['rows']:
        assert np.isfinite(row['fdp_by_repetition']).all()
        assert 0<=row['fdr']['mean']<=1 and 0<=row['power']['mean']<=1


@pytest.mark.parametrize('g,s,k',[(3,4,6),(32,3,6),(32,4,1),(32.5,4,6),(32,4,True)])
def test_bad_dimensions_rejected(g,s,k):
    with pytest.raises(ValueError,match='Generator requires'):
        data(np.random.default_rng(1),{},g=g,s=s,k=k)


@pytest.mark.parametrize('k,rho',[(6,-.15),(3,-.4),(10,-.1),(4,0),(6,.95)])
def test_full_simulation_compound_range(k,rho):
    cov=simulation_equicorrelation(k,rho)
    assert np.linalg.eigvalsh(cov).min()>0
    np.testing.assert_array_equal(np.diag(cov),np.ones(k))
    assert cov[0,1]==rho


@pytest.mark.parametrize('rho',[-.2,-.3,1,1.1,np.nan,np.inf,True])
def test_invalid_simulation_compound_range(rho):
    with pytest.raises(ValueError,match='strictly positive definite'):
        simulation_equicorrelation(6,rho)


def test_negative_pipeline_stress_reaches_all_methods_without_clipping():
    result=one({'name':'negative_pipeline_regression','distribution':'t5','rho':-.15,
                'n':24,'effect':3.5,'genes':32},0,1,58317,profile='frontier',
               patterns=True,pilot_patterns=True,loading_patterns=True,loading_audit_patterns=True)
    assert result['case']['rho']==-.15
    for row in result['rows']:
        assert np.isfinite(row['fdp_by_repetition']).all()
