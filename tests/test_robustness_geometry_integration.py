from copy import deepcopy
import numpy as np
import pytest
from sca3_compass.robustness_methods import evaluate_candidates,contrasts
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates


def test_no_geometry_fallback_is_exact_and_custom_interface_matches():
    rng=np.random.default_rng(515009)
    z=rng.normal(size=(32,4,6));z[:6,:2]+=3.5
    x=rng.normal(size=(4,32,6))
    _,original=evaluate_candidates(z,x,'frontier')
    a,b,c=deepcopy(original),deepcopy(original),deepcopy(original)
    p=energy_prior_candidates(z,x,a)
    same=energy_prior_candidates(z,x,b,geometries=[None,None])
    for name in p:
        np.testing.assert_array_equal(p[name],same[name])
    assert a['energy_prior']==b['energy_prior']
    geometry=[]
    y=(z-z.mean(-1)[...,None])@contrasts(6)
    rho=original['fit']['rho']
    for info in original['shape']:
        shape=np.asarray(info['matrix'])
        geometry.append({'means':z.mean(-1),'q':np.einsum('gsk,st,gtk->g',y,np.linalg.inv(shape),y)/(1-rho),
            'dimension':20,'study_shape':shape,'projection_variance':(1+5*rho)/6})
    custom=energy_prior_candidates(z,x,c,geometries=geometry)
    for name in p:
        np.testing.assert_array_equal(p[name],custom[name])
    out=pattern_test_candidates(z,a,p['target_only_weighted_cone_PC'],pilot=True)
    projected=pattern_test_candidates(z,c,custom['target_only_weighted_cone_PC'],pilot=True,geometries=geometry)
    for name in out:
        np.testing.assert_array_equal(out[name],projected[name])
    with pytest.raises(ValueError,match='Missing custom geometry'):
        pattern_test_candidates(z,c,custom['target_only_weighted_cone_PC'])


def test_loading_runner_connects_all_shared_information_baselines():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('geometry_runner',Path(__file__).resolve().parents[1]/'scripts/robustness_screen.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.one({'name':'integration','distribution':'t5','rho':.8,'n':32,'effect':3.5,
        'pipeline_heterogeneity':.35},0,1,1930481,profile='frontier',patterns=True,pilot_patterns=True,loading_patterns=True,loading=True)
    names={r['method'] for r in result['rows']}
    for suffix in ['pattern_projection_gatedmix_eBH','pilot80_pattern_projection_gatedmix_eBH',
                   'pilot80_ordinary_bonf_eBH','pilot80_ordinary_simes_eBH','pilot80_pattern_support_simes_eBH']:
        assert 'geometry_joint_'+suffix in names
    assert 'capped_loading_limma_robust_f0.0_eBH' in names
    assert result['diagnostics'][0]['geometry_joint_combined']['loading_geometry']['fallback_gene_fraction']<1
