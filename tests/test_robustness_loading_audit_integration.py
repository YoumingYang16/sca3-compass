from copy import deepcopy
from unittest.mock import patch
import numpy as np
import pytest
from sca3_compass.robustness_geometry import fit_fold_geometries


@pytest.mark.parametrize('mean,se,intersection_adopt,veto_adopt',[
    (.309,.122,False,True),(-.7,.08,False,False),(.5,.1,True,True),(-.01,.1,False,True)])
def test_intersection_and_veto_are_distinct_prespecified_rules(mean,se,intersection_adopt,veto_adopt):
    rng=np.random.default_rng(21689)
    z=rng.normal(size=(32,4,6))
    info={'fit':{'rho':.3}}
    raw={'all_scores_finite':True,'all_inner_fits_converged':True,
         'mean_log_residual_gain':mean,'gene_level_standard_error':se,'old_gate':{'adopt':True}}
    def gate(*args,**kwargs):
        return intersection_adopt,deepcopy(raw)
    with patch('sca3_compass.robustness_loading_audit_gate.raw_loading_gate',side_effect=gate):
        for mode,expected in [('intersection',intersection_adopt),('veto',veto_adopt)]:
            geometries,receipt=fit_fold_geometries(z,info,audit_mode=mode)
            assert all((g is not None)==expected for g in geometries)
            assert receipt['fallback_gene_fraction']==(0 if expected else 1)
            assert receipt['audit_mode']==mode


def test_audit_failure_does_not_pass_veto_by_missing_negative_score():
    rng=np.random.default_rng(21690)
    z=rng.normal(size=(32,4,6))
    raw={'all_scores_finite':False,'all_inner_fits_converged':False,
         'mean_log_residual_gain':None,'gene_level_standard_error':None,'old_gate':{'adopt':True}}
    with patch('sca3_compass.robustness_loading_audit_gate.raw_loading_gate',return_value=(False,raw)):
        geometries,receipt=fit_fold_geometries(z,{'fit':{'rho':.3}},audit_mode='veto')
    assert geometries==[None,None] and receipt['fallback_gene_fraction']==1


def test_runner_applies_audit_and_scale_to_equal_baselines():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from robustness_screen import one
    result=one({'name':'audit_integration','distribution':'t5','rho':.1,'calibration_rho':.95,
                'n':32,'effect':3.5,'genes':32},0,1,21691,profile='frontier',patterns=True,
               pilot_patterns=True,loading_audit_patterns=True)
    names={r['method'] for r in result['rows']}
    for mode in ['intersection','veto']:
        for scale in ['fixed','joint','pooled']:
            for method in ['pilot80_pattern_projection_gatedmix_eBH','pilot80_pattern_support_simes_eBH',
                           'pilot80_ordinary_simes_eBH','pilot80_ordinary_bonf_eBH']:
                assert f'geometry_{mode}_{scale}_{method}' in names
