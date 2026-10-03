"""Fair scale wiring, failure receipts and gene holdout isolation; not FDR evidence."""
from copy import deepcopy
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates
import sca3_compass.robustness_continuous_scale as continuous


def fixture():
    rng=np.random.default_rng(67217)
    z=rng.normal(size=(32,4,6));z[:4,:2]+=3.5;z[4:8]-=3.5
    x=rng.normal(size=(4,32,6))
    _,diag=evaluate_candidates(z,x,'frontier')
    return z,x,diag


@pytest.mark.parametrize('mode,floor',[('continuous_pattern',0.),('continuous_floor_pattern',.5)])
@threadpool_limits.wrap(limits=1)
def test_continuous_actual_fit_gene_isolation_and_simple_baselines(mode,floor):
    z,x,diag=fixture();other=deepcopy(diag)
    result=energy_prior_candidates(z,x,diag,scale_mode=mode)
    changed=z.copy();changed[::2]+=30
    energy_prior_candidates(changed,x,other,scale_mode=mode)
    assert diag['energy_prior']['folds'][0]['domain_scale']=={
        **other['energy_prior']['folds'][0]['domain_scale'],
        'continuous_fit':diag['energy_prior']['folds'][0]['domain_scale']['continuous_fit']}
    # Timing is not a scientific fitting decision; compare all remaining fields.
    def without_timing(value):
        if isinstance(value,dict):
            return {k:without_timing(v) for k,v in value.items() if not k.endswith('_seconds')}
        if isinstance(value,list):return [without_timing(v) for v in value]
        return value
    assert without_timing(diag['energy_prior']['folds'][0]['domain_scale'])==without_timing(
        other['energy_prior']['folds'][0]['domain_scale'])
    for info in diag['energy_prior']['folds']:
        scale=info['domain_scale']
        assert scale['minimum_null_weight']==floor
        assert not scale['uses_held_gene'] and not scale['uses_truth_labels']
        assert info['projection_variance']==scale['selected_variance']
        assert scale['converged']
        assert scale['continuous_fit']['converged']
    extra=pattern_test_candidates(z,diag,result['target_only_weighted_cone_PC'],pilot=True)
    for name in ['pilot80_pattern_projection_eBH','pilot80_pattern_support_simes_eBH',
                 'pilot80_ordinary_bonf_eBH','pilot80_ordinary_simes_eBH']:
        assert extra[name].shape==(32,2) and np.isfinite(extra[name]).all()
    assert np.isfinite(np.array(list(result.values()))).all()
    json.dumps(diag,allow_nan=False)


@threadpool_limits.wrap(limits=1)
def test_failed_continuous_fit_retained_and_identical_fallback(monkeypatch):
    z,x,diag=fixture()
    monkeypatch.setattr(continuous,'fit_continuous_scale',lambda *a,**kw:{
        'converged':False,'variance_multiplier':None,
        'diagnostics':{'converged':False,'numerical_failure_count':1},'status':'injected_failure'})
    expected_diag=deepcopy(diag)
    result=energy_prior_candidates(z,x,diag,scale_mode='continuous_floor_pattern')
    expected=energy_prior_candidates(z,x,expected_diag,scale_mode='transport_fcentral')
    for key in result:np.testing.assert_array_equal(result[key],expected[key])
    for fold in diag['energy_prior']['folds']:
        scale=fold['domain_scale']
        assert scale['fallback_to_transport_fcentral'] and not scale['converged']
        assert scale['continuous_fit']['diagnostics']['numerical_failure_count']==1


@threadpool_limits.wrap(limits=1)
def test_original_path_never_calls_continuous_fit(monkeypatch):
    z,x,diag=fixture()
    expected=energy_prior_candidates(z,x,deepcopy(diag))
    def forbidden(*args,**kwargs):raise AssertionError('Old path invoked a new fit')
    monkeypatch.setattr(continuous,'fit_continuous_scale',forbidden)
    result=energy_prior_candidates(z,x,diag)
    for key in result:np.testing.assert_array_equal(result[key],expected[key])
