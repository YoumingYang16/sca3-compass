from copy import deepcopy
import json

import numpy as np
import pytest
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates
from sca3_compass import robustness_directional_budget as budget_module
from sca3_compass import robustness_predictive_gate as predictive_module


def inputs():
    rng=np.random.default_rng(60101)
    z=rng.normal(size=(64,4,6)); z[:12,:2]+=3.5; z[12:24]-=3.5
    calibration=rng.normal(size=(4,32,6))
    _,diag=evaluate_candidates(z,calibration,'frontier')
    baseline=energy_prior_candidates(z,calibration,diag)['target_only_weighted_cone_PC']
    return z,diag,baseline


def test_budget_integration_isolated_and_equal_baseline_opportunity():
    z,diag,p=inputs(); other=deepcopy(diag)
    result=pattern_test_candidates(z,diag,p,pilot=True,directional=True)
    assert not any('predictive80' in name for name in result)
    names=['pattern_projection_raw','pattern_projection_gated','pattern_support_simes_raw',
           'pattern_support_simes_gated','cone']
    for name in names:
        assert result['budget80_'+name+'_eBH'].shape==(64,2)
    for fold in diag['pattern_testing']['folds']:
        policies=fold['directional_budget']
        assert set(policies)==set(names)
        assert len({v['simulation_sha256'] for v in policies.values()})==1
        for policy in policies.values():
            assert sum(policy['allocation']['selected_weights'])==2
            assert policy['allocation']['positive_weight_grid']==list(budget_module.POSITIVE_WEIGHT_GRID)
    changed=z.copy(); changed[::2]+=50
    pattern_test_candidates(changed,other,p,pilot=True,directional=True)
    assert diag['pattern_testing']['folds'][0]['directional_budget']==other['pattern_testing']['folds'][0]['directional_budget']
    json.dumps(diag,allow_nan=False)


def test_budget_does_not_reselect_gamma_or_renormalize_held_evidence(monkeypatch):
    def no_gamma(*args):
        raise AssertionError('Gamma must not be selected by budget-only mode')
    monkeypatch.setattr(predictive_module,'select_predictive_gamma',no_gamma)
    weights=np.array([1.25,.75])
    monkeypatch.setattr(budget_module,'directional_budget',lambda *args,**kwargs:
        (weights.copy(),{'selected_weights':weights.tolist(),'test_stub':True}))
    z,diag,p=inputs()
    result=pattern_test_candidates(z,diag,p,pilot=True,directional=True)
    for mode in ['pattern_projection','pattern_support_simes']:
        np.testing.assert_array_equal(result[f'budget80_{mode}_raw_eBH'],result[f'pilot80_{mode}_eBH']*weights)
        np.testing.assert_array_equal(result[f'budget80_{mode}_gated_eBH'],result[f'pilot80_{mode}_gatedmix_eBH']*weights)
    np.testing.assert_array_equal(result['budget80_cone_eBH'],result['pilot80_cone_eBH']*weights)


def test_budget_requires_pilot():
    z,diag,p=inputs()
    with pytest.raises(ValueError,match='pilot'):
        pattern_test_candidates(z,diag,p,directional=True)
