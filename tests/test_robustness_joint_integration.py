from copy import deepcopy
import json
import numpy as np
import pytest
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_prior import energy_prior_candidates


@pytest.mark.parametrize('mode',['joint_pattern','central_capped_joint_pattern','free_central_capped_joint_pattern'])
def test_joint_scale_integration_held_isolation_and_finite_outputs(mode):
    rng=np.random.default_rng(194071)
    z=rng.normal(size=(64,4,6));z[:12,:2]+=3.5;z[12:24]-=3.5
    calibration=rng.normal(size=(4,32,6))
    _,diag=evaluate_candidates(z,calibration,'frontier')
    frozen=deepcopy(diag)
    result=energy_prior_candidates(z,calibration,diag,scale_mode=mode)
    changed=z.copy();changed[::2]+=30
    energy_prior_candidates(changed,calibration,frozen,scale_mode=mode)
    assert diag['energy_prior']['folds'][0]['domain_scale']==frozen['energy_prior']['folds'][0]['domain_scale']
    assert diag['energy_prior']['folds'][0]['target_only']==frozen['energy_prior']['folds'][0]['target_only']
    for value in result.values():
        assert np.isfinite(value).all() and np.min(value)>=0 and np.max(value)<=1
    if 'capped' in mode:
        for info in diag['energy_prior']['folds']:
            scale=info['domain_scale'];cap=scale['central_cap']
            assert scale['selected_variance']==min(cap['joint_variance'],cap['central_cap'])
            assert cap['cap_applied']==(cap['joint_variance']>cap['central_cap'])
            expected='transport_fcentral' if mode.startswith('free_') else 'anchored_fcentral'
            assert cap['central_fit']['mode']==expected
            assert cap['central_fit']['uses_held_gene'] is False
    json.dumps(diag,allow_nan=False)
