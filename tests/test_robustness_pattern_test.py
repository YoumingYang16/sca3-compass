from copy import deepcopy
import json
import numpy as np
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates


def test_pattern_integration_finite_and_held_fold_isolation():
    rng=np.random.default_rng(12919)
    z=rng.normal(size=(64,4,6))
    z[:12,:2]+=3.5
    z[12:24]-=3.5
    calibration=rng.normal(size=(4,64,6))
    _,diag=evaluate_candidates(z,calibration,'frontier')
    p=energy_prior_candidates(z,calibration,diag)
    baseline=p['target_only_weighted_cone_PC']
    original=deepcopy(diag)
    result=pattern_test_candidates(z,diag,baseline,pilot=True,block=True)
    for name,value in result.items():
        assert value.shape==(64,2) and np.isfinite(value).all() and np.min(value)>=0
        if name.endswith('_PC'):
            assert np.max(value)<=1
    json.dumps(diag,allow_nan=False)
    # Freeze input nuisance estimates to isolate the downstream learner;
    # changing held data cannot change that fold's learned directions.
    changed=z.copy();changed[::2]+=100
    pattern_test_candidates(changed,original,baseline,pilot=True,block=True)
    assert original['pattern_testing']['folds'][0]['fit']==diag['pattern_testing']['folds'][0]['fit']
    assert original['pattern_testing']['folds'][0]['pilot_calibration']==diag['pattern_testing']['folds'][0]['pilot_calibration']
