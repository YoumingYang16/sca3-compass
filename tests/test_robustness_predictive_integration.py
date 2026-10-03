from copy import deepcopy
import numpy as np
import pytest
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates


def test_predictive_selector_and_simes_receive_identical_training_simulations():
    rng=np.random.default_rng(721038)
    z=rng.normal(size=(32,4,6));z[:6,:2]+=3.5
    calibration=rng.normal(size=(4,32,6))
    _,diag=evaluate_candidates(z,calibration,'frontier')
    prior=energy_prior_candidates(z,calibration,diag)
    original=deepcopy(diag)
    output=pattern_test_candidates(z,diag,prior['target_only_weighted_cone_PC'],pilot=True,predictive=True)
    for fold in diag['pattern_testing']['folds']:
        info=fold['predictive_mixing'];experts=info['experts']
        assert experts['pattern_projection']['simulation_sha256']==experts['pattern_support_simes']['simulation_sha256']
        held=np.arange(len(z))%2==fold['fold']
        np.testing.assert_array_equal(output['predictive80_selector_eBH'][held],
            output[f'predictive80_{info["selected_expert"]}_eBH'][held])
    changed=z.copy();changed[::2]+=7
    other=deepcopy(original)
    pattern_test_candidates(changed,other,prior['target_only_weighted_cone_PC'],pilot=True,predictive=True)
    # Fold0's training genes and all their frozen nuisances are unchanged.
    assert diag['pattern_testing']['folds'][0]['predictive_mixing']==other['pattern_testing']['folds'][0]['predictive_mixing']
    for name,value in output.items():
        assert np.isfinite(value).all() and np.all(value>=0)
    with pytest.raises(ValueError,match='matching pilot'):
        pattern_test_candidates(z,deepcopy(original),prior['target_only_weighted_cone_PC'],predictive=True)
