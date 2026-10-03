from copy import deepcopy
import json
import numpy as np
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates


def test_pilot_selection_exact_old_multiplier_and_held_isolation():
    rng=np.random.default_rng(61103)
    z=rng.normal(size=(64,4,6)); z[:12,:2]+=3.5; z[12:24]-=3.5
    x=rng.normal(size=(4,32,6))
    _,diag=evaluate_candidates(z,x,'frontier')
    base=energy_prior_candidates(z,x,diag)['target_only_weighted_cone_PC']
    other=deepcopy(diag)
    result=pattern_test_candidates(z,diag,base,pilot=True,pilot_selection=True)
    for name,values in result.items():
        if name.startswith('pilotc0.5') and not name.endswith('support_gate_eBH'):
            np.testing.assert_array_equal(values,result[name.replace('pilotc0.5','pilot80')])
    for fold,info in enumerate(diag['pattern_testing']['folds']):
        held=np.arange(len(z))%2==fold
        gamma=np.array(info['full_support_gate_fraction_by_sign'])
        old=np.array(info['mixing_fraction_by_sign'])
        np.testing.assert_array_equal(old,np.minimum(gamma,.9))
        for mode in ['pattern_projection','pattern_support_simes']:
            expected=(1-gamma)*result['pilotc0.65_cone_eBH'][held]+gamma*result[f'pilotc0.65_{mode}_eBH'][held]
            np.testing.assert_array_equal(result[f'pilotc0.65_{mode}_support_gate_eBH'][held],expected)
    z[::2]+=50
    pattern_test_candidates(z,other,base,pilot=True,pilot_selection=True)
    a,b=diag['pattern_testing']['folds'][0],other['pattern_testing']['folds'][0]
    assert a['pilot_multiplier_selection']==b['pilot_multiplier_selection']
    keys=set(a['pilot_multiplier_selection'])
    for prefix in ['pilotc0.25','pilotc0.5','pilotc0.65','pilotc0.8','pilotcv']:
        assert all(f'{prefix}_{mode}_eBH' in keys for mode in
            ['pattern_projection','pattern_support_simes','cone','ordinary_bonf','ordinary_simes'])
    json.dumps(diag,allow_nan=False)
