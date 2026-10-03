import numpy as np
import pytest
from r3_model import evaluate, geometric_kappa, components, references
from finite_calibration import shape_fit, contrasts
from predictive_bridge import reference_bank

def samples():
    rng=np.random.default_rng(57133)
    return rng.normal(size=(32,4,6)),rng.normal(size=(12,4,6))

def test_geometric_scale_equivariance():
    _,cal=samples();mu=cal.mean(-1,keepdims=True)
    transformed=(cal-mu)+np.sqrt(3.2)*mu
    assert np.isclose(geometric_kappa(transformed)/geometric_kappa(cal),3.2,rtol=1e-12)

def test_geometric_left_and_radial_invariance():
    _,cal=samples();rng=np.random.default_rng(55)
    a=rng.normal(size=(12,4,4))+4*np.eye(4)
    assert np.isclose(geometric_kappa(a@cal),geometric_kappa(cal),rtol=1e-12)

def test_reference_replays_r2_random_schedule():
    refs=references(578,32,12,127)
    expected=reference_bank(np.random.default_rng(578),16,12,127,2)
    assert np.array_equal(refs['R2_replay'],expected)

def test_inference_shape_does_not_select_direction():
    z,_=samples();h=shape_fit(z@contrasts(6),2); ref=np.linspace(-8,8,127)
    _,a=components(z,h,1.,np.ones((2,4)),ref,h)
    _,b=components(z,np.diag([.2,1,2,3]),1.,np.ones((2,4)),ref,h)
    assert all(np.array_equal(x['direction'],y['direction']) for x,y in zip(a,b))

def test_role_isolation():
    z,cal=samples();a=evaluate(z,cal,seed=22,reference_draws=127)
    changed=z.copy();changed[np.arange(len(z))%4==3]*=np.array([1.,2.,4.,8.])[None,:,None]
    b=evaluate(changed,cal,seed=22,reference_draws=127)
    assert a['status']==b['status']=='completed'
    af=a['folds'][0];bf=b['folds'][0]
    assert af['direction_fold']==2 and af['shape_fold']==3 and af['pilot_fold']==1
    assert np.array_equal(af['direction_profiles'],bf['direction_profiles'])
    assert np.array_equal(af['gamma'],bf['gamma'])
    assert all(np.array_equal(x['direction'],y['direction']) for x,y in zip(af['directions'],bf['directions']))
    assert not np.allclose(af['inference_shape'],bf['inference_shape'])

def test_delta_only_changes_inference():
    z,cal=samples();a=evaluate(z,cal,seed=22,reference_draws=127)
    b=evaluate(z,cal,seed=22,reference_draws=127,mismatch_bound=5.)
    assert a['status']==b['status']=='completed'
    assert np.all(b['p']['R3_main']>=a['p']['R3_main'])
    assert np.array_equal(a['references']['R3_main'],b['references']['R3_main'])
    assert all(np.array_equal(x['gamma'],y['gamma']) for x,y in zip(a['folds'],b['folds']))

def test_invalid_inputs_and_conservative_failure():
    z,cal=samples()
    with pytest.raises(ValueError): evaluate(z,cal,seed=True)
    a=evaluate(z,np.zeros_like(cal),seed=3,reference_draws=127)
    assert a['status']=='conservative_numerical_failure'
    assert not a['decisions']['R3_main'].any()
