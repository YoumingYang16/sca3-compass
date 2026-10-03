import numpy as np
import pytest
from scipy.special import expit
from selective_reference import selected_errors
from test_selective_reference import NormalLaw
from smooth_borrowing import evaluate

def test_smooth_pair_sampler_is_not_hard_truncation():
    x,y,r=selected_errors(NormalLaw(1),NormalLaw(1),0,0,1,True,
        [np.random.default_rng(1),np.random.default_rng(2),np.random.default_rng(3)],20000,soft_tau=1.)
    assert np.any(x-y>1) and np.any(x-y<=1)
    assert r['soft_tau']==1. and r['retained']==20000

def test_mixture_is_actual_data_weighted_component_e_not_p_average():
    r=np.random.default_rng(892);z=r.normal(size=(32,4,6));cs=r.normal(size=(8,4,6));ct=r.normal(size=(4,4,6))
    out=evaluate(z,cs,ct,bound=3.,seed=7,reference_draws=127,inner_draws=255)
    assert out['status']=='completed'
    c=out['calibration'];p=expit((c['switch_threshold']-c['centered_observed_slack'])/c['soft_tau'])
    assert p==c['borrow_probability']
    assert np.array_equal(out['e'],p*out['component_e'][0]+(1-p)*out['component_e'][1])
    assert out['p'].shape==(2,32,2,2)
    assert 'NOT_A_COMBINED' in out['p_kind']
    repeat=evaluate(z,cs,ct,bound=3.,seed=7,reference_draws=127,inner_draws=255)
    assert np.array_equal(out['e'],repeat['e'])
    changed=z.copy();changed[::4]+=3.
    newer=evaluate(changed,cs,ct,bound=3.,seed=7,reference_draws=127,inner_draws=255)
    assert newer['calibration']==out['calibration']
    for side in ['bridge','target']:
        assert np.array_equal(newer['branch_folds'][side][0]['gamma'],out['branch_folds'][side][0]['gamma'])

def test_common_failure_is_explicit_zero_e():
    r=np.random.default_rng(80);z=r.normal(size=(32,4,6));cs=np.zeros((8,4,6));ct=r.normal(size=(4,4,6))
    out=evaluate(z,cs,ct,bound=3.,seed=7,reference_draws=31,inner_draws=63)
    assert out['status']=='conservative_numerical_failure' and np.all(out['e']==0)
    assert out['branch_failures']['bridge']['seconds']>=0
    assert 'WHOLE_FAMILY' in out['failure_policy']

def test_public_adapter_rejects_standalone_conditional_component():
    with pytest.raises(ValueError,match='internal'):
        evaluate(None,None,None,bound=1,seed=1,method='soft_bridge')
