import numpy as np
import pytest
from action_reference import probabilities,selected_errors
from action_borrowing import evaluate
from test_selective_reference import NormalLaw


def test_kernel_logconcavity_and_partition():
    u=np.linspace(-8,8,1001);p=probabilities(u,1.,.55)
    np.testing.assert_allclose(p.sum(1),1,atol=1e-14)
    assert np.max(np.diff(np.log(p),n=2,axis=0))<1e-12
    assert p[0,2]>.99 and p[-1,0]>.99


def test_selected_pairs_and_cap():
    x,y,r=selected_errors(NormalLaw(1),NormalLaw(1),0,0,1,.55,2,
        [np.random.default_rng(s) for s in [1,2,3]],4096)
    assert np.mean(x-y)<0 and r['action']==2 and r['retained']==4096
    with pytest.raises(RuntimeError) as failure:selected_errors(NormalLaw(1),NormalLaw(1),0,0,1,.55,1,
        [np.random.default_rng(s) for s in [1,2,3]],100,cap=1)
    assert failure.value.receipt['requested']==100
    assert failure.value.receipt['retained']<=1
    assert failure.value.receipt['returned']==0


def test_integrated_action_e_and_no_test_selection():
    rng=np.random.default_rng(764);z=rng.normal(size=(32,4,6));cs=rng.normal(size=(8,4,6));ct=rng.normal(size=(4,4,6))
    kw={'bound':3.,'seed':8,'reference_draws':127,'inner_draws':255}
    out=evaluate(z,cs,ct,**kw);assert out['status']=='completed'
    p=np.asarray(out['calibration']['action_probabilities'])
    np.testing.assert_array_equal(out['e'],np.einsum('j,jgd->gd',p,out['component_e']))
    changed=z.copy();changed[::4]+=4;again=evaluate(changed,cs,ct,**kw)
    np.testing.assert_array_equal(again['calibration']['action_probabilities'],p)
    assert out['p'].shape==(3,32,2,2) and 'NOT_COMBINED' in out['p_kind']
