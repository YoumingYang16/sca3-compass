import inspect
import numpy as np
import pytest
from r5_kernel import evaluate,components
from r5_common import model,fc
from selection_profile import inner_rank

def data():
    r=np.random.default_rng(9811)
    return r.normal(size=(32,4,6)),r.normal(size=(8,4,6)),r.normal(size=(4,4,6))

def test_full_reproducibility_grid_and_folds():
    z,cs,ct=data();a=evaluate(z,cs,ct,bound=3.,seed=4,reference_draws=127,inner_draws=255)
    b=evaluate(z,cs,ct,bound=3.,seed=4,reference_draws=127,inner_draws=255)
    assert a['status']==b['status']=='completed'
    for key in ['p','e','decision']:assert np.array_equal(a[key],b[key])
    assert np.all((a['p']>=1/128)&(a['p']<=1))
    assert np.all(a['p']*128==np.round(a['p']*128))
    for f in a['folds']:
        assert len({f[x] for x in ['test_fold','pilot_fold','direction_fold','shape_fold']})==4
    for value in a['profile_counts'].values():assert value['maximum']>=value['at_infinity']

def test_no_oracle_api():
    assert set(inspect.signature(evaluate).parameters)=={'z','cs','ct','bound','seed','reference_draws','inner_draws','audit','method'}

def test_zero_calibration_fails_closed():
    z,cs,ct=data();a=evaluate(z,0*cs,ct,bound=1.,seed=4,reference_draws=31,inner_draws=31)
    assert a['status']=='conservative_numerical_failure'
    assert not a['decision'].any() and np.all(a['e']==0) and np.all(a['p']==1)

def test_invalid_bound_not_silently_repaired():
    z,cs,ct=data()
    with pytest.raises(ValueError):evaluate(z,cs,ct,bound=.9,seed=4)

def test_components_identical_to_original_for_identity_reference():
    class Adapter:
        def pvalues(self,x,borrow):return inner_rank(x,ref)/(len(ref)+1)
    z,cs,ct=data();r=np.random.default_rng(11);ref=np.sort(r.normal(size=127))
    h=fc.shape_fit(z@fc.contrasts(6),2);profiles=np.ones((2,4))
    p,_=model.components(z,h,1.,profiles,ref,h)
    q,_=components(z,h,1.,profiles,Adapter(),h,True)
    assert np.array_equal(p,q)

def test_inner_outer_streams_distinct_and_outer_complete():
    z,cs,ct=data();a=evaluate(z,cs,ct,bound=1.,seed=1,reference_draws=31,inner_draws=63,audit=True,method='closure')
    assert a['status']=='completed';r=a['reference_tuples']
    assert len(set(r['streams']))==4
    assert len(r['es'])==len(r['et'])==len(r['base'])==31
    assert len(r['inner_target'])==len(r['inner_bridge'])==63
    for b in r['outer_shape_tuples']:assert np.allclose(np.linalg.eigvalsh(b['H']),b['eigen'])
