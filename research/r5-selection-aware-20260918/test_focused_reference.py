import numpy as np
import pytest
from focused_reference import cover_sum,components,mc_e

def test_exchangeable_sum_normalization_is_exact_for_permuted_scores():
    for x in [np.array([0.,.2,.9,1.]),np.zeros(4),np.array([1.,0.,0.,0.])]:
        es=[mc_e(v,float(x.sum()-v),len(x)-1) for v in x]
        assert np.isclose(np.mean(es),1 if x.sum()>0 else 0)

def test_monotone_cover_bounds_dense_drift_not_only_grid():
    r=np.random.default_rng(889);t=r.normal(size=90)*np.exp(r.normal(size=90));u=r.normal(size=90)
    args=dict(a=.65,c=.4,tau=.17,qt=1.1,qb=.8)
    Q,receipt=cover_sum(t,u,**args,max_cells=32)
    for s in np.r_[np.linspace(0,receipt['S'],501),receipt['S']+np.arange(1,20)]:
        b,a=components(t,u,s,**args)
        assert Q>=np.sum(b+a)-1e-10
    assert Q>=np.sum(t>args['qt'])

def test_coarsening_and_reference_dependent_tail_remain_safe():
    t=np.array([10.,.1,3.,-2]);u=np.array([-10.,.1,4.,-4])
    args=dict(a=.01,c=.1,tau=.001,qt=2.,qb=1.)
    Q,receipt=cover_sum(t,u,**args,max_cells=1)
    for s in [0.,1.,10.,1000.,1e8]:
        b,a=components(t,u,s,**args);assert Q>=sum(b+a)
    assert receipt['cells']==1

def test_mc_normalization_monotone_and_fail_closed_inputs():
    assert np.all(np.diff(mc_e(np.linspace(0,1,100),2.,99))>=0)
    assert mc_e(0.,0.,99)==0
    with pytest.raises(ArithmeticError):mc_e(1.,-1.,99)

def test_internal_valueerror_is_conservative_but_external_input_rejected(monkeypatch):
    import focused_kernel
    r=np.random.default_rng(191);z=r.normal(size=(16,4,6));c=r.normal(size=(4,4,6))
    def bad(*args,**kwargs):raise ValueError('internal numerical fixture')
    monkeypatch.setattr(focused_kernel,'make_reference',bad)
    v=focused_kernel.evaluate(z,c,c,bound=1.,seed=91,outer_draws=63,meta_draws=63)
    assert v['status']=='conservative_failure'
    assert all(not d.any() for d in v['decisions'].values())
    with pytest.raises(ValueError):focused_kernel.evaluate(z,c,c,bound=.1,seed=91)

def test_indicator_vector_slack_matches_scalar_calls():
    from focused_kernel import indicators
    t=np.array([-1.,0.,1.,5.]);u=np.array([3.,2.,1.,-1.])
    actual=indicators(t,u,.5,1.)
    expected=np.array([indicators(np.array([x]),s,.5,1.)[0] for x,s in zip(t,u)])
    assert np.array_equal(actual,expected)
