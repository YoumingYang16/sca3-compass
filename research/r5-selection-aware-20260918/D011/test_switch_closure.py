import numpy as np
from switch_closure import SwitchClosure

def fixture():
    r=np.random.default_rng(338)
    return SwitchClosure(r.normal(size=513),r.normal(size=513),r.normal(size=513),32,4,
        r.standard_t(4,size=1023),r.standard_t(20,size=1023),source_weight=.87,switch_threshold=.8)

def test_complete_trajectory_monotone_including_switches():
    p=fixture();points=np.unique(np.r_[0.,np.linspace(0,8,150),p.c-p.es+p.et]);points=points[points>=0]
    previous=p.draw_scores(0)
    for s in points:
        now=p.draw_scores(float(s))
        assert np.all(now<=previous+2e-14)
        previous=now
    assert np.allclose(p.draw_scores(100),p.mt.forward_log(p.logtarget))

def test_switch_correction_is_exact_maximum_lower_bound():
    p=fixture();x=np.linspace(-20,20,101)
    v=p.bridge_score(x)
    assert np.array_equal(v,np.maximum(p.mb.forward_log(x),p.mt.forward_log(x+p.a*p.c/2)))
    assert np.all(v>=p.mt.forward_log(x+p.a*p.c/2))

def test_boundary_rank_and_nonpositive_inputs():
    p=fixture()
    for b in [False,True]:
        x=np.array([-1.,0.,.001,.5,2.,100.,1e50]);actual=p.pvalues(x,b)
        score=p.bridge_score(np.log(x[2:])) if b else p.mt.forward_log(np.log(x[2:]))
        expected=(1+np.sum(p.draw_scores(0)[:,None]>=score[None,:],axis=0))/(p.m+1)
        assert np.array_equal(actual[:2],[1.,1.]);assert np.array_equal(actual[2:],expected)

def test_independent_map_fallback_has_no_positive_sample_requirement():
    p=SwitchClosure(np.zeros(7),np.zeros(7),np.arange(-3.,4.),4,4,-np.ones(31),-np.ones(31),
                    source_weight=.5,switch_threshold=1.)
    assert p.map_fallback=={'target':True,'bridge':True}
    assert np.isfinite(p.draw_scores(0)).all()
