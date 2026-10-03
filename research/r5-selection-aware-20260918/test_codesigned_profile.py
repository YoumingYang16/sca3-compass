import numpy as np
from codesigned_profile import CodesignedProfile
from selection_profile import select

def test_co_design_equals_max_and_monotone():
    r=np.random.default_rng(111);n=127
    p=CodesignedProfile(r.normal(size=n)/3,r.normal(size=n),r.normal(size=n),32,4,
                        r.normal(size=4095)*1.6,r.normal(size=4095))
    previous=p.draw_scores(0)
    for s in [0,.1,1,5,100]:
        scores=p.draw_scores(s);assert np.all(scores<=previous);previous=scores
        for i in np.flatnonzero(p.positive):
            ch=select(p.es[i],p.et[i],32,4,np.exp(s),p.c)
            chosen=p.logbase[i]-ch['log_scale']/2-(p.lqb if ch['borrow'] else p.lqt)
            assert np.isclose(chosen,scores[i])

def test_boundary_rank_matches_direct_selected_reference():
    r=np.random.default_rng(3);p=CodesignedProfile(r.normal(size=31),r.normal(size=31),r.normal(size=31),4,4,
                                                 r.normal(size=127),r.normal(size=127))
    for w in [.01,.2,2,10,1e6]:
        for branch,q in [(True,p.qb),(False,p.qt)]:
            expected=(1+np.sum(p.draw_scores(0)>=np.log(w/q)))/32
            assert p.pvalues(np.array([w]),branch)[0]==expected

def test_all_negative_inner_reference_is_positive_independent_normalizer():
    p=CodesignedProfile(np.zeros(7),np.zeros(7),np.arange(-3.,4.),4,4,-np.ones(31),-2*np.ones(31))
    assert p.qt==p.qb==1.
    assert p.normalizer_fallback=={'target':True,'bridge':True}
    assert p.pvalues(np.array([2.]),False)[0]==3/8
