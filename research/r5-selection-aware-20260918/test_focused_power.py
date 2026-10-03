import numpy as np
from focused_power_score import power_score,cover,components

def test_score_increases_and_focus_excludes_only_below_threshold():
    x=np.linspace(-2,6,101);y=power_score(x,.1,2.)
    assert np.all(np.diff(y)>=0)
    assert np.all(y[x<=2*np.exp(.1)]==0)
    assert np.all(y<=power_score(x,.1,2.,False))

def test_power_cover_not_grid_only():
    rng=np.random.default_rng(731);t=rng.normal(size=80)*np.exp(rng.normal(size=80));u=rng.normal(size=80)
    kw=dict(a=.8,c=.6,tau=.3,qt=1.2,qb=.8)
    for focused in [True,False]:
        Q,receipt=cover(t,u,**kw,max_cells=16,focused=focused)
        for s in np.r_[np.linspace(0,receipt['S'],801),receipt['S']+np.array([1,10,100])]:
            b,a=components(t,u,s,**kw,focused=focused)
            assert np.sum(b+a)<=Q+1e-9
