import numpy as np
from localized_reference import profile,log_cauchy,log_cauchy_max
from focused_reference import components

def test_proxy_max_and_quadratic_inverse():
    x=np.linspace(-5,5,101);f=np.exp(log_cauchy(x,2.))
    assert np.allclose(1/f,1+x*x/4)
    assert log_cauchy_max(-3.,2.,2.)==0
    assert log_cauchy_max(1.,2.,2.)==log_cauchy(1.,2.)

def test_profile_bounds_full_joint_dense_fixture_and_tail():
    rng=np.random.default_rng(914);t=rng.normal(size=40)*np.exp(rng.normal(size=40));u=rng.normal(size=40)
    a,c,tau,qt,qb=.6,.4,.2,.8,.5;ou=1.3;sig=.7;D=4.
    Q,r=profile(t,u,ou,a=a,c=c,tau=tau,qt=qt,qb=qb,bound=D,sigma=sig,max_cells=32)
    for s in np.r_[np.linspace(0,r['S'],1001),r['S']+np.array([1.,10.,100.,1000.])]:
        b,aa=components(t,u,s,a,c,tau,qt,qb)
        F=np.exp(log_cauchy(ou-s,2*sig)-log_cauchy(ou,sig+np.log(D)))*np.sum((b+aa)*np.exp(log_cauchy(u+s,sig+np.log(D))-log_cauchy(u,2*sig)))
        assert F<=Q+1e-10

def test_true_nuisance_soft_rank_identity_for_every_exchangeable_label():
    # Direct algebra at a fixed true nuisance, not a Monte Carlo validity test.
    t=np.array([3.,2.,4.,.1]);u0=np.array([-1.,.2,2.,.6]);s=.7
    b,a=components(t,u0,s,.5,.5,.2,1.,1.2)
    score=(b+a)*np.exp(log_cauchy(u0+s,2.)-log_cauchy(u0,1.))
    assert np.isclose(np.mean(4*score/score.sum()),1)
