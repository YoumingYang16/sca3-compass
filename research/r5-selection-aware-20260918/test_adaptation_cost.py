import numpy as np
from scipy.special import expit,roots_hermitenorm
from ancillary_calibration import ConditionalLogF
from check_adaptation_cost import prob

def test_logistic_translation_lr_certificate_and_uniform_sharpness():
    r=2.;n=4;d=4*np.arctanh(r/(3*n));x=np.linspace(-25,25,301)
    for w in [np.zeros(n),np.array([-3.,-1.,1.,3.])]:
        f=ConditionalLogF(w)
        assert np.max(-r+f.derivative(x)-f.derivative(x+d))<1e-12
    f=ConditionalLogF(np.zeros(n));d_bad=d*1.01;y=-np.log(2)-d_bad/2
    assert -r+f.derivative(y)-f.derivative(y+d_bad)>0

def test_gaussian_constant_curvature_identity():
    x,w=roots_hermitenorm(128);w=w/np.sqrt(2*np.pi);vs=.3;vt=.7;r=.5
    sx=np.sqrt(vs)*x;tx=.2+np.sqrt(vt)*x;d=r*vt
    for s in [0.,.5,1.]:
        tilted=prob(s,sx,w,tx-d,w,.6,.5)
        shifted=prob(s+d,sx,w,tx,w,.6,.5)
        assert abs(tilted-shifted)<2e-15

def test_actual_shape_pivot_cauchy_coupling_not_t_equality():
    rng=np.random.default_rng(1702)
    for _ in range(20):
        a=rng.normal(size=(4,4));h=a@a.T+1e-5*np.eye(4)
        y=rng.normal(size=(4,5));z=abs(rng.normal())
        denom=h[0,0]*np.trace(y.T@np.linalg.inv(h)@y)/20
        witness=np.sum(y[0]**2)/20
        assert denom>=witness-1e-10
        assert z/np.sqrt(denom)<=z/np.sqrt(witness)+1e-10
