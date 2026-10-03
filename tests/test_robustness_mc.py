import numpy as np
from sca3_compass.robustness_mc import projection_weights, conditional_scan


def test_scan_weights_have_unit_null_variance_and_nonnegative_entries():
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    w=projection_weights(shape,(0,1,3))
    assert w.shape==(4,7) and np.all(w>=0) and np.all(w[2]==0)
    np.testing.assert_allclose(np.diag(w.T@shape@w),1,atol=1e-14)


def test_scan_monotone_and_probability_resolution():
    rng=np.random.default_rng(128)
    shape=np.eye(4)
    z=rng.normal(size=(12,2,4))
    a=conditional_scan(z,[shape,shape],25,np.random.default_rng(1),4095)
    b=conditional_scan(z+1,[shape,shape],25,np.random.default_rng(1),4095)
    for name in a:
        assert a[name].shape==(12,2)
        assert np.all(a[name]>=b[name])
        assert np.all((a[name]>=1/4096)&(a[name]<=1))
