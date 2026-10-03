import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import beta,t
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_pivotal import (calibration_scale,ratio_components,predictive_table,
    predictive_tail,pc_components,evaluate)


def test_calibration_radius_cancels_and_scaling_is_safe():
    rng=np.random.default_rng(720016);cal=rng.normal(size=(32,6))
    scale,_=calibration_scale(cal)
    modified=cal*np.exp(rng.normal(size=(32,1))*5)
    assert calibration_scale(modified)[0]==pytest.approx(scale,rel=1e-13)
    assert calibration_scale(cal*1e200)[0]==pytest.approx(scale,rel=1e-13)


@pytest.mark.parametrize('n',[8,32,128])
def test_predictive_enclosure_against_independent_quadrature(n):
    with threadpool_limits(1):table=predictive_table(n,5)
    k=(n+1)//2
    for index in [30,200,450,700]:
        x=table['z'][index]
        value,error=quad(lambda u:t.sf(x*t.ppf((1+u)/2,5)/t.ppf(.75,5),5)*beta.pdf(u,k,n+1-k),0,1,epsabs=2e-12,epsrel=2e-10)
        assert table['lower'][index]-2*error<=value<=table['upper'][index]+2*error
        assert predictive_tail(x,n,5)>=value-2*error
    assert np.all(np.diff(table['upper'])<=1e-14)


def test_conservative_grid_no_extrapolation():
    with threadpool_limits(1):table=predictive_table(32,5)
    x=(table['z'][300]+table['z'][301])/2
    assert predictive_tail(x,32,5)==table['upper'][300]
    assert predictive_tail(1e100,32,5)==table['upper'][-1]
    assert np.all(predictive_tail(np.array([-100.,0.]),32,5)==1)


def test_nonpositive_common_location_is_conservative():
    rng=np.random.default_rng(720017);z=rng.normal(size=(40,4,6))
    profiles=np.ones((2,4));shape=np.eye(4)
    with threadpool_limits(1):
        p0=pc_components(z,profiles,shape,1.,32)['projection']
        p1=pc_components(z-2.,profiles,shape,1.,32)['projection']
    assert np.all(p1[:,0]>=p0[:,0]-1e-12)


def test_singleton_does_not_remove_null_triple():
    rng=np.random.default_rng(720018);z=rng.normal(size=(20,4,6));modified=z.copy();modified[:,0,:]+=1e5
    with threadpool_limits(1):
        p=pc_components(modified,np.ones((2,4)),np.eye(4),1.,32)
    # For a positive singleton the remaining3study Bonf intersection must remain.
    m,y=ratio_components(modified);den=np.sqrt(np.sum(y*y,axis=-1)/5)
    null_triplet=np.minimum(1,3*predictive_tail(m[:,1:]/den[:,1:],32,5).min(1))
    assert np.all(p['ordinary_bonf'][:,0]>=null_triplet-1e-14)


def test_validation_calibration_cannot_change_training_choices():
    rng=np.random.default_rng(720019);z=rng.normal(size=(24,4,6));cal=rng.normal(size=(4,8,6))
    shifted=cal.copy().reshape(-1,6);shifted[1::2]+=3.;shifted=shifted.reshape(cal.shape)
    with threadpool_limits(1):
        _,a=evaluate(z,cal);_,b=evaluate(z,shifted)
    for f1,f2 in zip(a['folds'],b['folds']):
        for key in ['profiles','gamma','pilot','rho_for_learning_only','shape_for_direction']:
            assert f1[key]==f2[key]
    assert a['calibration_validation']['scale']!=b['calibration_validation']['scale']


@pytest.mark.parametrize('cal',[np.zeros((8,6)),np.full((8,6),np.nan),np.ones((3,6))])
def test_invalid_calibration_fails_explicitly(cal):
    with pytest.raises((ValueError,FloatingPointError)):calibration_scale(cal)
