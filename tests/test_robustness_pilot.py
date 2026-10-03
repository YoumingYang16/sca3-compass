import numpy as np
import pytest
from sca3_compass.robustness_pilot import pilot_calibrator
from sca3_compass.robustness_calibrators import by_calibrator


@pytest.mark.parametrize('fraction',[.5,.8])
@pytest.mark.parametrize('count',[0,1,5,25,200])
def test_training_selected_calibrator_integrates_one(fraction,count):
    training=np.ones(256);training[:count]=1e-8
    _,info=pilot_calibrator(np.ones(10),training,512,fraction)
    cap=64;h=np.sum(1/np.arange(1,cap+1))
    cut=np.unique(np.r_[0,.05*np.arange(1,cap+1)/(512*h),info['threshold'],1])
    e,_=pilot_calibrator((cut[:-1]+cut[1:])/2,training,512,fraction)
    assert abs(np.sum(np.diff(cut)*e)-1)<1e-13
    assert np.all(np.diff(e)<=0)
    assert info['pilot_discoveries']==count


def test_no_pilot_fallback_and_held_independence():
    p=np.linspace(0,1,513)
    output,info=pilot_calibrator(p,np.ones(256),512)
    np.testing.assert_array_equal(output,by_calibrator(p,512,cap=64))
    training=np.r_[np.full(20,.0001),np.ones(236)]
    _,a=pilot_calibrator(p,training,512)
    _,b=pilot_calibrator(np.array([.01,.1]),training[::-1],512)
    assert a==b
    assert info['zero_count_fallback']


@pytest.mark.parametrize('bad',[[],[np.nan],[-.01],[1.01]])
def test_invalid_pilot(bad):
    with pytest.raises(ValueError):
        pilot_calibrator(np.ones(2),bad,512)
