import numpy as np
import pytest
from grid_calibration import grid_calibrate


def test_exact_support_budget_and_monotonicity():
    for draws in [31,4095]:
        for c in [0,1]:
            step=3 if c==0 else 1
            num=np.r_[np.arange(step,draws+1,step),draws+1]
            grid=num/(draws+1); weights=np.diff(np.r_[0,num])/(draws+1)
            for count in [0,1,12]:
                e,info=grid_calibrate(grid,{'count':count,'threshold':.025*max(1,count)/128},512,draws,c)
                assert np.all(np.diff(e)<=0)
                assert float(weights@e)<=1+1e-14
                if info['normalizer']>0: assert abs(float(weights@e)-1)<1e-14


def test_impossible_focus_is_not_lost_budget():
    p=np.array([1/4096,20/4096,1])
    e,info=grid_calibrate(p,{'count':1,'threshold':.025/128},512,4095,1)
    assert info['normalizer']<.2
    assert e[0]>1000


def test_offgrid_rejected():
    with pytest.raises(ValueError,match='grid'):
        grid_calibrate(np.array([.005]),{'count':1,'threshold':.001},512,4095,1)
    with pytest.raises(ValueError,match='three-rank'):
        grid_calibrate(np.array([1/4096]),{'count':1,'threshold':.001},512,4095,0)
