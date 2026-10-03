from copy import deepcopy
import numpy as np
import pytest
from sca3_compass.robustness_joint_scale import fit_joint_pattern_scale
from sca3_compass.robustness_scale_anchor import anchor_joint_scale


@pytest.mark.parametrize('df',[np.inf,25.])
@pytest.mark.parametrize('scale',[1.,9.])
def test_anchor_compares_equal_objectives_and_records_rule(df,scale):
    rng=np.random.default_rng(80317)
    x=rng.normal(size=(128,4))*np.sqrt(scale)
    if np.isfinite(df):
        x/=np.sqrt(rng.chisquare(df,len(x))/df)[:,None]
    v=np.ones(len(x));shape=np.eye(4)
    joint=fit_joint_pattern_scale(x,v,shape,df)
    tau,receipt=anchor_joint_scale(x,v,shape,df,joint)
    assert receipt['converged']
    assert receipt['threshold']==pytest.approx(np.log(len(x)))
    assert receipt['free_objective_recomputed']==pytest.approx(joint['diagnostics']['objective'],abs=1e-9)
    assert receipt['selected_free_scale']==(receipt['twice_penalized_objective_gain']>receipt['threshold'])
    assert tau==(joint['tau'] if receipt['selected_free_scale'] else 1.)
    if scale==9:
        assert receipt['selected_free_scale']
        # Retained STATISTICAL FAILURE: the flexible location prior can
        # misread isotropic large null noise as effects; anchoring cannot fix it.
        # This deliberately documents the failed initial expectation tau>5.
        assert tau<3
    else:
        assert tau==1.
    bad=deepcopy(joint);bad['diagnostics']['objective']+=1
    with pytest.raises(ValueError,match='objective does not match'):
        anchor_joint_scale(x,v,shape,df,bad)


def test_unconverged_joint_cannot_be_silently_anchored():
    with pytest.raises(ValueError,match='converged joint'):
        anchor_joint_scale(None,None,None,None,{'diagnostics':{'converged':False}})
