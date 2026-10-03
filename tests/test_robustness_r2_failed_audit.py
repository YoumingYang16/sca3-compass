from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from robustness_r2_failed_audit import completion_bounds,paired_macro


def test_failed_family_is_not_deleted_from_denominator_or_imputed_as_zero():
    assert completion_bounds([0,1,0],4)==[.25,.5]
    assert completion_bounds([],2)==[0,1]
    assert completion_bounds([0,1],2)==[.5,.5]
    with pytest.raises(ValueError):completion_bounds([2],2)
    with pytest.raises(ValueError):completion_bounds([0,1],1)
    # No add-n/subtract-n cancellation when the sample is fully observed.
    x=np.array([.003363061650536876]*800)
    assert completion_bounds(x,800)[0]==completion_bounds(x,800)[1]


def test_pairing_uses_scene_weights_and_missing_strata_are_rejected():
    arrays=[np.array([0.,.2]),np.array([.2,.4,.6,.8])]
    value=paired_macro(arrays)
    assert value['mean_difference']==pytest.approx(.3)
    assert value['paired_mcse']==pytest.approx(np.sqrt(.02/2+np.var(arrays[1],ddof=1)/4)/2)
    assert 'DESCRIPTIVE' in value['role']
    with pytest.raises(ValueError):paired_macro([np.array([0,np.nan])])
