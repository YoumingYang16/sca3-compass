from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from robustness_r2_component_audit import partition


def test_partition_uses_equal_scene_not_pooled_weights_and_preserves_failures():
    x=[np.array([.1,.2]),np.array([.3,.4,.5,.6])]
    g=[['both_success','R1_failure_only'],['both_success','K_failure_only','both_failure','both_failure']]
    r=partition(x,g)
    assert r['total_I']==pytest.approx(.3)
    assert sum(v['families'] for v in r['groups'].values())==6
    assert r['groups']['R1_failure_only']['equal_scene_weighted_fraction']==.25
    assert r['groups']['R1_failure_only']['contribution_to_total_I']==.05
    assert r['groups']['both_failure']['conditional_mean_I']==pytest.approx(.55)


def test_empty_group_is_undefined_not_zero_and_unknown_group_rejected():
    r=partition([np.array([.1,.2])],[['both_success','both_success']])
    assert r['groups']['K_failure_only']['conditional_mean_I'] is None
    with pytest.raises(ValueError):partition([np.array([1])],[['silently_dropped']])
