"""Rank-calibrator wiring and holdout isolation, not a power experiment."""
from copy import deepcopy
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_methods import evaluate_candidates
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_pattern_test import pattern_test_candidates


@threadpool_limits.wrap(limits=1)
def test_rank_integration_old_arrays_exact_and_training_only():
    rng=np.random.default_rng(171193)
    z=rng.normal(size=(64,4,6));z[:12,:2]+=3.5;z[12:24]-=3.5
    x=rng.normal(size=(4,32,6))
    _,diag=evaluate_candidates(z,x,'frontier')
    base=energy_prior_candidates(z,x,diag)['target_only_weighted_cone_PC']
    old=pattern_test_candidates(z,deepcopy(diag),base,pilot=True,pilot_selection=True)
    result=pattern_test_candidates(z,diag,base,pilot=True,pilot_selection=True,rank_budget=True)
    for key in old:
        np.testing.assert_array_equal(result[key],old[key])
    for mode in ['pattern_projection','pattern_support_simes','ordinary_bonf','ordinary_simes','cone']:
        assert result[f'rank80_{mode}_eBH'].shape==(64,2)
        assert np.isfinite(result[f'rank80_{mode}_eBH']).all()
    changed=z.copy();changed[::2]+=50
    other=deepcopy(diag)
    pattern_test_candidates(changed,other,base,pilot=True,rank_budget=True)
    a=diag['pattern_testing']['folds'][0]['rank_budget_calibration']
    assert a==other['pattern_testing']['folds'][0]['rank_budget_calibration']
    for receipt in a.values():
        assert receipt['selection']['family']==128
        assert receipt['selection']['training_gene_count']==32


def test_rank_requires_pilot():
    with pytest.raises(ValueError):
        pattern_test_candidates(np.ones((32,4,6)),{'energy_prior':{}},np.ones((32,2)),rank_budget=True)
