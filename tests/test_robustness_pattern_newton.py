import json
from pathlib import Path

import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sca3_compass import robustness_patterns as patterns


def test_newton_solves_same_concave_weight_objective_without_relaxed_tolerance():
    rng=np.random.default_rng(60701)
    m=rng.normal(size=(80,4))
    m[:20]+=3.5
    args=(m,np.ones(80),np.eye(4),25)
    new=patterns.fit_pattern_mixture(*args,max_iter=1,slsqp_max_iter=0)
    old=patterns.fit_pattern_mixture(*args,newton_max_iter=0)
    assert new['diagnostics']['newton']['attempted']
    assert new['diagnostics']['converged'] and old['diagnostics']['converged']
    assert new['diagnostics']['relative_duality_gap']<=1e-7
    assert abs(new['diagnostics']['objective']-old['diagnostics']['objective'])<1e-8
    assert np.max(np.abs(new['weights']-old['weights']))<1e-6


@threadpool_limits.wrap(limits=1)
def test_original_four_failures_preserved_and_recovered():
    fixture=Path(__file__).resolve().parents[1]/'artifacts/robustness/pattern-R0055-stationarity-inputs.json'
    if not fixture.exists():
        pytest.skip('Optional large research-regression fixture not shipped')
    rows=json.loads(fixture.read_text(encoding='utf-8'))['records']
    assert len(rows)==4
    for row in rows:
        df=np.inf if row['df'] is None else row['df']
        args=(row['training_means'],row['variance'],row['shape'],df)
        before=patterns.fit_pattern_mixture(*args,newton_max_iter=0)
        after=patterns.fit_pattern_mixture(*args)
        assert not before['diagnostics']['converged']
        np.testing.assert_array_equal(before['weights'],row['before_weights'])
        assert after['diagnostics']['converged']
        assert after['diagnostics']['relative_duality_gap']<=row['before']['tolerance']
        assert abs(after['diagnostics']['objective']-row['before']['objective'])<1e-8
        assert after['diagnostics']['newton']['accepted_steps']>=1


def test_numerical_failure_remains_unconverged(monkeypatch):
    def failed(*args,**kwargs):
        raise np.linalg.LinAlgError('injected factorization failure')
    monkeypatch.setattr(patterns,'cho_factor',failed)
    fit=patterns.fit_pattern_mixture(np.ones((20,4)),np.ones(20),np.eye(4),25,
        max_iter=0,slsqp_max_iter=0)
    assert not fit['diagnostics']['converged']
    assert 'injected' in fit['diagnostics']['newton']['failure']


@pytest.mark.parametrize('bad',[-1,.5,False])
def test_newton_budget_validation(bad):
    with pytest.raises(ValueError,match='newton_max_iter'):
        patterns.fit_pattern_mixture(np.ones((20,4)),np.ones(20),np.eye(4),25,newton_max_iter=bad)
