"""Constrained optimization correctness, not scientific calibration."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sca3_compass import robustness_continuous_scale as module


@pytest.mark.parametrize('floor',[0.,.25,.5,.8])
@pytest.mark.parametrize('seed',[11,419,3037])
@threadpool_limits.wrap(limits=1)
def test_constrained_weights_agree_with_independent_em(floor,seed):
    rng=np.random.default_rng(seed)
    density=rng.normal(size=(37,7))*2;density[:,0]-=2
    penalty=module._penalty(7)
    weights,state,_=module._fit_weights_floor(density,floor)
    # Independent constrained EM, including its closed-form active-face M
    # step. SLSQP previously reported success while up to .0743 below this
    # concave optimum; a status flag is not an independent certificate.
    other=np.r_[max(floor,.2),np.full(6,(1-max(floor,.2))/6)]
    offset=density.max(1);likelihood=np.exp(density-offset[:,None])
    for _ in range(100000):
        denominator=likelihood@other
        counts=(likelihood*other/denominator[:,None]).sum(0)+penalty
        proposal=counts/counts.sum()
        if proposal[0]<floor:
            proposal[0]=floor;proposal[1:]=(1-floor)*counts[1:]/counts[1:].sum()
        other=proposal
        denominator=likelihood@other
        gradient=(likelihood/denominator[:,None]).sum(0)+penalty/other
        gap=max(0.,floor*gradient[0]+(1-floor)*gradient.max()-other@gradient)
        if gap/(len(density)+penalty.sum())<1e-10:
            break
    else:
        pytest.fail('Independent constrained EM reference did not converge')
    objective=np.sum(offset+np.log(denominator))+np.dot(penalty,np.log(other))
    assert weights[0]>=floor and np.all(weights>0)
    assert state['weight_kkt_residual']<=1e-8
    assert objective==pytest.approx(state['objective'],abs=1e-7)
    np.testing.assert_allclose(weights,other,atol=2e-6)
    if weights[0]==floor:
        g=state['gradient'];n=state['normalizer']
        assert state['conditional_weight_gap']==pytest.approx(
            max(0.,floor*g[0]+(1-floor)*g.max()-weights@g),abs=1e-12)
        assert state['conditional_weight_gap']/n<1e-8


@threadpool_limits.wrap(limits=1)
def test_zero_floor_is_exact_old_weight_path():
    density=np.random.default_rng(331).normal(size=(80,9))
    a,s,_=module._fit_weights(density)
    b,t,_=module._fit_weights_floor(density,0.)
    np.testing.assert_array_equal(a,b)
    assert s['objective']==t['objective']


@pytest.mark.parametrize('floor',[True,-.1,1.,float('nan'),float('inf'),[.5]])
def test_invalid_floor_rejected(floor):
    with pytest.raises(ValueError,match='minimum_null_weight'):
        module._fit_weights_floor(np.zeros((8,3)),floor)
    with pytest.raises(ValueError,match='minimum_null_weight'):
        module.fit_continuous_scale(np.zeros((8,4)),np.ones(8),np.eye(4),np.inf,minimum_null_weight=floor)


@threadpool_limits.wrap(limits=1)
def test_known_noise_fixture_floor_is_not_silently_ignored():
    x=np.random.default_rng(80317).normal(size=(128,4))*3
    fit=module.fit_continuous_scale(x,np.ones(128),np.eye(4),np.inf,
        covariances=np.array([np.zeros((4,4)),9*np.eye(4)]),minimum_null_weight=.5)
    assert fit['converged']
    assert fit['weights'][0]>=.5
    assert fit['diagnostics']['null_floor_is_assumed_not_estimated']
    assert any('ADDITIONAL working sparsity' in text for text in fit['assumptions'])
    # No assertion that the scale equals its known value: finite-sample bias
    # and nuisance model identification remain actual research questions.
