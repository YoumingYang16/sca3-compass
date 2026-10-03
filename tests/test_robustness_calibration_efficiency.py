from copy import deepcopy
import gzip
import json
from pathlib import Path
import numpy as np
import pytest
from sca3_compass.robustness_calibration_efficiency import evaluate, rho_guard_theta
from sca3_compass.robustness_bootstrap_guard import components, variance

ROOT = Path(__file__).resolve().parents[1]


def theta(df=5.):
    return {'rho': .8, 'shape': .65*np.ones((4,4))+.35*np.eye(4), 'dimension': 20,
            'prior': {'df': df, 'scatter': .6, 'gaussian_bic_selected': np.isinf(df)}}


@pytest.mark.parametrize('df', [1.5, 5., np.inf])
def test_scalar_guard_preserves_target_contrast_scale_and_maximizes_all_tails(df):
    original = theta(df)
    values = [.1, .8, .91, .86]
    guarded = rho_guard_theta(original, values)
    assert guarded['rho'] == .91
    assert guarded['prior']['scatter']*(1-guarded['rho']) == pytest.approx(.12)
    y = np.arange(140, dtype=float).reshape(7,4,5)/100-.7
    means = np.arange(28, dtype=float).reshape(7,4)/5-1
    profile = np.array([[2,2,0,0],[0,0,2,2]], float)
    bank = [rho_guard_theta(original, [rho]) for rho in values]
    full = components(means, y, profile, original, bank)
    scalar = components(means, y, profile, original, [original, guarded])
    for mode in full['guard']:
        np.testing.assert_allclose(scalar['guard'][mode], full['guard'][mode], rtol=2e-13, atol=1e-15)
        assert np.all(scalar['guard'][mode] >= scalar['plugin'][mode])
    assert np.all(variance(y, guarded)[0] >= variance(y, original)[0])


@pytest.mark.parametrize('values', [[], [np.nan], [1.], [-.2]])
def test_invalid_rho_rejected(values):
    with pytest.raises(ValueError):
        rho_guard_theta(theta(), values)


@pytest.mark.parametrize('case', [0, 27, 59, 63, 75])
def test_fresh_and_cached_exact_P_parity_and_K_parity(case):
    base = ROOT/f'artifacts/robustness/R0072-repetitions/case-{case:04}/rep-000000'
    if not base.with_suffix('.json.gz').exists():
        pytest.skip('Archived local DEV fixture unavailable')
    with gzip.open(base.with_suffix('.json.gz'), 'rt', encoding='utf-8') as stream:
        record = json.load(stream)
    with np.load(str(base)+'-input.npz') as data:
        z, cal = data['z'], data['calibration']
    seed = int(np.random.SeedSequence([7205107, case, 0, 913]).generate_state(1)[0])
    cached, cached_info = evaluate(z, cal, seed=seed, cached_folds=record['diagnostics_R1']['folds'])
    fresh, fresh_info = evaluate(z, cal, seed=seed)
    only_p, _ = evaluate(z, cal, seed=seed, mode='P')
    with np.load(str(base)+'-evidence.npz') as old:
        for name, value in fresh.items():
            np.testing.assert_array_equal(value, cached[name])
            if name.startswith('R2P_'):
                np.testing.assert_array_equal(value, old[name.replace('R2P_', 'R1B_plugin_')])
                np.testing.assert_array_equal(value, only_p[name])
    for name in cached_info['held_pvalues']:
        np.testing.assert_array_equal(cached_info['held_pvalues'][name], fresh_info['held_pvalues'][name])


def test_K_failure_blocks_only_its_fold_and_preserves_P(monkeypatch):
    import sca3_compass.robustness_calibration_efficiency as mod
    original = theta(np.inf)
    monkeypatch.setattr(mod, 'nuisance', lambda *a: deepcopy(original))
    monkeypatch.setattr(mod, 'learn_profiles', lambda *a: (np.ones((2,4)), np.ones(2), {'fallback': False}))
    calls = []
    def failed_bank(*args):
        calls.append(1)
        return [.8], [{'status': 'failed' if len(calls)==1 else 'completed'}], len(calls)!=1
    monkeypatch.setattr(mod, 'calibration_rho_bank', failed_bank)
    z = 20+np.arange(16*4*6, dtype=float).reshape(16,4,6)/1000
    out, info = evaluate(z, np.ones((4,16,6)))
    assert not info['folds'][0]['K_success'] and info['folds'][1]['K_success']
    assert np.all(out['R2K_pilotc0.5_projection_gate_eBH'][::2] == 0)
    assert np.any(out['R2P_pilotc0.5_projection_gate_eBH'][::2] > 0)


def test_held_observations_cannot_modify_own_training():
    from sca3_compass.robustness_pattern_test import _json_value
    z = np.random.default_rng(852).normal(size=(16,4,6))
    cal = np.random.default_rng(853).normal(size=(4,16,6))
    _, old = evaluate(z, cal, draws=1, seed=52)
    changed = z.copy()
    changed[::2] *= 2
    changed[::2,:,0] += 4
    _, new = evaluate(changed, cal, draws=1, seed=52)
    assert json.dumps(_json_value(old['folds'][0]), sort_keys=True) == json.dumps(_json_value(new['folds'][0]), sort_keys=True)
