"""Independent identities, finite-sample coverage, and exchange safeguards."""
from itertools import product
import math

import numpy as np
import pytest
from scipy.stats import binom

from sca3_compass.robustness_pc_cells import (
    DEFAULT_CELLS, baseline_from_ordered, cell_decision, ordered_observations,
    summarize_bank,
)
from sca3_compass.robustness_pc_exchange import (
    PAIRS, alternative_design, audit_frozen, audit_sentinels, calibrate_once,
    exchange_fit, face_membership, kl_lower, kl_upper, noise_batches, null_design,
    power_contrast, summarize_batches,
)


def test_design_full_faces_unequal_negatives_fine_boundary_and_offsets():
    pool, initial, membership = null_design()
    assert len(pool) > 2000 and len(initial) < 200
    assert membership.shape == (len(pool), 4) and membership.any(axis=1).all()
    for face in range(4):
        row = np.zeros(4)
        for a in np.arange(0, 6.001, .0625):
            row[face] = a
            assert np.any(np.all(pool == row, axis=1))
        assert np.any(membership[:, face] & (pool[:, face] > 0) & (np.sum(pool < 0, axis=1) == 3))
    np.testing.assert_array_equal(face_membership(pool), membership)
    assert len(np.unique(pool, axis=0)) == len(pool)
    sentinels = audit_sentinels()
    unseen = {tuple(row) for row in sentinels} - {tuple(row) for row in pool}
    assert len(unseen) >= 50
    assert np.any(np.all(pool < 0, axis=1))
    means, labels = alternative_design()
    assert means.shape == (19, 4)
    assert sum(label.startswith('support2_') for label in labels) == 3


def test_bank_is_reproducible_batch_invariant_and_shared_radial_known_shape():
    first = np.concatenate(list(noise_batches(40001, 12345, 4096)))
    second = np.concatenate(list(noise_batches(40001, 12345, 65536)))
    np.testing.assert_array_equal(first, second)
    # Known Student covariance equals shape * df/(df-2).
    np.testing.assert_allclose(np.diag(np.cov(first.T)), 25 / 23, atol=.045)
    np.testing.assert_allclose(np.corrcoef(first.T)[np.triu_indices(4, 1)], .65, atol=.018)
    assert not np.array_equal(first[:100], np.concatenate(list(noise_batches(100, 12346))))


def test_batched_counts_match_existing_scalar_summaries_for_every_beta():
    noise = np.concatenate(list(noise_batches(1300, 56021, 500)))
    means = np.array([[0., 0., 0., 1.5], [2., 3., 4., 5.]])
    summaries = summarize_batches((noise[:377], noise[377:]), means)
    for key, kind, tau, fraction in PAIRS:
        expected = summarize_bank(noise, means, tau, baseline=kind, beta_fraction=fraction)
        for field in expected:
            np.testing.assert_array_equal(summaries[key][field], expected[field])
        for i, mu in enumerate(means):
            ordered = ordered_observations(noise + mu)
            other = 'projection' if kind == 'simes' else 'simes'
            simple = (baseline_from_ordered(ordered, fraction * tau, kind=kind)
                      | baseline_from_ordered(ordered, (1 - fraction) * tau, kind=other))
            assert simple.sum() == summaries[key]['simple_enhanced_counts'][i]


def synthetic_summaries():
    null, alt = {}, {}
    for key, _, _, _ in PAIRS:
        cells = np.zeros((3, DEFAULT_CELLS.count), dtype=np.int32)
        cells[:, :2] = [[1, 0], [0, 100], [100, 100]]
        objective = np.zeros((1, DEFAULT_CELLS.count), dtype=np.int32)
        objective[0, :2] = [1000, 2000]
        null[key] = {'n': 10000, 'cell_counts': cells, 'baseline_beta_counts': np.zeros(3, int)}
        alt[key] = {'n': 10000, 'cell_counts': objective, 'baseline_beta_counts': np.zeros(1, int)}
    return null, alt


def test_train_exchange_uses_shared_rows_and_preserves_initial_overfit():
    null, alt = synthetic_summaries()
    records = exchange_fit(null, alt, [0], max_rounds=3, add_per_pair=1)
    assert len(records) >= 2
    assert any(records[0]['fits'][key]['pool_violation_count'] > 0 for key, _, _, _ in PAIRS)
    for key, _, tau, fraction in PAIRS:
        final = records[-1]['fits'][key]
        assert final['receipt']['success']
        assert final['pool_added_budget_ratio'].max() <= 1 + 1e-7
        np.testing.assert_allclose(final['arrays']['baseline_used'], fraction * tau)
        assert len(final['arrays']['fitted_slack']) == len(records[-1]['active_indices'])
    assert set(records[0]['active_indices']) <= set(records[-1]['active_indices'])
    assert len(records[0]['fits']) == 12


def test_kl_bounds_endpoints_scale_and_monotonicity():
    alpha, n = .01, 100
    np.testing.assert_allclose(kl_upper(0., n, alpha), -np.expm1(np.log(alpha) / n), atol=1e-15)
    assert kl_upper(1., n, alpha) == 1
    assert kl_lower(0., n, alpha) == 0
    assert kl_upper(0., n, alpha, bound=0) == 0
    means = np.linspace(0, 1, 101)
    upper = kl_upper(means, n, alpha)
    lower = kl_lower(means, n, alpha)
    assert np.all(np.diff(upper) >= 0) and np.all(lower <= means) and np.all(upper >= means)
    np.testing.assert_allclose(kl_upper(.3 * means, n, alpha, bound=.3), .3 * upper, atol=1e-15)
    assert np.all(kl_upper(means, n * 2, alpha) <= upper + 1e-15)


def test_kl_finite_sample_binomial_coverage_exact_enumeration():
    n, alpha = 30, .05
    k = np.arange(n + 1)
    upper = kl_upper(k / n, n, alpha)
    lower = kl_lower(k / n, n, alpha)
    for p in (.001, .01, .1, .3, .5, .8, .99):
        mass = binom.pmf(k, n, p)
        assert mass[upper < p].sum() <= alpha + 1e-13
        assert mass[lower > p].sum() <= alpha + 1e-13


def test_kl_fractional_scalar_coverage_multinomial_exact_enumeration():
    # X in {0,.2,.7}; no binomial-count interpretation is possible here.
    n, alpha = 9, .05
    probabilities = (.65, .25, .1)
    expectation = .2 * probabilities[1] + .7 * probabilities[2]
    failure = 0.
    for a, b in product(range(n + 1), repeat=2):
        c = n - a - b
        if c < 0:
            continue
        mass = math.factorial(n) / (math.factorial(a) * math.factorial(b) * math.factorial(c))
        mass *= probabilities[0] ** a * probabilities[1] ** b * probabilities[2] ** c
        estimate = (.2 * b + .7 * c) / n
        if kl_upper(estimate, n, alpha, bound=.7) < expectation:
            failure += mass
    assert failure <= alpha + 1e-13


def test_calibration_shrinks_once_with_full_multiplicity_and_audit_never_refits():
    noise = np.concatenate(list(noise_batches(1200, 91777)))
    means = np.array([[0., 0., 0., 2.], [-.1, -.2, -.3, 1.5]])
    summaries = summarize_batches((noise,), means)
    weights = {key: np.full(512, .7) for key, _, _, _ in PAIRS}
    calibrated = calibrate_once(summaries, weights)
    for key, _, tau, fraction in PAIRS:
        row = calibrated[key]
        assert row['multiplicity'] == 24
        assert row['alpha_per_row'] == .01 / 24
        assert 0 < row['shrink'] < 1
        assert np.max(row['finite_row_high_probability_total_upper']) <= fraction * tau + .8 * (1 - fraction) * tau + 1e-15
        np.testing.assert_array_equal(weights[key], np.full(512, .7))
    # Deliberately adverse audit counts: expose failure without further shrink.
    frozen = {key: row['weights'].copy() for key, row in calibrated.items()}
    for summary in summaries.values():
        summary['cell_counts'][:, 0] = 1190
    # Keep counts a valid partition for the upper-bound range checks.
    for summary in summaries.values():
        summary['cell_counts'][:, 1:] = 0
    result = audit_frozen(summaries, frozen)
    assert any(row['rows_upper_above_tau'] for row in result.values())
    for key in frozen:
        np.testing.assert_array_equal(frozen[key], calibrated[key]['weights'])
        assert result[key]['continuous_null_coverage'] is False


@pytest.mark.parametrize('comparison', ['tau', 'simple'])
def test_paired_power_moments_and_signed_components_match_direct(comparison):
    noise = np.concatenate(list(noise_batches(2500, 41671)))
    mu = np.array([2., 3., 4., 2.])
    summaries = summarize_batches((noise,), mu[None, :])
    rng = np.random.default_rng(783)
    weights = rng.uniform(size=512)
    for key, kind, tau, fraction in PAIRS:
        result = power_contrast(summaries[key], weights, tau, comparison=comparison)
        phi = cell_decision(noise + mu, weights, tau, baseline=kind, beta_fraction=fraction)
        ordered = ordered_observations(noise + mu)
        if comparison == 'tau':
            baseline = baseline_from_ordered(ordered, tau, kind=kind)
        else:
            other = 'simes' if kind == 'projection' else 'projection'
            baseline = (baseline_from_ordered(ordered, fraction * tau, kind=kind)
                        | baseline_from_ordered(ordered, (1 - fraction) * tau, kind=other))
        delta = phi - baseline
        np.testing.assert_allclose(result['gain'][0], delta.mean(), atol=1e-14)
        np.testing.assert_allclose(result['positive_mass'][0], np.maximum(delta, 0).mean(), atol=1e-14)
        np.testing.assert_allclose(result['negative_mass'][0], np.maximum(-delta, 0).mean(), atol=1e-14)
        np.testing.assert_allclose(result['paired_mc_se'][0], delta.std(ddof=1) / np.sqrt(len(delta)), atol=1e-14)
        assert result['simultaneous_gain_lower'][0] <= delta.mean() <= result['simultaneous_gain_upper'][0]


@pytest.mark.parametrize('operation', [
    lambda: kl_upper(-.1, 10, .05), lambda: kl_upper(.1, 0, .05),
    lambda: kl_upper(.1, 10, 0), lambda: kl_upper(.1, 10, 1),
    lambda: kl_upper(.1, 10, .05, bound=0), lambda: kl_upper(np.nan, 10, .05),
    lambda: list(noise_batches(10, 10, 100000)), lambda: face_membership([[1, 1, 0, 0]]),
    lambda: summarize_batches((), [[0, 0, 0, 0]]),
])
def test_invalid_inputs_fail_closed(operation):
    with pytest.raises(ValueError):
        operation()
