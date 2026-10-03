"""Analytical/API tests for the isolated known-model cell development prototype."""
from itertools import combinations, permutations

import numpy as np
import pytest
from scipy.stats import multivariate_t, t

from sca3_compass.robustness_pc_cells import (
    CellSpec, DEFAULT_CELLS, baseline_from_ordered, cell_decision,
    cells_from_ordered, density_inflation, equicorrelation, evaluate_summary,
    fit_cell_lp, ordered_observations, outer_null_bound, parameter_box_radius,
    required_outer_radius, student_location_lipschitz, summarize_bank,
    summarize_bank_many,
)


def test_prespecified_cells_and_original_coordinate_domain():
    assert DEFAULT_CELLS.count == 512
    x = np.array([[0, 0, 0, 0], [6, 6, 6, 6], [-.01, 1, 2, 3], [1, 2, 3, 6.01]])
    ids = cells_from_ordered(ordered_observations(x))
    assert np.all((ids[:2] >= 0) & (ids[:2] < 512))
    np.testing.assert_array_equal(ids[2:], [-1, -1])


def test_cell_assignment_is_permutation_invariant():
    x = np.asarray(list(permutations([.2, 1.3, 2.9, 4.6])))
    assert len(np.unique(cells_from_ordered(ordered_observations(x)))) == 1


@pytest.mark.parametrize('kind', ['projection', 'simes'])
@pytest.mark.parametrize('tau', [.001, .003, .01])
def test_baseline_matches_all_four_intersections(kind, tau):
    rng = np.random.default_rng(15600)
    x = rng.normal(size=(2000, 4)) * 2 + 2
    rho, df = .65, 25
    if kind == 'projection':
        values = [t.sf(x[:, idx].sum(axis=1) / np.sqrt(3 + 6 * rho), df)
                  for idx in combinations(range(4), 3)]
    else:
        values = [np.min(np.sort(t.sf(x[:, idx], df), axis=1) * 3 / np.arange(1, 4), axis=1)
                  for idx in combinations(range(4), 3)]
    expected = np.max(values, axis=0) <= tau
    np.testing.assert_array_equal(baseline_from_ordered(ordered_observations(x), tau, df, rho, kind), expected)


def test_retained_baseline_is_unchanged_outside_box_and_not_double_counted():
    x = np.array([[10, 10, 10, 10], [-1, 0, 1, 2], [6, 6, 6, 6], [2, 2, 2, 2]])
    weights = np.full(512, .7)
    phi = cell_decision(x, weights, .001)
    np.testing.assert_array_equal(phi[:2], [1, 0])
    assert phi[2] == 1
    assert 0 <= phi[3] <= 1
    empty = cell_decision(x, np.zeros(512), .001)
    np.testing.assert_array_equal(empty, baseline_from_ordered(ordered_observations(x), .0009))


def test_summary_moments_equal_direct_observations():
    rng = np.random.default_rng(51623)
    noise = rng.normal(size=(3000, 4))
    means = np.array([[0, 0, 0, 0], [2.8, 2.8, 2.8, 2.8]])
    weights = rng.uniform(size=512)
    summary = summarize_bank(noise, means, .003)
    result = evaluate_summary(summary, weights, .003)
    for i, mu in enumerate(means):
        phi = cell_decision(noise + mu, weights, .003)
        base = baseline_from_ordered(ordered_observations(noise + mu), .003).astype(float)
        np.testing.assert_allclose(result['mean_phi'][i], phi.mean(), atol=1e-14)
        np.testing.assert_allclose(result['mc_se'][i], phi.std(ddof=1) / np.sqrt(len(phi)), atol=1e-14)
        np.testing.assert_allclose(result['paired_mc_se_gain'][i], (phi-base).std(ddof=1) / np.sqrt(len(phi)), atol=1e-14)


def test_shared_sort_summary_is_exactly_single_comparator_summary():
    rng = np.random.default_rng(72161)
    noise = rng.normal(size=(1000, 4))
    means = np.array([[0, 0, 0, 0], [2, 3, 4, 5]])
    multi = summarize_bank_many(noise, means)
    for kind in ('projection', 'simes'):
        for tau in (.001, .003):
            one = summarize_bank(noise, means, tau, baseline=kind)
            for key in one:
                np.testing.assert_array_equal(one[key], multi[f'{kind}_{tau:g}'][key])


def miniature_summary(counts, baseline_counts):
    return {'n': 10000, 'cell_counts': np.asarray(counts),
            'baseline_beta_counts': np.asarray(baseline_counts)}


def test_lp_preserves_all_constraints_and_reveals_infeasible_raw_baseline():
    null = miniature_summary([[10, 0], [0, 10]], [11, 5])
    alt = miniature_summary([[3000, 2000]], [0])
    weights, strict, _ = fit_cell_lp(null, alt, .001, baseline_accounting='strict_empirical')
    assert not strict['success'] and weights is None
    weights, clip, arrays = fit_cell_lp(null, alt, .001, baseline_accounting='clipped_estimate')
    assert clip['success'] and clip['raw_baseline_above_tau_count'] == 1
    assert clip['fitted_max_Ephi_over_tau'] <= 1 + 1e-8
    assert clip['raw_max_Ephi_over_tau'] > 1
    assert not clip['continuous_null_coverage'] and not clip['monte_carlo_upper_bound']
    np.testing.assert_allclose(weights, [.1, .5], atol=1e-8)
    assert arrays['fitted_slack'].shape == (2,)


def test_analytic_baseline_accounting_does_not_use_empirical_slack():
    null = miniature_summary([[10, 0], [0, 10]], [0, 0])
    alt = miniature_summary([[3000, 2000]], [0])
    weights, fit, arrays = fit_cell_lp(null, alt, .001, baseline_accounting='bound')
    assert fit['success']
    np.testing.assert_allclose(weights, [.1, .1], atol=1e-8)
    np.testing.assert_allclose(arrays['baseline_used'], .0009)


def test_box_radius_formula_and_density_bound():
    shape = equicorrelation(.65)
    lo, hi = np.full(4, -.2), np.full(4, .2)
    radius = parameter_box_radius(lo, hi, shape)
    rng = np.random.default_rng(92161)
    draws = rng.uniform(lo, hi, size=(1000, 4))
    norms = np.sqrt(np.einsum('ni,ij,nj->n', draws, np.linalg.inv(shape), draws))
    assert norms.max() <= radius + 1e-12
    np.testing.assert_allclose(parameter_box_radius(lo, hi, np.eye(4)), .4)
    for df in (1.5, 5, 25):
        for delta in draws[:20]:
            x = np.array([2, -3, 4, 1])
            ratio = np.exp(multivariate_t.logpdf(x, delta, shape, df) - multivariate_t.logpdf(x, np.zeros(4), shape, df))
            assert ratio <= density_inflation(radius, df) + 1e-12


def test_outer_bound_covers_unbounded_complement_formula():
    for tau in (.001, .003):
        for df in (1.5, 25):
            radius = required_outer_radius(tau, .9 * tau, df)
            np.testing.assert_allclose(outer_null_bound(radius, df, beta=.9*tau), tau, rtol=1e-8)
            assert outer_null_bound(radius+1, df, beta=.9*tau) < tau


@pytest.mark.parametrize('operation', [
    lambda: CellSpec(min_edges=(0, 1, 1, 6)),
    lambda: ordered_observations([[0, 1, 2, np.nan]]),
    lambda: cells_from_ordered(np.array([[2, 1, 3, 4]])),
    lambda: baseline_from_ordered(np.ones((1, 4)), .01, rho=-.1, kind='simes'),
    lambda: student_location_lipschitz(np.inf),
    lambda: density_inflation(-1, 25),
    lambda: required_outer_radius(.001, .002, 25),
    lambda: cell_decision(np.ones((1, 4)), np.full(512, 2), .001),
])
def test_invalid_inputs_fail_closed(operation):
    with pytest.raises(ValueError):
        operation()
