"""Deterministic unit checks and exact step-function integrals; no simulations."""

from itertools import pairwise, product
from math import fsum

import numpy as np
import pytest

from sca3_compass.molecular_methods import ebh
from sca3_compass.robustness_block_e import (
    block_bh_evalues,
    block_bh_evalues_reference,
)
from sca3_compass.robustness_calibrators import by_calibrator

IMPLEMENTATIONS = (block_bh_evalues, block_bh_evalues_reference)


def _bh_count(p, alpha):
    """Independent literal sort-and-step-up oracle, including zero rejections."""
    ordered = sorted(np.asarray(p).ravel())
    passing = [j for j, value in enumerate(ordered, 1)
               if value <= alpha * j / len(ordered)]
    return max(passing, default=0)


def _literal_block(p, alpha):
    p = np.asarray(p, dtype=float)
    counts = []
    for gene in range(len(p)):
        zeroed = p.copy()
        zeroed[gene, :] = 0
        counts.append(_bh_count(zeroed, alpha))
    counts = np.array(counts)
    tau = alpha * counts / p.size
    return (p <= tau[:, None]) / tau[:, None], counts, tau


def _single_sign_e(p, alpha):
    """Deliberately invalid under arbitrary within-pair dependence."""
    flat = np.asarray(p).ravel()
    e = np.zeros(flat.size)
    for sign in range(flat.size):
        zeroed = flat.copy()
        zeroed[sign] = 0
        tau = alpha * _bh_count(zeroed, alpha) / flat.size
        e[sign] = (flat[sign] <= tau) / tau
    return e.reshape(np.shape(p))


def _integral(breakpoints, function):
    """Exactly integrate a piecewise constant function on its full partition."""
    cuts = np.unique(np.r_[0.0, breakpoints, 1.0])
    assert cuts[0] == 0 and cuts[-1] == 1
    return fsum((right - left) * function(left + (right - left) / 2)
                for left, right in pairwise(cuts))


def _ebh_count(e, level):
    flat = np.sort(np.asarray(e).ravel())[::-1]
    passing = np.flatnonzero(flat >= flat.size / (level * np.arange(1, flat.size + 1)))
    return int(passing[-1] + 1) if passing.size else 0


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_known_counts_and_diagnostics(implementation):
    p = np.array([[.25, .8], [.5, .5]])
    e, diagnostic = implementation(p, .5)
    np.testing.assert_array_equal(diagnostic["R"], [4, 3])
    np.testing.assert_array_equal(diagnostic["tau"], [.5, .375])
    np.testing.assert_array_equal(e, [[2, 0], [0, 0]])
    assert diagnostic["n_genes"] == 2
    assert diagnostic["family_size"] == 4
    assert diagnostic["alpha"] == .5
    assert diagnostic["excluded_signs_per_gene"] == 2
    assert diagnostic["conditions"]
    assert diagnostic["conditions_verified"] is False
    assert diagnostic["arbitrary_gene_dependence_valid"] is False


@pytest.mark.parametrize("n,grid,alpha", [
    (1, [0, .025, .05, 1], .05),
    (2, [0, .125, .25, .375, .5, 1], .5),
    (3, [0, .25, 1], .75),
])
def test_exhaustive_small_arrays_match_literal_pair_zeroing(n, grid, alpha):
    for entries in product(grid, repeat=2 * n):
        p = np.array(entries).reshape(n, 2)
        expected_e, expected_r, expected_tau = _literal_block(p, alpha)
        for implementation in IMPLEMENTATIONS:
            e, diagnostic = implementation(p, alpha)
            np.testing.assert_array_equal(diagnostic["R"], expected_r)
            np.testing.assert_array_equal(diagnostic["tau"], expected_tau)
            np.testing.assert_array_equal(e, expected_e)


@pytest.mark.parametrize("n", [7, 32, 101])
def test_larger_deterministic_rank_patterns_match_reference(n):
    # Permutations, duplicates, exact BH cutoffs, and one-ULP neighbors.
    alpha = .05
    cuts = alpha * np.arange(1, 2 * n + 1) / (2 * n)
    for flat in [cuts, cuts[::-1], np.roll(cuts, n), np.nextafter(cuts, 1),
                 np.nextafter(cuts, 0), np.tile([0, .001, .02, 1], n)[:2 * n]]:
        p = flat.reshape(n, 2)
        expected_e, expected_r, expected_tau = _literal_block(p, alpha)
        for implementation in IMPLEMENTATIONS:
            e, diagnostic = implementation(p, alpha)
            np.testing.assert_array_equal(diagnostic["R"], expected_r)
            np.testing.assert_array_equal(diagnostic["tau"], expected_tau)
            np.testing.assert_array_equal(e, expected_e)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_neither_own_value_affects_threshold_including_reordering_and_ties(implementation):
    p = np.array([[.001, .99], [.01, .01], [.02, .7], [0, 1]])
    _, initial = implementation(p)
    for gene in range(len(p)):
        for pair in product([0, .001, .01, .02, .05, .5, 1], repeat=2):
            changed = p.copy()
            changed[gene] = pair
            e, diagnostic = implementation(changed)
            assert diagnostic["R"][gene] == initial["R"][gene]
            assert diagnostic["tau"][gene] == initial["tau"][gene]
            np.testing.assert_array_equal(
                e[gene], (changed[gene] <= initial["tau"][gene]) / initial["tau"][gene],
            )


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_bh_ties_use_last_passing_rank_not_number_of_passes(implementation):
    # After zeroing gene 0: [0,0,.5,.5]. Rank 3 fails, but rank 4 passes.
    p = np.array([[.9, .8], [.5, .5]])
    _, diagnostic = implementation(p, .5)
    np.testing.assert_array_equal(diagnostic["R"], [4, 2])
    p[1] = np.nextafter(.5, 1)
    _, diagnostic = implementation(p, .5)
    assert diagnostic["R"][0] == 2


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_indicator_equality_is_inclusive_without_tolerance(implementation):
    p = np.array([[.25, np.nextafter(.25, 1)], [1, 1]])
    e, diagnostic = implementation(p, .5)
    assert diagnostic["tau"][0] == .25
    np.testing.assert_array_equal(e[0], [4, 0])
    p[0, 1] = np.nextafter(.25, 0)
    np.testing.assert_array_equal(implementation(p, .5)[0][0], [4, 4])


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_permutations_noncontiguous_and_readonly_inputs(implementation):
    p = np.array([[.01, .02], [0, 0], [.04, .04], [1, 1], [.001, .999]])
    before = p.copy()
    p.setflags(write=False)
    e, diagnostic = implementation(p)
    order = [3, 1, 4, 2, 0]
    shuffled_e, shuffled = implementation(p[order, ::-1])
    np.testing.assert_array_equal(shuffled_e, e[order, ::-1])
    np.testing.assert_array_equal(shuffled["tau"], diagnostic["tau"][order])
    reversed_e, reversed_diag = implementation(p[::-1, ::-1])
    np.testing.assert_array_equal(reversed_e, e[::-1, ::-1])
    np.testing.assert_array_equal(reversed_diag["R"], diagnostic["R"][::-1])
    np.testing.assert_array_equal(p, before)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("alpha", [.025, .05, .125, .75])
@pytest.mark.parametrize("fixed", [
    [[0, 1]],
    [[0, 0], [.01, .9]],
    [[.001, .999], [.02, .98], [.2, .8], [1, 1]],
])
def test_exact_conditional_uniform_expectations_for_opposite_signs(implementation, alpha, fixed):
    p = np.array(fixed, dtype=float)
    # All possible thresholds and their reflections give a complete exact
    # breakpoint partition. Other genes are held fixed throughout each integral.
    grid = alpha * np.arange(2, p.size + 1) / p.size
    breaks = np.r_[grid, 1 - grid]
    for gene in range(len(p)):
        expected_tau = _literal_block(p, alpha)[2][gene]
        for sign in [0, 1]:
            def evaluate(u, gene=gene, sign=sign, expected_tau=expected_tau):
                varied = p.copy()
                varied[gene] = [u, 1 - u]
                e, diagnostic = implementation(varied, alpha)
                assert diagnostic["tau"][gene] == expected_tau
                return e[gene, sign]

            assert _integral(breaks, evaluate) == pytest.approx(1, rel=0, abs=4e-14)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_exact_superuniform_expectation_with_dependent_mate(implementation):
    # max(U, c) is superuniform; the mate U**2 need not be a null p-value.
    p = np.array([[0, 0], [.01, .99], [.2, .8]])
    tau = _literal_block(p, .125)[2][0]
    for floor in [tau / 2, tau, 2 * tau]:
        def evaluate(u, floor=floor):
            varied = p.copy()
            varied[0] = [max(u, floor), u * u]
            return implementation(varied, .125)[0][0, 0]

        assert _integral([floor, tau], evaluate) == pytest.approx(
            1 if floor <= tau else 0, rel=0, abs=1e-14,
        )


def test_single_sign_deletion_can_have_expectation_three_halves_at_small_alpha():
    # Both coordinates are marginally Uniform(0,1). Keeping the mate lets
    # tau(U) be alpha/2 for 0<U<alpha/2, then alpha for alpha/2<U<alpha.
    alpha = .125
    half = alpha / 2
    breaks = [half, alpha, alpha + half]

    def pair(u):
        return np.array([[u, (u - half) % 1]])

    broken = _integral(breaks, lambda u: _single_sign_e(pair(u), alpha)[0, 0])
    assert broken == 1.5
    for implementation in IMPLEMENTATIONS:
        repaired = _integral(
            breaks, lambda u, implementation=implementation:
            implementation(pair(u), alpha)[0][0, 0],
        )
        assert repaired == 1


def test_single_sign_deletion_also_fails_for_exact_opposite_signs():
    # At alpha=3/4, tau=3/8 below U=1/4 and 3/4 above it.
    alpha = .75
    breaks = [1 - alpha, alpha / 2, alpha, 1 - alpha / 2]
    expectation = _integral(
        breaks, lambda u: _single_sign_e([[u, 1 - u]], alpha)[0, 0],
    )
    assert expectation == pytest.approx(4 / 3, rel=0, abs=1e-14)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_arbitrary_gene_dependence_invalidates_block_expectation(implementation):
    # Every sign is marginally uniform, and both genes have complementary
    # signs, but the two genes share U. tau(U)=a on (0,a), b on (a,b).
    alpha = .125
    a, b = alpha / 2, 3 * alpha / 4
    shift = 1 - a - b
    grid = np.r_[0, alpha * np.arange(1, 5) / 4, 1]
    breaks = np.r_[grid, 1 - grid, (grid - shift) % 1, (1 - grid - shift) % 1]

    def evaluate(u):
        v = (u + shift) % 1
        return implementation([[u, 1 - u], [v, 1 - v]], alpha)[0][0, 0]

    assert _integral(breaks, evaluate) == pytest.approx(4 / 3, rel=0, abs=1e-14)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_not_bh_equivalent_even_for_one_complementary_pair(implementation):
    p = [[.01, .99]]
    e, diagnostic = implementation(p)
    assert _bh_count(p, .05) == 1
    np.testing.assert_array_equal(diagnostic["R"], [2])
    np.testing.assert_array_equal(e, [[20, 0]])
    assert _ebh_count(e, .05) == 0


@pytest.mark.parametrize("alpha", [.05, .125, .49])
def test_complementary_pairs_have_no_standalone_ebh_rejections_at_same_level(alpha):
    for u in product([0, .001, .02, .04, .5, .98, 1], repeat=3):
        u = np.array(u)
        e, _ = block_bh_evalues(np.column_stack((u, 1 - u)), alpha)
        assert _ebh_count(e, alpha) == 0


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_smaller_construction_level_recovers_two_signs_with_fixed_capped_mixture(implementation):
    # A deterministic alternative fixture, not a repeated-sampling power claim.
    p = np.array([[.001, .999], [.008, .992], [.3, .7], [.4, .6]])
    alpha0, final_alpha = .025, .05
    block, diagnostic = implementation(p, alpha0)
    same_level, _ = implementation(p, final_alpha)
    capped = by_calibrator(p, family=p.size, alpha=final_alpha, cap=4)
    mixed = .2 * capped + .8 * block
    expected = np.array([[True, False], [True, False], [False, False], [False, False]])

    np.testing.assert_array_equal(diagnostic["R"], [3, 3, 4, 4])
    np.testing.assert_allclose(block[:2, 0], [320 / 3, 320 / 3], rtol=0, atol=3e-14)
    np.testing.assert_allclose(mixed[:2, 0], [352 / 3, 96], rtol=0, atol=3e-14)
    np.testing.assert_array_equal(ebh(block, final_alpha), expected)
    np.testing.assert_array_equal(ebh(mixed, final_alpha), expected)
    assert ebh(capped, final_alpha).sum() == 1
    assert not ebh(same_level, final_alpha).any()
    assert not ebh(.2 * capped + .8 * same_level, final_alpha).any()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_half_level_construction_allows_even_one_strong_sign_at_inclusive_boundary(implementation):
    p = np.array([[.001, .999], [.3, .7], [.4, .6], [.5, .5]])
    block, diagnostic = implementation(p, .025)
    assert diagnostic["R"][0] == 2
    assert block[0, 0] == p.size / (.05 * 1) == 160
    assert ebh(block, .05).sum() == 1
    assert ebh(block, .05)[0, 0]
    capped = by_calibrator(p, family=p.size, alpha=.05, cap=4)
    assert ebh(.2 * capped + .8 * block, .05)[0, 0]


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_fixed_mixture_at_distinct_levels_has_exact_conditional_uniform_expectations(implementation):
    p = np.array([[.001, .999], [.008, .992], [.3, .7], [.4, .6]])
    alpha0, final_alpha, cap = .025, .05, 4
    harmonic = np.sum(1 / np.arange(1, cap + 1))
    grid = np.r_[alpha0 * np.arange(2, p.size + 1) / p.size,
                 final_alpha * np.arange(1, cap + 1) / (p.size * harmonic)]
    breaks = np.r_[grid, 1 - grid]
    for gene in range(len(p)):
        for sign in [0, 1]:
            def evaluate(u, gene=gene, sign=sign):
                varied = p.copy()
                varied[gene] = [u, 1 - u]
                block = implementation(varied, alpha0)[0]
                capped = by_calibrator(varied, family=p.size, alpha=final_alpha, cap=cap)
                return (.2 * capped + .8 * block)[gene, sign]

            assert _integral(breaks, evaluate) == pytest.approx(1, rel=0, abs=4e-14)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_smaller_construction_level_can_still_lose_all_near_boundary_alternatives(implementation):
    # For k=2 equal small signs x=.02: alpha0*(k+1)/m=.01875 < x,
    # while ordinary BH at final alpha has x <= final_alpha*k/m=.025.
    p = np.array([[.02, .98], [.02, .98]])
    block, diagnostic = implementation(p, .025)
    np.testing.assert_array_equal(diagnostic["R"], [2, 2])
    np.testing.assert_array_equal(diagnostic["tau"], [.0125, .0125])
    np.testing.assert_array_equal(block, np.zeros_like(p))
    assert _bh_count(p, .05) == 2
    assert not ebh(block, .05).any()
    capped = by_calibrator(p, family=p.size, alpha=.05, cap=2)
    assert not ebh(.2 * capped + .8 * block, .05).any()


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("p", [
    [], np.empty((0, 2)), [0, 1], [[0]], [[0, 1, 0]], np.zeros((1, 2, 1)),
    [[0, 1], [0]], [[-1e-100, .5]], [[np.nextafter(1., 2.), 0]],
    [[np.nan, 0]], [[np.inf, 0]], [[-np.inf, 0]],
    np.ma.array([[.1, .2]], mask=[[False, True]]),
])
def test_invalid_shape_bounds_and_masks(implementation, p):
    with pytest.raises(ValueError):
        implementation(p)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("p", [
    [[True, False]], [[.1j, .2]], [[".1", ".2"]],
    np.array([[.1, .2]], dtype=object),
])
def test_invalid_p_dtypes(implementation, p):
    with pytest.raises(TypeError):
        implementation(p)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("alpha", [0, 1, -1, np.nan, np.inf, -np.inf])
def test_invalid_alpha_range(implementation, alpha):
    with pytest.raises(ValueError):
        implementation([[.1, .2]], alpha)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
@pytest.mark.parametrize("alpha", [True, False, [.05], ".05", .05 + 0j, None])
def test_invalid_alpha_type(implementation, alpha):
    with pytest.raises(TypeError):
        implementation([[.1, .2]], alpha)


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_numeric_lists_endpoints_and_integer_probabilities(implementation):
    e, diagnostic = implementation([[0, 1]], np.float64(.05))
    np.testing.assert_array_equal(e, [[20, 0]])
    np.testing.assert_array_equal(diagnostic["R"], [2])
    for value, expected_r, expected_e in [(0, 8, 20), (1, 2, 0)]:
        e, diagnostic = implementation(np.full((4, 2), value))
        np.testing.assert_array_equal(diagnostic["R"], np.full(4, expected_r))
        np.testing.assert_array_equal(e, np.full((4, 2), expected_e))


@pytest.mark.parametrize("implementation", IMPLEMENTATIONS)
def test_tiny_safe_thresholds_and_rejection_of_overflow_without_warnings(implementation):
    smallest_positive = np.nextafter(0., 1.)
    with np.errstate(all="raise"):
        e, diagnostic = implementation([[0, 1]], 1e-308)
        assert diagnostic["tau"][0] == 1e-308
        assert np.isfinite(e).all()
        assert e[0, 0] == 1 / 1e-308
        for alpha, n in [(1e-308, 2), (smallest_positive, 1)]:
            with pytest.raises(ValueError, match="too small"):
                implementation(np.ones((n, 2)), alpha)
        e, diagnostic = implementation([[0, 1]], np.nextafter(1., 0.))
        assert np.isfinite(e).all()
        assert 0 < diagnostic["tau"][0] < 1
