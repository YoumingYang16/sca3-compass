"""Bounded correctness/provenance checks, NOT research success evidence."""

import json
import math
from copy import deepcopy
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

import sca3_compass.robustness_rank_budget as module
from sca3_compass.molecular_methods import ebh
from sca3_compass.robustness_calibrators import by_calibrator, focused_calibrator
from sca3_compass.robustness_pilot import pilot_calibrator
from sca3_compass.robustness_rank_budget import (
    apply_rank_budget,
    categorical_count_law,
    fit_rank_budget,
    rank_budget_calibrator,
    rank_threshold,
)


def _enumerate(q, genes):
    law = [Fraction(0) for _ in range(2 * genes + 1)]
    for counts in product(range(3), repeat=genes):
        law[sum(counts)] += math.prod(q[count] for count in counts)
    return np.asarray([float(x) for x in law])


@pytest.mark.parametrize("genes", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("q", [
    (Fraction(1, 3),) * 3,
    (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)),
    (Fraction(3, 5), Fraction(0), Fraction(2, 5)),
])
def test_polynomial_matches_exact_rational_enumeration(q, genes):
    law, info = categorical_count_law([float(x) for x in q], genes)
    np.testing.assert_allclose(law, _enumerate(q, genes), rtol=3e-15, atol=1e-17)
    assert len(law) == 2 * genes + 1
    assert np.all(law >= 0)
    assert abs(math.fsum(law.tolist()) - 1) < 3e-16
    assert info["negative_coefficient_count"] == 0
    assert not info["uses_fft"]
    assert not info["negative_coefficients_clipped"]


@pytest.mark.parametrize("genes", [1, 2, 7, 64, 256])
@pytest.mark.parametrize("q", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]])
def test_structural_zeros_are_not_falsely_called_underflow(genes, q):
    law, info = categorical_count_law(q, genes)
    assert not info["underflow_detected"]
    assert info["zero_coefficients_on_mathematical_support"] == 0
    assert int(np.sum(law == 0)) == info["structural_zero_count"]
    assert info["mass"] == 1.
    assert abs(info["mean"] - genes * (q[1] + 2 * q[2])) < 1e-12


def test_underflow_and_subnormal_coefficients_are_disclosed_not_clipped():
    law, info = categorical_count_law([1 - 1e-10, 5e-11, 5e-11], 256)
    assert info["underflow_detected"]
    assert info["zero_coefficients_on_mathematical_support"] == np.sum(law == 0) > 0
    assert info["structural_zero_count"] == 0
    assert info["subnormal_positive_coefficients"] > 0
    assert info["intermediate_underflow_zero_count_max"] > 0
    assert abs(info["mass"] - 1.) < 3e-16
    assert info["absolute_mean_error"] < 1e-18


def test_input_normalization_is_explicit_and_large_mass_errors_are_rejected():
    q = np.array([.2, .3, .5 + 2e-16])
    law, info = categorical_count_law(q, 8)
    assert info["q_input_mass"] == math.fsum(q.tolist())
    np.testing.assert_array_equal(info["q_normalized"], q / math.fsum(q.tolist()))
    assert np.all(law >= 0)
    for operation in info["operations"]:
        assert operation["normalization_divisor"] == operation["mass_before_normalization"]
        assert abs(operation["mass_after_normalization"] - 1.) < 3e-16
    with pytest.raises(ValueError, match="sum"):
        categorical_count_law([.2, .3, .5001], 8)


@pytest.mark.parametrize("bad_result", [np.array([-1., 2., 0.]), np.array([np.nan]), np.array([0., 0., 0.]), np.array([.2, .2, .2])])
def test_convolution_errors_are_not_silently_repaired(monkeypatch, bad_result):
    monkeypatch.setattr(module.np, "convolve", lambda left, right: bad_result)
    with pytest.raises(ArithmeticError):
        categorical_count_law([.2, .3, .5], 1)


@pytest.mark.parametrize("genes", [1, 2, 3, 4, 8, 17, 64, 128, 256, 512])
@pytest.mark.parametrize("alpha", [.001, .05, .2, .99])
def test_every_rank_has_safe_actual_step_height_and_downward_only_boundary(genes, alpha):
    previous = 0.
    for rank in range(1, max(1, genes // 4) + 1):
        tau, info = rank_threshold(rank, genes, alpha)
        nominal = alpha * .8 * rank / (2 * genes)
        exact_bound = Fraction.from_float(alpha) * Fraction.from_float(.8) * rank / (2 * genes)
        assert 0 < previous < tau if previous else 0 < tau
        previous = tau
        assert tau <= nominal
        assert Fraction.from_float(tau) <= exact_bound
        assert .8 / tau >= (2 * genes) / (alpha * rank)
        assert info["step_evidence_height"] == .8 / tau
        assert info["ebh_rank_cutoff"] == (2 * genes) / (alpha * rank)
        corrected = nominal
        for _ in range(info["ulp_downward_corrections"]):
            corrected = np.nextafter(corrected, 0.)
        assert tau == corrected
        assert float.fromhex(info["threshold_hex"]) == tau
        # Boundary test uses the REAL calibrator and the REAL eBH, not a
        # tolerance comparison. All remaining 2G-r claims stay in the family.
        p = np.ones(2 * genes)
        p[:rank] = tau
        values = focused_calibrator(p, 2 * genes, tau, .8, alpha, cap=max(1, genes // 4))
        assert np.all(values[:rank] >= 2 * genes / (alpha * rank))
        assert ebh(values, alpha).sum() >= rank


def test_retained_one_ulp_height_cliff_is_corrected_before_any_observation():
    tau, info = rank_threshold(1, 4)
    assert info["nominal_threshold"] == .005000000000000001
    assert .8 / info["nominal_threshold"] < 8 / .05
    assert info["ulp_downward_corrections"] == 1
    assert tau == .005
    assert .8 / tau >= 8 / .05


def _reference_table(training, genes, alpha):
    table = []
    for rank in range(1, max(1, genes // 4) + 1):
        tau, _ = rank_threshold(rank, genes, alpha)
        counts = [sum(sum(row <= tau) == k for row in training) for k in range(3)]
        # Independent sequential polynomial, not exponentiation or a cache.
        q = [(x + .5) / (len(training) + 1.5) for x in counts]
        law = np.ones(1)
        for _ in range(genes):
            extended = np.zeros(len(law) + 2)
            for k in range(3):
                extended[k:k+len(law)] += q[k] * law
            law = extended
        table.append((rank, counts, q, sum(law[rank:]), sum(k * law[k] for k in range(rank, len(law)))))
    return table


@pytest.mark.parametrize("genes,n", [(1, 8), (4, 1), (12, 9), (32, 12), (64, 15), (256, 128)])
def test_full_utility_table_matches_independent_reference_and_uses_actual_G(genes, n):
    rng = np.random.default_rng(92831 + genes + n)
    training = rng.uniform(size=(n, 2))
    training[:max(1, n // 3), 0] *= .004
    training[0, 0] = 0.
    info = fit_rank_budget(training, genes)
    expected = _reference_table(training, genes, .05)
    assert info["family"] == 2 * genes != 2 * n
    assert info["training_gene_count"] == n
    assert info["training_signed_count"] == 2 * n
    assert info["cap"] == max(1, genes // 4)
    assert len(info["candidates"]) == info["cap"]
    for row, (rank, counts, q, tail, utility) in zip(info["candidates"], expected, strict=True):
        assert row["rank"] == rank
        assert row["category_counts"] == counts
        np.testing.assert_array_equal(row["q"], q)
        assert abs(row["tail_probability"] - tail) < 2e-14
        assert abs(row["utility"] - utility) < 2e-11
        assert row["utility"] == row["expected_count_above_rank"]
        assert abs(row["mean_count"] - genes * (q[1] + 2*q[2])) < 3e-12
    winner = min(info["candidates"], key=lambda row: (-row["utility"], row["rank"]))
    assert info["selected_rank"] == winner["rank"]
    assert info["threshold"] == winner["threshold"]
    assert info["concentration_fraction"] == .8


def test_exact_computed_ties_choose_lower_rank_and_identical_laws_are_cached():
    # Every count is two; near-certain large counts produce exact float64
    # utility ties for the first ranks. This is not an exact-real tie claim.
    info = fit_rank_budget(np.zeros((128, 2)), 256)
    assert info["candidates"][0]["utility"] == info["candidates"][1]["utility"]
    assert info["selected_rank"] == 1
    assert info["distinct_count_laws_evaluated"] == 1
    assert not info["candidates"][0]["law_reused_for_identical_category_counts"]
    assert all(row["law_reused_for_identical_category_counts"] for row in info["candidates"][1:])
    info["candidates"][0]["law_diagnostics"]["mass"] = -999
    assert info["candidates"][1]["law_diagnostics"]["mass"] > 0


def _cuts(info):
    family, alpha, cap = info["family"], info["alpha"], info["cap"]
    h = np.sum(1. / np.arange(1, cap + 1))
    return np.unique(np.r_[0., alpha * np.arange(1, cap+1)/(family*h), info["threshold"], 1.])


@pytest.mark.parametrize("genes", [1, 3, 8, 17, 64, 256])
@pytest.mark.parametrize("pilot", [False, True])
@pytest.mark.parametrize("alpha", [.01, .05, .3])
def test_fitted_calibrator_is_monotone_and_integral_one_at_actual_breakpoints(genes, pilot, alpha):
    training = np.ones((12, 2))
    if pilot:
        training[:4, 0] = 1e-6
    fit = fit_rank_budget(training, genes, alpha)
    cuts = _cuts(fit)
    mid = (cuts[:-1] + cuts[1:]) / 2
    values, _ = apply_rank_budget(mid, fit)
    assert abs(np.dot(np.diff(cuts), values) - 1.) < 3e-14
    assert np.all(values >= 0)
    assert np.all(np.diff(values) <= 0)
    neighbors = np.unique(np.r_[cuts, np.maximum(0, np.nextafter(cuts, -np.inf)), np.minimum(1, np.nextafter(cuts, np.inf))])
    boundary_values, _ = apply_rank_budget(neighbors, fit)
    assert np.all(np.diff(boundary_values) <= 0)


@pytest.mark.parametrize("genes", [1, 7, 32, 256])
@pytest.mark.parametrize("n", [1, 8, 129])
def test_zero_BH_pilot_exact_legacy_fallback_even_when_smoothed_utility_positive(genes, n):
    training = np.ones((n, 2))
    fit = fit_rank_budget(training, genes)
    held = _cuts(fit)
    old, old_info = pilot_calibrator(held, training, 2 * genes, fraction=.8)
    values, application = apply_rank_budget(held, fit)
    np.testing.assert_array_equal(values, old)
    np.testing.assert_array_equal(values, by_calibrator(held, 2 * genes, cap=max(1, genes // 4)))
    assert fit["threshold"] == old_info["threshold"]
    assert fit["selected_rank"] is None
    assert fit["selected_utility"] is None
    assert fit["concentration_fraction"] == 0.
    assert fit["zero_count_fallback"] and application["zero_count_fallback"]
    assert all(row["utility"] > 0 for row in fit["candidates"])


def test_BH_uses_largest_passing_rank_not_number_of_passing_inequalities():
    fit = fit_rank_budget([[.02, .03], [.03, .05]], 4)
    assert fit["pilot_discoveries"] == 4


def test_retained_smoothing_only_threshold_issue_when_BH_pilot_nonzero():
    # BH rejects all .04 values, but no allowed tau (<=.005 for G=256)
    # reaches them. The SPECIFIED zero-BH veto does not cover this case.
    fit = fit_rank_budget(np.full((128, 2), .04), 256)
    assert fit["pilot_discoveries"] == 256
    assert not fit["zero_count_fallback"]
    assert fit["selected_rank"] == 1
    assert fit["selected_has_no_training_threshold_exceedances"]
    assert fit["all_rank_candidates_have_zero_training_threshold_exceedances"]
    assert all(row["category_counts"] == [128, 0, 0] for row in fit["candidates"])
    assert fit["selected_utility"] > 0


def test_counts_preserve_within_gene_two_sign_dependence_not_only_marginals():
    together = np.ones((8, 2))
    together[:4] = 0.
    apart = np.ones((8, 2))
    apart[:4, 0] = 0.
    apart[4:, 1] = 0.
    assert np.sum(together == 0) == np.sum(apart == 0)
    a, b = fit_rank_budget(together, 16), fit_rank_budget(apart, 16)
    assert a["candidates"][0]["category_counts"] == [4, 0, 4]
    assert b["candidates"][0]["category_counts"] == [0, 8, 0]
    assert a["candidates"][0]["q"] != b["candidates"][0]["q"]
    assert a["candidates"][0]["tail_probability"] != b["candidates"][0]["tail_probability"]


def test_determinism_provenance_serialization_permutation_and_no_input_mutation():
    training = np.array([[0., 1.], [.001, .5], [.05, .02], [1., .01]])
    before = training.copy()
    fit = fit_rank_budget(training, np.int64(32), np.float64(.05))
    saved = deepcopy(fit)
    held = np.array([[0., .05], [.001, 1.]])
    held_before = held.copy()
    a, ai = apply_rank_budget(held, fit)
    b, bi = apply_rank_budget(np.ones((17, 2)), fit)
    assert a.shape == held.shape and b.shape == (17, 2)
    assert ai == bi
    assert fit == saved == fit_rank_budget(training.copy(), 32)
    assert json.loads(json.dumps(fit, allow_nan=False)) == fit
    json.dumps(ai, allow_nan=False)
    np.testing.assert_array_equal(training, before)
    np.testing.assert_array_equal(held, held_before)
    permuted = fit_rank_budget(training[::-1, ::-1], 32)
    assert permuted["training_sha256"] != fit["training_sha256"]
    assert permuted["candidates"] == fit["candidates"]
    assert permuted["selected_rank"] == fit["selected_rank"]
    assert not fit["uses_held_probabilities_for_selection"]
    assert not fit["uses_truth_labels"]
    assert not fit["fitted_pvalue_fdr_proven"]
    assert not fit["originality_claimed"]


def test_wrapper_finishes_selection_before_held_conversion(monkeypatch):
    finished = False
    original = module.fit_rank_budget

    def record_fit(*args, **kwargs):
        nonlocal finished
        result = original(*args, **kwargs)
        finished = True
        return result

    class HeldProbe:
        def __array__(self, dtype=None, copy=None):
            assert finished
            return np.asarray([0., .5], dtype=dtype)

    monkeypatch.setattr(module, "fit_rank_budget", record_fit)
    _, first = rank_budget_calibrator(HeldProbe(), [[0., 1.]], 32)
    _, second = rank_budget_calibrator(np.ones(100), [[0., 1.]], 32)
    assert first == second


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [-.001], [1.001], [True], [".1"], [.1+0j], np.ma.array([.2], mask=[True])])
def test_invalid_probabilities_fail_closed(bad):
    with pytest.raises((TypeError, ValueError)):
        fit_rank_budget(bad, 8)
    with pytest.raises((TypeError, ValueError)):
        apply_rank_budget(bad, fit_rank_budget([[0., 1.]], 8))


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "8", None, 2**60])
def test_invalid_family_gene_count(bad):
    with pytest.raises((TypeError, ValueError)):
        fit_rank_budget([[0., 1.]], bad)
    with pytest.raises((TypeError, ValueError)):
        categorical_count_law([.5, .5, 0.], bad)


@pytest.mark.parametrize("bad", [0., 1., -.1, np.nan, np.inf, True, [.05], "0.05", 1e-320])
def test_invalid_or_unsafe_alpha(bad):
    with pytest.raises((TypeError, ValueError)):
        fit_rank_budget([[0., 1.]], 8, bad)


@pytest.mark.parametrize("bad", [0, -1, 3, 1.0, True, "1"])
def test_invalid_rank(bad):
    with pytest.raises((TypeError, ValueError)):
        rank_threshold(bad, 8)


@pytest.mark.parametrize("bad", [np.ones(2), np.ones((2, 1)), np.ones((3, 3)), np.ones((2, 2, 2)), np.empty((0, 2))])
def test_invalid_training_shapes(bad):
    with pytest.raises(ValueError):
        fit_rank_budget(bad, 8)


@pytest.mark.parametrize("bad", [[0, 0, 0], [1, 1, 1], [.5, .5], [[.2, .3, .5]], [np.nan, 0, 1], [-.1, .1, 1]])
def test_invalid_categorical_probabilities(bad):
    with pytest.raises(ValueError):
        categorical_count_law(bad, 8)


@pytest.mark.parametrize("update", [
    {"version": "unknown"}, {"family": 32}, {"cap": 8},
    {"training_signed_count": 9}, {"training_gene_count": 0},
    {"pilot_discoveries": 3}, {"pilot_discoveries": True}, {"pilot_discoveries": -1},
    {"uses_truth_labels": True}, {"uses_held_probabilities_for_selection": True},
    {"zero_count_fallback": True}, {"zero_count_fallback": "false"},
    {"threshold": .99}, {"threshold": [0.001]}, {"concentration_fraction": .5},
    {"selected_rank": 0}, {"selected_rank": 3}, {"selected_rank": True},
    {"pilot_level": .1}, {"family_gene_count": True},
])
def test_malformed_or_inconsistent_fitted_receipts_fail_closed(update):
    fit = fit_rank_budget([[0., 1.]], 8)
    fit.update(update)
    with pytest.raises((TypeError, ValueError)):
        apply_rank_budget([.001], fit)


def test_incomplete_receipts_and_fallback_with_selected_rank_are_rejected():
    with pytest.raises(TypeError):
        apply_rank_budget([.1], None)
    with pytest.raises(ValueError):
        apply_rank_budget([.1], {})
    fit = fit_rank_budget([[1., 1.]], 8)
    fit["selected_rank"] = 1
    with pytest.raises(ValueError):
        apply_rank_budget([.1], fit)
