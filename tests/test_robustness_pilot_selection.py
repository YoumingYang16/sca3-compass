"""Bounded API/algorithm checks, not evidence of scientific power or FDR."""

import json
from copy import deepcopy
from time import perf_counter

import numpy as np
import pytest

from sca3_compass.molecular_methods import ebh
from sca3_compass.robustness_calibrators import by_calibrator, focused_calibrator
from sca3_compass.robustness_pilot import pilot_calibrator
from sca3_compass.robustness_pilot_selection import (
    CONCENTRATION_FRACTION,
    MULTIPLIERS,
    apply_pilot_multiplier,
    pilot_multiplier_calibrator,
    pilot_training_receipt,
    select_pilot_multiplier,
    selected_pilot_calibrator,
)


def _piecewise_cutpoints(info):
    cap, family, alpha = info["cap"], info["family"], info["pilot_level"]
    harmonic = np.sum(1. / np.arange(1, cap + 1))
    return np.unique(np.r_[0., alpha * np.arange(1, cap + 1) / (family * harmonic),
                           info["threshold"], 1.])


@pytest.mark.parametrize("multiplier", MULTIPLIERS)
@pytest.mark.parametrize("family", [2, 7, 16, 512])
@pytest.mark.parametrize("count", [0, 1, 7, 64])
def test_fixed_calibrator_exact_piecewise_integral_one(multiplier, family, count):
    training = np.ones((32, 2))
    training.flat[:count] = 1e-9
    receipt = pilot_training_receipt(training)
    _, info = apply_pilot_multiplier([1.], receipt, family, multiplier)
    cut = _piecewise_cutpoints(info)
    values, _ = apply_pilot_multiplier((cut[:-1] + cut[1:]) / 2, receipt, family, multiplier)
    assert abs(np.dot(np.diff(cut), values) - 1.) < 2e-14
    assert np.all(np.diff(values) <= 0)
    assert info["pilot_discoveries"] == count
    assert info["concentration_fraction"] == (CONCENTRATION_FRACTION if count else 0.)


@pytest.mark.parametrize("family", [2, 17, 128, 512])
@pytest.mark.parametrize("count", [0, 1, 18, 64])
@pytest.mark.parametrize("alpha", [.01, .05, .2])
def test_half_multiplier_bitwise_legacy_values_and_legacy_receipt(family, count, alpha):
    training = np.ones((32, 2))
    training.flat[:count] = 1e-8
    initial = pilot_training_receipt(training, alpha)
    _, info = apply_pilot_multiplier([1.], initial, family)
    cut = _piecewise_cutpoints(info)
    # Include exactly represented jumps and their immediate neighbors, not
    # just generic values far from inclusive <= / ceil boundaries.
    held = np.unique(np.r_[cut, np.maximum(0, np.nextafter(cut, -np.inf)),
                           np.minimum(1, np.nextafter(cut, np.inf))])
    old, old_info = pilot_calibrator(held, training, family, fraction=.8, alpha=alpha)
    new, new_info = pilot_multiplier_calibrator(held, training, family, alpha=alpha)
    np.testing.assert_array_equal(new, old)
    assert {key: new_info[key] for key in old_info} == old_info
    direct, direct_info = apply_pilot_multiplier(held, old_info, family)
    np.testing.assert_array_equal(direct, old)
    assert direct_info == new_info


@pytest.mark.parametrize("multiplier", MULTIPLIERS)
def test_zero_count_fallback_exactly_old_capped_by(multiplier):
    held = np.linspace(0, 1, 513)
    value, info = pilot_multiplier_calibrator(held, np.ones((8, 2)), 512, multiplier)
    np.testing.assert_array_equal(value, by_calibrator(held, 512, cap=64))
    assert info["zero_count_fallback"]
    assert info["threshold"] == multiplier * .05 / 16
    assert info["concentration_fraction"] == 0.


def test_pilot_uses_largest_passing_bh_rank_not_count_of_passing_comparisons():
    # Only ranks 3 and 4 pass; R_BH is 4, not 2.
    training = np.array([[.02, .03], [.03, .05]])
    receipt = pilot_training_receipt(training)
    assert receipt["pilot_discoveries"] == 4
    for multiplier in MULTIPLIERS:
        _, info = apply_pilot_multiplier([.02], receipt, 4, multiplier)
        assert info["threshold"] == multiplier * .05


def _reference_selector(training, alpha=.05):
    table = []
    sizes = [training[parity::2].size for parity in (0, 1)]
    for multiplier in MULTIPLIERS:
        folds = []
        for parity in (0, 1):
            train, validation = training[1-parity::2], training[parity::2]
            ordered = np.sort(train.ravel())
            passing = np.flatnonzero(ordered <= alpha * np.arange(1, train.size+1) / train.size)
            count = int(passing[-1] + 1) if len(passing) else 0
            threshold = multiplier * alpha * max(1, count) / train.size
            value = focused_calibrator(validation, validation.size, threshold,
                                       .8 if count else 0., alpha, cap=max(1, validation.size//8))
            # Literal full e-BH, independently implementing the rank criterion.
            flat = value.ravel()
            order = np.argsort(-flat, kind="stable")
            passing_e = np.flatnonzero(flat[order] >= flat.size / (alpha*np.arange(1, flat.size+1)))
            rejected = np.zeros(flat.size, bool)
            if len(passing_e):
                rejected[order[:passing_e[-1]+1]] = True
            folds.append((count, threshold, value, rejected.reshape(validation.shape)))
        numerator = int(folds[0][3].sum()) * sizes[1] + int(folds[1][3].sum()) * sizes[0]
        table.append((multiplier, numerator, folds))
    winner = min(table, key=lambda row: (-row[1], abs(row[0]-.5), row[0]))
    return winner[0], table


@pytest.mark.parametrize("genes", [2, 3, 5, 16, 127, 128])
def test_selector_literal_reference_actual_signed_families_and_exact_receipts(genes):
    rng = np.random.default_rng(49100 + genes)
    training = rng.uniform(size=(genes, 2))
    training[rng.uniform(size=training.shape) < .3] *= 1e-3
    selected, info = select_pilot_multiplier(training)
    reference, table = _reference_selector(training)
    assert selected == reference
    for candidate, (multiplier, numerator, folds) in zip(info["candidates"], table, strict=True):
        assert candidate["multiplier"] == multiplier
        assert candidate["utility_numerator"] == numerator
        assert candidate["utility_denominator"] == 2 * training[::2].size * training[1::2].size
        assert candidate["mean_discovery_proportion"] == numerator / candidate["utility_denominator"]
        for parity, (receipt, expected) in enumerate(zip(candidate["folds"], folds, strict=True)):
            count, threshold, values, rejection = expected
            assert receipt["training_bh_receipt"]["pilot_discoveries"] == count
            assert receipt["calibrator_receipt"]["threshold"] == threshold
            assert receipt["validation_signed_family"] == training[parity::2].size
            assert receipt["calibrator_receipt"]["family"] == training[parity::2].size
            assert receipt["training_row_indices"] == list(range(1-parity, genes, 2))
            assert receipt["validation_row_indices"] == list(range(parity, genes, 2))
            np.testing.assert_array_equal(receipt["validation_evalues"], values)
            np.testing.assert_array_equal(receipt["validation_rejections"], rejection)
            assert receipt["discoveries"] == int(rejection.sum())
    # Entire receipt is strict JSON, including all utility and fallback detail.
    assert json.loads(json.dumps(info, allow_nan=False)) == info


@pytest.mark.parametrize("p, expected", [(1., .5), (0., .5), (.007, .65), (.009, .5)])
def test_tie_rule_and_threshold_sensitivity_are_active(p, expected):
    half = np.ones((10, 2))
    half[:5, 0] = p
    training = np.repeat(half, 2, axis=0)
    selected, info = select_pilot_multiplier(training)
    assert selected == expected
    if p == 1.:
        assert info["all_inner_pilots_zero"] and info["all_utilities_zero"]
        assert info["zero_count_fallback"]
    if p == 0.:
        assert not info["all_inner_pilots_zero"]
        assert len({row["utility_numerator"] for row in info["candidates"]}) == 1
    if p == .009:
        # f=c=.8 puts evidence exactly at the mathematical e-BH boundary.
        # Legacy float64 operations yield 79.99999999999999, not 80 here.
        # Preserve this real numerical cliff; do not invent a tie tolerance.
        assert not info["all_inner_pilots_zero"] and info["all_utilities_zero"]
        row = info["candidates"][-1]["folds"][0]
        assert row["validation_evalues"][0][0] < 20 / (.05 * 5)


def test_higher_threshold_can_hit_evidence_cliff_and_quarter_can_win():
    # Fixed finite fixture, not a research simulation. Odd/even groups have
    # different BH counts. A higher tau admits p-values but dilutes their e.
    training = np.ones((20, 2))
    training[0:8:2, 0] = .004
    training[1::2, 0] = .02
    selected, info = select_pilot_multiplier(training)
    reference, table = _reference_selector(training)
    assert selected == reference == .25
    scores = {row[0]: row[1] for row in table}
    assert scores[.25] > scores[.5]
    assert info["selected_multiplier"] == .25


def test_point_eight_wins_without_relying_on_a_floating_equality():
    training = np.ones((20, 2))
    training[0:12:2, 0] = .009
    training[1:11:2, 0] = .009
    selected, info = select_pilot_multiplier(training)
    assert selected == _reference_selector(training)[0] == .8
    counts = [[fold["discoveries"] for fold in row["folds"]] for row in info["candidates"]]
    assert counts == [[0, 0], [0, 0], [0, 5], [6, 0]]


def test_held_values_and_outer_family_cannot_change_selection_or_refit():
    training = np.repeat(np.column_stack((np.r_[np.full(5, .007), np.ones(5)], np.ones(10))), 2, axis=0)
    first, info1 = selected_pilot_calibrator(np.array([[0., 1.], [.0002, .2]]), training, 64)
    second, info2 = selected_pilot_calibrator(np.array([.01, .9, .5]), training, 512)
    assert first.shape == (2, 2) and second.shape == (3,)
    assert info1["selection"] == info2["selection"]
    selection = info1["selection"]
    all_receipt = pilot_training_receipt(training)
    assert selection["all_training_bh_receipt"] == all_receipt
    assert selection["refit_calibrator_receipt"]["threshold"] == .65 * .05 * 10 / 40
    assert info1["application"]["threshold"] == selection["refit_calibrator_receipt"]["threshold"]
    assert not selection["uses_held_probabilities_for_selection"]
    assert not selection["uses_truth_labels"]
    assert not selection["nuisances_refitted_inside_inner_split"]
    assert not selection["fitted_pvalue_fdr_proven"]


def test_selector_finishes_before_held_array_is_accessed(monkeypatch):
    import sca3_compass.robustness_pilot_selection as module
    original = module.select_pilot_multiplier
    completed = []

    def traced_select(training, alpha):
        result = original(training, alpha)
        completed.append(result[0])
        return result

    class HeldProbe:
        def __array__(self, dtype=None, copy=None):
            assert len(completed) == 1
            return np.asarray([[.01, .9]], dtype=dtype)

    monkeypatch.setattr(module, "select_pilot_multiplier", traced_select)
    selected_pilot_calibrator(HeldProbe(), np.ones((4, 2)), 8)
    assert completed == [.5]


def test_reused_receipt_does_not_recompute_bh_or_use_a_new_training_family(monkeypatch):
    import sca3_compass.robustness_pilot_selection as module
    receipt = pilot_training_receipt([[0., .8], [.0001, 1.]])

    def forbidden(*args, **kwargs):
        raise AssertionError("Fixed receipt application must not rerun training BH")

    monkeypatch.setattr(module, "pilot_calibrator", forbidden)
    for multiplier in MULTIPLIERS:
        _, info = apply_pilot_multiplier([.001], receipt, 512, multiplier)
        assert info["training_signed_count"] == 4
        assert info["pilot_discoveries"] == 2


def test_fixed_pilot_is_order_invariant_and_selector_preserves_two_signs():
    rng = np.random.default_rng(7719)
    training = rng.uniform(size=(9, 2))
    training[:4, 0] *= .001
    held = np.array([[0., .7], [.02, .0001]])
    for multiplier in MULTIPLIERS:
        first, receipt1 = pilot_multiplier_calibrator(held, training, 32, multiplier)
        other, receipt2 = pilot_multiplier_calibrator(held, training[::-1, ::-1], 32, multiplier)
        np.testing.assert_array_equal(first, other)
        assert receipt1 == receipt2
    selected, info = select_pilot_multiplier(training)
    swapped, other = select_pilot_multiplier(training[:, ::-1])
    assert selected == swapped
    assert [row["utility_numerator"] for row in info["candidates"]] == [
        row["utility_numerator"] for row in other["candidates"]
    ]


def test_input_arrays_and_reused_receipt_are_never_mutated():
    training = np.array([[.0001, .9], [.02, 1.], [.03, .5]])
    held = np.array([[.01, .8]])
    before_training, before_held = training.copy(), held.copy()
    receipt = pilot_training_receipt(training)
    before_receipt = deepcopy(receipt)
    training.flags.writeable = held.flags.writeable = False
    for multiplier in MULTIPLIERS:
        apply_pilot_multiplier(held, receipt, 8, multiplier)
    selected_pilot_calibrator(held, training, 8)
    np.testing.assert_array_equal(training, before_training)
    np.testing.assert_array_equal(held, before_held)
    assert receipt == before_receipt


def test_one_inner_zero_pilot_preserves_both_fallback_receipts():
    training = np.ones((6, 2))
    training[::2, 0] = 0.
    _, info = select_pilot_multiplier(training)
    assert not info["all_inner_pilots_zero"]
    for row in info["candidates"]:
        assert row["folds"][0]["calibrator_receipt"]["zero_count_fallback"]
        assert not row["folds"][1]["calibrator_receipt"]["zero_count_fallback"]


@pytest.mark.parametrize("training", [np.ones((1, 2)), np.ones(4), np.ones((4, 1)),
                                      np.ones((4, 3)), np.ones((2, 2, 2))])
def test_selector_rejects_invalid_split_shape(training):
    with pytest.raises(ValueError, match="shape"):
        select_pilot_multiplier(training)


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [-.01], [1.01],
                                 [True], [".5"], [.2+0j], np.ma.array([.5], mask=[True])])
def test_invalid_probabilities_fail_closed(bad):
    with pytest.raises((ValueError, TypeError)):
        pilot_training_receipt(bad)
    with pytest.raises((ValueError, TypeError)):
        pilot_multiplier_calibrator(bad, [[.01, 1.]], 2)


@pytest.mark.parametrize("alpha", [0., 1., -.1, np.nan, np.inf, True, [.05], "0.05", 1e-320])
def test_invalid_or_numerically_unsafe_levels(alpha):
    with pytest.raises((ValueError, TypeError)):
        select_pilot_multiplier(np.ones((4, 2)), alpha)


@pytest.mark.parametrize("multiplier", [0., .3, 1., np.nan, True, [.5], ".5"])
def test_only_prespecified_multipliers(multiplier):
    with pytest.raises((ValueError, TypeError)):
        pilot_multiplier_calibrator([.01], [0., 1.], 2, multiplier)


@pytest.mark.parametrize("family", [0, 1, 2.5, True, "8"])
def test_invalid_family(family):
    with pytest.raises((ValueError, TypeError)):
        pilot_multiplier_calibrator([.01], [0., 1.], family)


@pytest.mark.parametrize("update", [
    {"training_signed_count": 0}, {"training_signed_count": 2.5},
    {"pilot_discoveries": -1}, {"pilot_discoveries": 3},
    {"pilot_discoveries": True}, {"pilot_level": 0},
    {"uses_truth_labels": True}, {"uses_held_probabilities_for_selection": True},
    {"zero_count_fallback": True}, {"zero_count_fallback": "false"},
])
def test_inconsistent_or_disallowed_receipts_rejected(update):
    receipt = pilot_training_receipt([0., 1.])
    receipt.update(update)
    with pytest.raises((ValueError, TypeError)):
        apply_pilot_multiplier([.01], receipt, 2)


def test_receipt_is_required_and_threshold_is_recomputed_from_counts():
    with pytest.raises(TypeError):
        apply_pilot_multiplier([.01], None, 2)
    with pytest.raises(ValueError):
        apply_pilot_multiplier([.01], {}, 2)
    receipt = pilot_training_receipt([0., 1.])
    receipt["threshold"] = .99
    receipt["concentration_fraction"] = .123
    _, info = apply_pilot_multiplier([.01], receipt, 2, .65)
    assert info["threshold"] == .65 * .05 / 2
    assert info["concentration_fraction"] == .8


def test_selected_rule_piecewise_integral_one_after_training_selection():
    training = np.ones((20, 2))
    training[0:12:2, 0] = .009
    training[1:11:2, 0] = .009
    _, info = selected_pilot_calibrator([1.], training, 512)
    assert info["selection"]["selected_multiplier"] == .8
    cut = _piecewise_cutpoints(info["application"])
    values, _ = selected_pilot_calibrator((cut[:-1] + cut[1:])/2, training, 512)
    assert abs(np.dot(np.diff(cut), values) - 1.) < 2e-14


def test_bounded_performance_smoke(capsys):
    # Eight small e-BH applications, no model fit/search, no stochastic timing
    # assertion. Timing is reported, not made into a hardware-dependent test.
    rng = np.random.default_rng(492031)
    training = rng.uniform(size=(128, 2))
    training[:32, 0] *= .001
    started = perf_counter()
    for _ in range(10):
        _, info = select_pilot_multiplier(training)
    elapsed = perf_counter() - started
    assert len(info["candidates"]) == 4
    assert sum(len(row["folds"]) for row in info["candidates"]) == 8
    for row in info["candidates"]:
        for fold in row["folds"]:
            np.testing.assert_array_equal(
                ebh(np.asarray(fold["validation_evalues"]), .05), fold["validation_rejections"],
            )
    assert elapsed > 0
    with capsys.disabled():
        print(f"\npilot_selector N=128 mean_seconds={elapsed / 10:.6f}")
