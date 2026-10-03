import inspect
import json
import time
from itertools import product

import numpy as np
import pytest

from sca3_compass import robustness_directional_budget as budget
from sca3_compass import robustness_predictive_gate as gate
from sca3_compass.molecular_methods import ebh
from sca3_compass.robustness_patterns import pattern_library
from sca3_compass.robustness_pilot import pilot_calibrator


def _reference_reject(e, alpha):
    # Threshold-value formulation, independent of the deployed argsort code.
    descending = np.sort(e.ravel())[::-1]
    passing = [r for r in range(1, e.size + 1)
               if descending[r - 1] >= e.size / (alpha * r)]
    return e >= descending[max(passing) - 1] if passing else np.zeros(e.shape, bool)


@pytest.mark.parametrize("shape", [(1, 1, 2), (1, 7, 2), (4, 9, 2), (16, 256, 2)])
def test_every_score_matches_independent_family_reference(shape):
    rng = np.random.default_rng(9015)
    evidence = rng.choice([0., 1., 20., 40., 80., 120., 400.], size=shape)
    truth = rng.random(shape) < .4
    original_e, original_truth = evidence.copy(), truth.copy()
    weights, receipt = budget.directional_budget(evidence, truth)
    assert len(receipt["utility_table"]) == 9
    for index, row in enumerate(receipt["utility_table"]):
        current = np.array(row["weights"])
        rejection = np.stack([_reference_reject(f * current, .05) for f in evidence])
        tp = (rejection & truth).sum((1, 2))
        fp = (rejection & ~truth).sum((1, 2))
        np.testing.assert_array_equal(row["true_positives_by_family"], tp)
        np.testing.assert_array_equal(row["false_positives_by_family"], fp)
        np.testing.assert_array_equal(row["rejections_by_family"], tp + fp)
        np.testing.assert_array_equal(row["true_positives_by_sign"], (rejection & truth).sum((0, 1)))
        np.testing.assert_array_equal(row["false_positives_by_sign"], (rejection & ~truth).sum((0, 1)))
        fdp = fp / np.maximum(1, tp + fp)
        np.testing.assert_array_equal(row["simulated_fdp_by_family"], fdp)
        assert row["mean_simulated_fdp"] == fdp.mean()
        assert row["grid_index"] == index
        assert row["pooled_true_positive_rate"] == tp.sum() / max(1, truth.sum())
        assert current.sum() == 2.
        if shape[0] > 1:
            assert row["descriptive_mc_se_mean_fdp"] == np.std(fdp, ddof=1) / np.sqrt(shape[0])
            t = truth.sum((1, 2))
            expected_se = np.std(tp - row["pooled_true_positive_rate"] * t, ddof=1) / np.sqrt(shape[0]) / t.mean()
            assert row["descriptive_mc_se_pooled_power"] == expected_se
        else:
            assert row["descriptive_mc_se_mean_fdp"] is None
            assert row["descriptive_mc_se_pooled_power"] is None
    expected = min(receipt["utility_table"], key=lambda r: (
        -r["total_true_positives"], abs(r["weights"][0] - 1), r["weights"][0]))
    assert receipt["selected_grid_index"] == expected["grid_index"]
    np.testing.assert_array_equal(weights, expected["weights"])
    np.testing.assert_array_equal(evidence, original_e)
    np.testing.assert_array_equal(truth, original_truth)
    assert receipt["selected_minus_equal_total_true_positives"] >= 0
    equal = receipt["utility_table"][receipt["equal_weight_grid_index"]]
    assert receipt["equal_weight_total_true_positives"] == equal["total_true_positives"]
    assert receipt["selected_total_true_positives"] == expected["total_true_positives"]
    assert receipt["selected_utility"] == expected["pooled_true_positive_rate"]
    assert receipt["selected_minus_equal_tp_by_sign"] == [
        b - e for b, e in zip(expected["true_positives_by_sign"], equal["true_positives_by_sign"])]
    json.dumps(receipt, allow_nan=False)


def test_mechanism_can_help_weak_but_concentrated_direction():
    evidence = np.zeros((1, 4, 2))
    evidence[0, :2, 0] = 60.
    truth = evidence > 0
    weights, receipt = budget.directional_budget(evidence, truth)
    # m=8, two-rejection threshold=80. The first nearest-equal passing
    # grid allocation is 1.5 x 60=90, with opposite weight .5.
    np.testing.assert_array_equal(weights, [1.5, .5])
    assert receipt["equal_weight_total_true_positives"] == 0
    assert receipt["selected_total_true_positives"] == 2
    assert not receipt["selected_equal_weights"]
    # Sign reversal changes only the semantic direction, not its opportunity.
    reversed_weights, reversed_receipt = budget.directional_budget(evidence[..., ::-1], truth[..., ::-1])
    np.testing.assert_array_equal(reversed_weights, weights[::-1])
    assert reversed_receipt["selected_total_true_positives"] == 2


def test_no_truth_selects_equal_even_when_it_has_more_false_discoveries():
    evidence = np.zeros((2, 4, 2))
    evidence[:, :2, 0] = 1e3
    truth = np.zeros_like(evidence, dtype=bool)
    weights, receipt = budget.directional_budget(evidence, truth)
    np.testing.assert_array_equal(weights, [1., 1.])
    assert receipt["total_simulated_true_claims"] == 0
    assert receipt["tied_best_grid_points"] == 9
    assert receipt["selected_descriptive_mc_se"] is None
    assert receipt["utility_table"][0]["total_false_positives"] == 0
    assert receipt["utility_table"][4]["total_false_positives"] == 4
    assert all(r["pooled_true_positive_rate"] == 0 for r in receipt["utility_table"])


def test_equal_tie_and_both_signs_true_are_allowed():
    evidence = np.full((2, 4, 2), 1e6)
    truth = np.ones_like(evidence, bool)
    weights, receipt = budget.directional_budget(evidence, truth)
    np.testing.assert_array_equal(weights, [1., 1.])
    assert receipt["selected_equal_weights"]
    assert receipt["total_simulated_true_claims"] == 16
    assert receipt["selected_total_true_positives"] == 16
    assert receipt["utility_table"][0]["total_true_positives"] == 8
    assert receipt["utility_table"][-1]["total_true_positives"] == 8


def test_each_family_not_pooled_across_simulations_and_zero_claims_retained():
    evidence = np.zeros((2, 4, 2))
    evidence[0, 0, 0] = 160.
    truth = evidence > 0
    _, receipt = budget.directional_budget(evidence, truth, positive_weight_grid=(1.,))
    row = receipt["utility_table"][0]
    assert row["true_positives_by_family"] == [1, 0]  # local m=8 threshold=160
    assert not ebh(evidence, .05).any()  # incorrect pooled m=16 threshold=320
    assert receipt["signed_family_size"] == 8
    assert receipt["zero_budget_claims_retained_in_family"]
    # At an endpoint it is still m=8, not the 4 nonzero-budget claims.
    smaller = evidence.copy()
    smaller[0, 0, 0] = 60.
    _, out = budget.directional_budget(smaller, truth)
    assert out["utility_table"][-1]["total_true_positives"] == 0


def test_exact_threshold_inclusion_and_nextafter():
    evidence = np.zeros((1, 4, 2))
    evidence[0, 0, 0] = 160.
    truth = evidence > 0
    _, exact = budget.directional_budget(evidence, truth, positive_weight_grid=(1.,))
    evidence[0, 0, 0] = np.nextafter(160., 0.)
    _, below = budget.directional_budget(evidence, truth, positive_weight_grid=(1.,))
    assert exact["selected_total_true_positives"] == 1
    assert below["selected_total_true_positives"] == 0


def test_determinism_hashes_no_input_mutation_and_layout():
    rng = np.random.default_rng(39)
    evidence = rng.exponential(100., (4, 11, 2))
    truth = rng.random(evidence.shape) < .4
    evidence.setflags(write=False)
    truth.setflags(write=False)
    a = budget.directional_budget(evidence, truth)
    b = budget.directional_budget(np.asfortranarray(evidence), np.asfortranarray(truth))
    np.testing.assert_array_equal(a[0], b[0])
    assert a[1] == b[1]
    a[0][:] = 0  # returned weights do not alias receipt or input
    assert sum(a[1]["selected_weights"]) == 2
    changed = np.array(evidence)
    changed[0, 0, 0] += 1
    c = budget.directional_budget(changed, truth)[1]
    assert c["simulated_e_sha256"] != b[1]["simulated_e_sha256"]
    assert c["simulated_truth_sha256"] == b[1]["simulated_truth_sha256"]
    assert c["inputs_sha256"] != b[1]["inputs_sha256"]


def test_training_family_and_gene_permutations_preserve_selection_and_scores():
    rng = np.random.default_rng(134)
    evidence = rng.choice([0., 20., 100., 800.], size=(4, 11, 2))
    truth = rng.random(evidence.shape) < .3
    w, a = budget.directional_budget(evidence, truth)
    order = [3, 1, 0, 2]
    w2, b = budget.directional_budget(evidence[order, ::-1], truth[order, ::-1])
    np.testing.assert_array_equal(w, w2)
    for first, second in zip(a["utility_table"], b["utility_table"]):
        assert first["total_true_positives"] == second["total_true_positives"]
        np.testing.assert_array_equal(np.array(first["true_positives_by_family"])[order], second["true_positives_by_family"])


@pytest.mark.parametrize("alpha", [0, 1, -.1, 2, np.nan, np.inf, -np.inf, True, False, np.bool_(True), [.05], "0.05", 1j, np.nextafter(0., 1.)])
def test_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        budget.directional_budget(np.ones((1, 2, 2)), np.ones((1, 2, 2), bool), alpha=alpha)


@pytest.mark.parametrize("grid", [[], [0., 2.], [1., 0.], [1., 1.], [-.1, 1.], [1., 2.1], [1., np.nan], [1., np.inf], [[1.]], [True], ["1"], [1j]])
def test_invalid_grid(grid):
    with pytest.raises(ValueError):
        budget.directional_budget(np.ones((1, 2, 2)), np.ones((1, 2, 2), bool), positive_weight_grid=grid)


@pytest.mark.parametrize("shape", [(), (2,), (2, 2), (0, 2, 2), (2, 0, 2), (2, 2, 1), (2, 2, 3), (1, 1, 1, 2)])
def test_invalid_shape(shape):
    with pytest.raises(ValueError):
        budget.directional_budget(np.ones(shape), np.ones(shape, bool))


@pytest.mark.parametrize("value", [-1., np.nan, np.inf, -np.inf, 1j, "1", True])
def test_invalid_evidence(value):
    with pytest.raises(ValueError):
        budget.directional_budget(np.full((1, 2, 2), value), np.ones((1, 2, 2), bool))


@pytest.mark.parametrize("truth", [np.ones((1, 2, 2)), np.zeros((1, 2, 2), int), np.ones((1, 2, 1), bool), True, [[["true", "false"]]]])
def test_truth_requires_boolean_exact_shape(truth):
    with pytest.raises(ValueError):
        budget.directional_budget(np.ones((1, 2, 2)), truth)


def test_nonrepresentable_multiplication_raises_without_clipping_or_hidden_fallback():
    with pytest.raises(FloatingPointError):
        budget.directional_budget(np.full((1, 2, 2), np.finfo(float).max), np.ones((1, 2, 2), bool))


def test_custom_prespecified_grid_receipt_and_method_neutrality():
    evidence = np.full((2, 4, 2), 80.)
    truth = np.ones_like(evidence, bool)
    grid = (0., .4, 1., 1.6, 2.)
    first = budget.directional_budget(evidence, truth, positive_weight_grid=grid)
    # A simple comparator receives exactly this same API/grid, not a smaller
    # search; no method-name or fitted-truth override exists in this function.
    second = budget.directional_budget(evidence.copy(), truth.copy(), positive_weight_grid=grid)
    assert first[1] == second[1]
    assert first[1]["positive_weight_grid"] == list(grid)
    assert not first[1]["default_grid_used"]
    assert list(inspect.signature(budget.directional_budget).parameters) == [
        "simulated_e", "simulated_truth", "alpha", "positive_weight_grid"]
    for key in ["uses_real_truth_labels", "held_observation_access", "provenance_verified",
                "uses_simulated_fdp_for_selection", "gamma_reselected",
                "individual_weighted_e_validity_claimed", "weights_renormalized_across_genes_or_folds"]:
        assert first[1][key] is False


def test_crossfold_tower_budget_and_fdr_by_exact_enumeration():
    # Fixed all-null signed hypotheses; independent gene states. Each raw
    # directional E has expectation .25*4=1. Weights for fold 0 depend only
    # on fold 1 and vice versa; all training sigma-fields together reveal ALL
    # gene outcomes, so one cannot condition on their union to prove validity.
    states = np.array([[4., 0.], [0., 4.], [0., 0.]])
    probability = [.25, .25, .5]
    expected_null_sum = 0.
    fdr = 0.
    for ids in product(range(3), repeat=4):
        mass = float(np.prod([probability[i] for i in ids]))
        e = states[list(ids)]
        w = np.zeros_like(e)
        for fold in [0, 1]:
            held = np.arange(4) % 2 == fold
            train_totals = e[~held].sum(0)
            pair = ([1., 1.] if train_totals[0] == train_totals[1]
                    else [2., 0.] if train_totals[0] > train_totals[1] else [0., 2.])
            w[held] = pair
        assert w.sum() == 8
        weighted = w * e
        expected_null_sum += mass * weighted.sum()
        fdr += mass * bool(ebh(weighted, .25).any())
    assert expected_null_sum == 8.
    assert fdr <= .25
    # An individual weighted null evidence can have mean 2, not mean 1.
    assert sum(p * 2 * e[0] for p, e in zip(probability, states)) == 2.


def test_held_adaptive_direction_breaks_control_despite_sum_two_counterexample():
    alpha = .05
    # One gene, both directions null. Disjoint rare events give valid raw
    # E+ and E- with means exactly 1, but choosing the observed lucky direction
    # spends the two-unit budget twice in expectation.
    states = np.array([[1 / alpha, 0.], [0., 1 / alpha], [0., 0.]])
    probabilities = [alpha, alpha, 1 - 2 * alpha]
    np.testing.assert_allclose(np.array(probabilities) @ states, [1., 1.])
    false_discovery_probability = 0.
    for e, p in zip(states, probabilities):
        bad_weights = np.array([2., 0.]) if e[0] > 0 else np.array([0., 2.])
        assert bad_weights.sum() == 2.
        false_discovery_probability += p * bool(ebh((e * bad_weights)[None], alpha).any())
    assert false_discovery_probability == 2 * alpha


@pytest.mark.parametrize("true_indices", [[], [(0, 0)], [(0, 0), (1, 1)], [(0, 0), (0, 1), (2, 1)]])
def test_exact_crossfold_budget_with_fixed_alternatives_and_unequal_folds(true_indices):
    states = np.array([[4., 0.], [0., 4.], [0., 0.]])
    probabilities = [.25, .25, .5]
    null = np.ones((3, 2), bool)
    for index in true_indices:
        null[index] = False
    expected_null_sum = 0.
    fdr = 0.
    for ids in product(range(3), repeat=3):
        mass = float(np.prod([probabilities[i] for i in ids]))
        raw = states[list(ids)].copy()
        raw[~null] = 1000.  # fixed alternatives need no expectation bound
        allocation = np.zeros_like(raw)
        for fold in [0, 1]:
            held = np.arange(3) % 2 == fold
            training_score = raw[~held].sum(0)
            positive = (1. if training_score[0] == training_score[1] else
                        1.75 if training_score[0] > training_score[1] else .25)
            allocation[held] = [positive, 2 - positive]
        assert allocation.sum() == 6
        weighted = raw * allocation
        expected_null_sum += mass * weighted[null].sum()
        rejected = ebh(weighted, .25)
        fdr += mass * (rejected & null).sum() / max(1, rejected.sum())
    assert expected_null_sum <= 6
    assert fdr <= .25


@pytest.mark.parametrize("candidate", ["projection", "support_simes"])
@pytest.mark.parametrize("df", [np.inf, 25., 1.5])
@pytest.mark.parametrize("gamma_policy", ["predictive", "old_mass_gate", "raw"])
def test_composition_uses_existing_simulated_e_after_gamma_without_reselection(monkeypatch, candidate, df, gamma_policy):
    # Capture the exact narrow interface needed by main; do not modify the
    # existing helper, rerun its simulator, fit a pilot, or add gamma options.
    patterns, amplitudes = pattern_library()
    weights = np.zeros(241)
    weights[0] = .6
    weights[np.flatnonzero((patterns == [1, 1, 0, 0]).all(1) & (amplitudes == 3.5))[0]] = .25
    weights[np.flatnonzero((patterns == [-1, -1, -1, -1]).all(1) & (amplitudes == 3.5))[0]] = .15
    n, genes, families = 24, 12, 3
    train_p = np.ones((n, 2))
    train_p[:8, 0] = 1e-5
    _, pilot = pilot_calibrator(np.ones((1, 2)), train_p, 2 * genes, .8)
    original = gate._grid_utilities
    captured = {}

    def capture(cone, candidate_e, truth, alpha):
        answer = original(cone, candidate_e, truth, alpha)
        gamma = answer[0]
        captured["mixed_e"] = (1 - gamma) * cone + gamma * candidate_e
        captured["cone_e"], captured["candidate_e"] = cone.copy(), candidate_e.copy()
        captured["truth"] = truth.copy()
        captured["gamma"] = gamma.copy()
        captured["alpha"] = alpha
        return answer

    monkeypatch.setattr(gate, "_grid_utilities", capture)
    gamma, original_receipt = gate.predictive_gate(
        weights, np.linspace(.4, 1.4, n), np.linspace(.5, 2., n),
        .65 * np.ones((4, 4)) + .35 * np.eye(4), df,
        [[1., 1., 0., 0.], [1., 1., 1., 1.]], genes, pilot, pilot,
        candidate=candidate, families=families,
    )
    selected, receipt = budget.directional_budget(captured["mixed_e"], captured["truth"], alpha=captured["alpha"])
    np.testing.assert_array_equal(gamma, captured["gamma"])
    assert receipt["equal_weight_total_true_positives"] == original_receipt["selected_total_true_positives"]
    assert receipt["total_simulated_true_claims"] == original_receipt["total_simulated_true_claims"]
    assert receipt["selected_total_true_positives"] >= original_receipt["selected_total_true_positives"]
    assert selected.sum() == 2
    # Alternate callers retain their gamma rather than inheriting predictive
    # selection. This is the OLD mass-gate formula on frozen training summaries.
    if gamma_policy == "old_mass_gate":
        mass = np.array([.25, .15])
        dominant_fraction = np.array([.7, .9])
        fixed_gamma = np.where(mass > max(.03, 3 / n),
                               np.clip((dominant_fraction - .4) / .4, 0, .9), 0.)
    elif gamma_policy == "raw":
        fixed_gamma = np.ones(2)
    else:
        fixed_gamma = gamma.copy()
    supplied = (1 - fixed_gamma) * captured["cone_e"] + fixed_gamma * captured["candidate_e"]
    gamma_snapshot = fixed_gamma.copy()
    _, independent_budget = budget.directional_budget(supplied, captured["truth"])
    no_budget, ablation = budget.directional_budget(supplied, captured["truth"], positive_weight_grid=(1.,))
    expected_tp = sum(int((ebh(e, .05) & t).sum()) for e, t in zip(supplied, captured["truth"]))
    assert independent_budget["equal_weight_total_true_positives"] == expected_tp
    assert ablation["selected_total_true_positives"] == expected_tp
    np.testing.assert_array_equal(no_budget, [1., 1.])
    np.testing.assert_array_equal(fixed_gamma, gamma_snapshot)


def test_microcost(capsys):
    rng = np.random.default_rng(32095)
    evidence = rng.exponential(100., (16, 256, 2))
    truth = rng.random(evidence.shape) < .2
    durations = []
    for _ in range(5):
        start = time.perf_counter()
        budget.directional_budget(evidence, truth)
        durations.append(time.perf_counter() - start)
    with capsys.disabled():
        print("\nDirectional budget 16x256x2, grid9 seconds:", durations,
              "median:", float(np.median(durations)))
    # Timing is descriptive, not a flaky machine-load acceptance threshold.
    assert all(np.isfinite(durations))
