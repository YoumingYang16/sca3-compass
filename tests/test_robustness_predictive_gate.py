import copy
import inspect
import json
import time

import numpy as np
import pytest

from sca3_compass import robustness_predictive_gate as gate
from sca3_compass.robustness_cone import cone_partial_conjunction
from sca3_compass.robustness_patterns import pattern_library
from sca3_compass.robustness_pilot import pilot_calibrator
from sca3_compass.robustness_projection import projection_pc


def inputs(genes=12, **changes):
    patterns, amplitudes = pattern_library()
    weights = np.zeros(len(patterns))
    weights[0] = .7
    weights[np.flatnonzero(((patterns == [1, 1, 0, 0]).all(1)) & (amplitudes == 3.5))[0]] = .2
    weights[np.flatnonzero(((patterns == [-1, -1, -1, -1]).all(1)) & (amplitudes == 2.5))[0]] = .1
    n = 24
    training = np.ones((n, 2))
    training[:8, 0] = 1e-5
    _, receipt = pilot_calibrator(np.ones((1, 2)), training, 2 * genes, .8)
    result = dict(
        pattern_weights=weights, training_variance=np.linspace(.4, 1.4, n),
        training_q_weights=np.linspace(.5, 2., n),
        study_shape=.65 * np.ones((4, 4)) + .35 * np.eye(4), df=23.,
        profiles=np.array([[1., 1., 0., 0.], [1., 1., 1., 1.]]),
        family_gene_count=genes, cone_pilot=copy.deepcopy(receipt),
        candidate_pilot=copy.deepcopy(receipt), families=3,
    )
    result.update(changes)
    return result


@pytest.mark.parametrize("candidate", ["projection", "support_simes"])
@pytest.mark.parametrize("df", [np.inf, 23., 1.5])
def test_reproducible_finite_and_json_safe(candidate, df):
    args = inputs(candidate=candidate, df=df)
    original = copy.deepcopy(args)
    gamma, receipt = gate.predictive_gate(**args)
    gamma_again, receipt_again = gate.predictive_gate(**args)
    np.testing.assert_array_equal(gamma, gamma_again)
    assert receipt == receipt_again
    json.dumps(receipt, allow_nan=False)
    assert gamma.shape == (2,) and np.isfinite(gamma).all()
    assert np.isin(gamma, gate.GAMMA_GRID).all()
    assert np.all(gamma >= 0) and np.all(1 - gamma >= 0)
    assert len(receipt["utility_table"]) == 25
    selected = receipt["utility_table"][receipt["selected_grid_index"]]
    assert receipt["selected_utility_name"] == "pooled_simulated_signed_true_positive_rate"
    assert receipt["selected_utility"] == selected["pooled_true_positive_rate"]
    assert receipt["selected_total_true_positives"] == selected["total_true_positives"]
    assert receipt["selected_grid_indices"] == selected["grid_indices"]
    assert receipt["selected_descriptive_mc_se"] == selected["descriptive_mc_se_pooled_power"]
    assert receipt["selected_grid_index"] == 5 * selected["grid_indices"][0] + selected["grid_indices"][1]
    assert receipt["gaussian"] == np.isinf(df)
    assert not receipt["uses_real_truth_labels"]
    assert receipt["uses_simulated_truth_for_utility"]
    assert not receipt["uses_simulated_fdp_for_selection"]
    assert not receipt["held_observation_access"]
    assert not receipt["q_weights_renormalized"]
    for key, value in args.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, original[key])
        else:
            assert value == original[key]


def test_row_permutation_preserves_pairs_and_all_results():
    args = inputs()
    expected = gate.predictive_gate(**args)
    order = np.random.default_rng(11).permutation(len(args["training_variance"]))
    args["training_variance"] = args["training_variance"][order]
    args["training_q_weights"] = args["training_q_weights"][order]
    actual = gate.predictive_gate(**args)
    np.testing.assert_array_equal(expected[0], actual[0])
    assert expected[1] == actual[1]


@pytest.mark.parametrize("genes", [1, 7, 12, 256])
@pytest.mark.parametrize("discoveries", [0, 1, 10, 48])
@pytest.mark.parametrize("fraction", [0., .5, .8, 1.])
def test_frozen_calibrator_exact_parity(genes, discoveries, fraction):
    training = np.ones((24, 2))
    training.ravel()[:discoveries] = 0.
    p = np.r_[0., np.geomspace(1e-10, 1., 30)].reshape(-1, 1)
    expected, receipt = pilot_calibrator(p, training, 2 * genes, fraction)
    actual = gate.apply_frozen_pilot(p, receipt, genes)
    np.testing.assert_array_equal(actual, expected)
    # Check threshold equality and both sides, including exact step inclusion.
    threshold = receipt["threshold"]
    boundary = np.array([np.nextafter(threshold, 0.), threshold, np.nextafter(threshold, 1.)])
    np.testing.assert_array_equal(
        gate.apply_frozen_pilot(boundary, receipt, genes),
        pilot_calibrator(boundary, training, 2 * genes, fraction)[0],
    )


@pytest.mark.parametrize("pattern, expected", [
    ([1, 1, 0, 0], [True, False]), ([-1, 0, -1, 0], [False, True]),
    ([1, 1, -1, -1], [True, True]), ([1, 0, 0, -1], [False, False]),
    ([0, 0, 0, 0], [False, False]),
])
def test_simulated_truth_uses_actual_location_not_observed_noise(pattern, expected):
    patterns, amplitudes = pattern_library()
    weights = np.zeros(len(patterns))
    match = np.flatnonzero((patterns == pattern).all(1))[0]
    weights[match] = 1.
    pair = np.array([[1e4, .1], [3e5, 7.]])  # noisy signs need not equal truth
    means, variance, q_weight, truth, components, indices = gate._sample_predictive(
        weights, pair, np.eye(4), 1.5, 11, 3, gate.DEFAULT_SEED,
    )
    np.testing.assert_array_equal(truth, np.broadcast_to(expected, (3, 11, 2)))
    np.testing.assert_array_equal(components, np.full((3, 11), match))
    np.testing.assert_array_equal(variance, pair[indices, 0])
    np.testing.assert_array_equal(q_weight, pair[indices, 1])
    assert np.isfinite(means).all()


@pytest.mark.parametrize("df", [np.inf, 23., 1.5])
def test_sampler_exact_gaussian_and_shared_student_radial_definition(df):
    args = inputs()
    weights = args["pattern_weights"]
    pairs = np.array([[.3, 1.1], [2., .7], [5., 3.]])
    shape = np.diag([2., 1., .5, 3.])
    seed, families, genes = 812, 2, 7
    actual = gate._sample_predictive(weights, pairs, shape, df, genes, families, seed)
    rng = np.random.Generator(np.random.PCG64(seed))
    components = rng.choice(241, size=(families, genes), p=weights)
    index = rng.integers(3, size=(families, genes))
    normals = rng.standard_normal((families, genes, 4)) @ np.linalg.cholesky(shape).T
    if np.isfinite(df):
        normals = normals / np.sqrt(rng.chisquare(df, size=(families, genes)) / df)[..., None]
    patterns, amplitudes = pattern_library()
    expected = patterns[components] * amplitudes[components, None] + np.sqrt(pairs[index, 0])[..., None] * normals
    np.testing.assert_array_equal(actual[0], expected)
    np.testing.assert_array_equal(actual[4], components)
    np.testing.assert_array_equal(actual[5], index)


def _reference_reject(e, alpha):
    ordered = np.sort(e.ravel())[::-1]
    ranks = [k for k in range(1, e.size + 1) if ordered[k - 1] >= e.size / (alpha * k)]
    return e >= ordered[max(ranks) - 1] if ranks else np.zeros(e.shape, bool)


def test_all_grid_utilities_against_independent_family_reference():
    rng = np.random.default_rng(8383)
    cone_e = rng.choice([0., 2., 10., 50., 80., 300.], size=(5, 8, 2))
    candidate_e = rng.choice([0., 2., 10., 50., 80., 300.], size=(5, 8, 2))
    truth = rng.random((5, 8, 2)) < .3
    gamma, table, selected = gate._grid_utilities(cone_e, candidate_e, truth, .05)
    for row in table:
        mixing = np.array(row["gamma"])
        e = (1 - mixing) * cone_e + mixing * candidate_e
        rejected = np.stack([_reference_reject(family, .05) for family in e])
        tp = (rejected & truth).sum((1, 2))
        fp = (rejected & ~truth).sum((1, 2))
        np.testing.assert_array_equal(row["true_positives_by_family"], tp)
        np.testing.assert_array_equal(row["false_positives_by_family"], fp)
        assert row["pooled_true_positive_rate"] == tp.sum() / truth.sum()
        np.testing.assert_allclose(row["simulated_fdp_by_family"], fp / np.maximum(1, tp + fp))
        u = row["pooled_true_positive_rate"]
        expected_se = np.std(tp - u * truth.sum((1, 2)), ddof=1) / np.sqrt(5) / truth.sum((1, 2)).mean()
        assert row["descriptive_mc_se_pooled_power"] == expected_se
    expected = sorted(table, key=lambda row: (-row["total_true_positives"], sum(row["gamma"]), *row["gamma"]))[0]
    assert table[selected] == expected
    np.testing.assert_array_equal(gamma, expected["gamma"])


def test_zero_simulated_truth_ties_choose_cone_even_if_fdp_differs():
    zeros = np.zeros((2, 4, 2))
    gamma, table, selected = gate._grid_utilities(zeros, zeros + 1e6, zeros.astype(bool), .05)
    np.testing.assert_array_equal(gamma, [0., 0.])
    assert selected == 0
    assert all(row["pooled_true_positive_rate"] == 0. for row in table)
    assert all(row["descriptive_mc_se_pooled_power"] is None for row in table)
    assert table[0]["mean_simulated_fdp"] == 0.
    assert table[-1]["mean_simulated_fdp"] == 1.


def test_equal_evalues_tie_break_and_single_family_uncertainty():
    rng = np.random.default_rng(999)
    e = rng.choice([0., 20., 100.], size=(1, 4, 2))
    truth = np.ones((1, 4, 2), bool)
    gamma, table, selected = gate._grid_utilities(e, e, truth, .05)
    np.testing.assert_array_equal(gamma, [0., 0.])
    assert selected == 0
    assert all(row["descriptive_mc_se_pooled_power"] is None for row in table)
    assert all(row["descriptive_mc_se_mean_fdp"] is None for row in table)


def test_fair_support_simes_same_simulation_and_training_information():
    args = inputs()
    _, projection = gate.predictive_gate(**args)
    _, simes = gate.predictive_gate(**args, candidate="support_simes")
    for key in ("simulation_sha256", "predictive_model_sha256", "component_indices_sha256", "prior_sha256", "training_pair_sha256",
                "training_pair_indices_sha256", "seed", "families", "gamma_grid", "cone_pilot", "candidate_pilot"):
        assert projection[key] == simes[key]
    assert projection["inputs_sha256"] != simes["inputs_sha256"]
    assert simes["support_simes_diagnostics"]["intersection_count"] == 8


def test_mode_specific_pilot_and_profiles_do_not_change_simulations(monkeypatch):
    # A trained Simes comparator may have different pilot counts/profiles.
    # Those differences must not alter C, J, Student radii or observations.
    args = inputs()
    sampled = []
    original = gate._sample_predictive
    def recording_sampler(*args, **kwargs):
        result = original(*args, **kwargs)
        sampled.append(tuple(array.copy() for array in result))
        return result
    monkeypatch.setattr(gate, "_sample_predictive", recording_sampler)
    _, projection = gate.predictive_gate(**args)
    training = np.ones((24, 2))
    training[:3, 1] = 1e-8
    _, different_pilot = pilot_calibrator(np.ones(2), training, 24, .8)
    args["candidate_pilot"] = different_pilot
    args["profiles"] = np.ones((2, 4))
    _, simes = gate.predictive_gate(**args, candidate="support_simes")
    assert len(sampled) == 2
    for left, right in zip(*sampled):
        np.testing.assert_array_equal(left, right)
    assert projection["simulation_sha256"] == simes["simulation_sha256"]
    assert projection["predictive_model_sha256"] == simes["predictive_model_sha256"]
    assert projection["inputs_sha256"] != simes["inputs_sha256"]
    assert projection["candidate_pilot"] != simes["candidate_pilot"]
    for result in (projection, simes):
        selected = result["utility_table"][result["selected_grid_index"]]
        assert result["selected_utility"] == selected["pooled_true_positive_rate"]
        assert result["selected_total_true_positives"] == sum(selected["true_positives_by_family"])


def test_negative_correlation_support_simes_uses_existing_fallback():
    shape = np.eye(4)
    shape[0, 1] = shape[1, 0] = -.4
    _, info = gate.predictive_gate(**inputs(study_shape=shape, candidate="support_simes"))
    assert info["support_simes_diagnostics"]["fallback_intersection_count"] > 0


@pytest.mark.parametrize("candidate", ["projection", "support_simes"])
def test_assembled_simulated_evalues_match_existing_deployed_algorithms(monkeypatch, candidate):
    args = inputs(genes=7, candidate=candidate, study_shape=np.diag([2., 1., .5, 3.]))
    captured = {}
    original = gate._grid_utilities
    def capture(cone_e, candidate_e, truth, alpha):
        captured.update(cone=cone_e.copy(), candidate=candidate_e.copy(), truth=truth.copy())
        return original(cone_e, candidate_e, truth, alpha)
    monkeypatch.setattr(gate, "_grid_utilities", capture)
    gate.predictive_gate(**args)
    pairs = np.column_stack((args["training_variance"], args["training_q_weights"]))
    means, variance, weight, truth, _, _ = gate._sample_predictive(
        args["pattern_weights"], pairs, args["study_shape"], args["df"], 7, 3, gate.DEFAULT_SEED,
    )
    signed = np.stack((means.reshape(-1, 4), -means.reshape(-1, 4)), axis=1)
    cone_pc = cone_partial_conjunction(signed, variance.ravel(), [args["study_shape"]] * 2, args["df"])
    alternative_pc = projection_pc(signed, variance.ravel(), args["study_shape"], args["df"],
                                   args["profiles"], support_simes=candidate == "support_simes")
    training = np.ones((24, 2))
    training[:8, 0] = 1e-5
    for key, pc in (("cone", cone_pc), ("candidate", alternative_pc)):
        expected, _ = pilot_calibrator(np.minimum(1, pc / weight.reshape(-1, 1)), training, 14, .8)
        np.testing.assert_array_equal(captured[key], expected.reshape(3, 7, 2))
    np.testing.assert_array_equal(captured["truth"], truth)


def test_nonrepresentable_tail_draw_raises_without_reseed_or_clipping():
    args = inputs()
    with pytest.raises(FloatingPointError, match="radial draw"):
        gate._sample_predictive(args["pattern_weights"], np.array([[1., 1.]]), np.eye(4),
                                1e-300, 4, 2, gate.DEFAULT_SEED)


def test_predictive_gate_has_no_real_observations_or_truth_interface(monkeypatch):
    parameters = set(inspect.signature(gate.predictive_gate).parameters)
    assert parameters == {
        "pattern_weights", "training_variance", "training_q_weights", "study_shape", "df", "profiles",
        "family_gene_count", "cone_pilot", "candidate_pilot", "candidate", "families", "seed", "alpha",
    }
    with pytest.raises(TypeError):
        gate.predictive_gate(**inputs(), held_means=np.zeros((4, 4)))
    with pytest.raises(TypeError):
        gate.predictive_gate(**inputs(), true_labels=np.ones((4, 2)))
    # Pilot MUST NOT be reselected on simulated/held probabilities.
    import sca3_compass.robustness_pilot as pilot_module
    args = inputs()
    def forbidden(*args, **kwargs):
        raise AssertionError("Pilot recalibration is forbidden")
    monkeypatch.setattr(pilot_module, "pilot_calibrator", forbidden)
    gate.predictive_gate(**args)


@pytest.mark.parametrize("key, value", [
    ("pattern_weights", np.ones(241)), ("pattern_weights", np.ones(240) / 240),
    ("pattern_weights", np.full(241, np.nan)), ("training_variance", []),
    ("training_variance", np.ones(23)), ("training_q_weights", np.zeros(24)),
    ("training_q_weights", np.full(24, np.inf)), ("df", -1.), ("df", 0.), ("df", np.nan),
    ("df", -np.inf), ("study_shape", np.zeros((4, 4))),
    ("study_shape", np.eye(3)), ("study_shape", np.full((4, 4), np.nan)),
    ("profiles", -np.ones((2, 4))), ("profiles", np.ones((4, 2))),
    ("family_gene_count", 0), ("family_gene_count", 3.5), ("family_gene_count", True),
    ("families", 0), ("families", False), ("seed", -1), ("seed", 4.5),
    ("candidate", "oracle"), ("alpha", np.nan), ("alpha", 0.),
])
def test_invalid_inputs_rejected(key, value):
    with pytest.raises(ValueError):
        gate.predictive_gate(**inputs(**{key: value}))


@pytest.mark.parametrize("field, value", [
    ("threshold", .002), ("pilot_level", .1), ("training_signed_count", 46),
    ("pilot_discoveries", 49), ("uses_held_probabilities_for_selection", True),
    ("uses_truth_labels", True), ("zero_count_fallback", True),
    ("concentration_fraction", np.nan), ("threshold_rule", "test-selected"),
])
def test_invalid_or_mismatched_pilot_receipt_rejected(field, value):
    args = inputs()
    args["candidate_pilot"][field] = value
    with pytest.raises(ValueError):
        gate.predictive_gate(**args)


def test_default_actual_family_cost_and_grid_accounting():
    args = inputs(genes=256)
    del args["families"]
    start = time.perf_counter()
    _, info = gate.predictive_gate(**args)
    elapsed = time.perf_counter() - start
    assert info["families"] == 16 and info["seed"] == 719991
    assert info["signed_family_size"] == 512
    assert sum(info["simulated_component_counts"]) == 16 * 256
    assert len(info["utility_table"]) == 25
    assert all(len(row["true_positives_by_family"]) == 16 for row in info["utility_table"])
    print(f"predictive gate: 16 x 256 genes, 25 pairs, {elapsed:.4f}s (not a throughput claim)")
