"""Bounded mechanism fixtures, not acceptance experiments or FDR evidence."""

import json
from copy import deepcopy

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_loading_audit_gate as module
from sca3_compass.molecular_envelope_benchmark import fixed_truth
from sca3_compass.robustness_loading import loading_gate, pipeline_roots


@pytest.fixture(autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1):
        yield


def shared_fixture():
    rng = np.random.default_rng(130987)
    _, root, inverse = pipeline_roots(.8, 6)
    noise = rng.normal(size=(240, 4, 6)) @ root
    effect = np.zeros((240, 4, 1))
    effect[:120] = rng.choice([-4., 4.], size=(120, 4, 1))
    loading = np.array([.2, .5, .8, 1.2, 1.5, 1.8])
    return noise, effect, loading, root, inverse


def mismatch_fixture(df=None):
    rng = np.random.default_rng(271)
    _, actual, _ = pipeline_roots(.1, 6)
    _, frozen, inverse = pipeline_roots(.95, 6)
    x = rng.normal(size=(240, 4, 6)) @ actual
    if df is not None:
        x *= np.sqrt((df-2 if df > 2 else df)/rng.chisquare(df, 240))[:, None, None]
    x[:48, :2] += 3.5
    return x, frozen, inverse


def test_raw_score_matches_scalar_orthogonal_projection_not_gls():
    rng = np.random.default_rng(801)
    x = rng.normal(size=(19, 4, 6))
    ell = np.array([.2, .5, .8, 1.2, 1.5, 1.8])
    actual, qa = module.euclidean_log_residual_gain(x, ell)
    u, v = ell/np.linalg.norm(ell), np.ones(6)/np.sqrt(6)
    expected = np.empty((19, 4))
    for g in range(19):
        for s in range(4):
            row = x[g, s]
            old = sum((row - np.dot(row, v)*v)**2)
            new = sum((row - np.dot(row, u)*u)**2)
            expected[g, s] = np.log(old/new)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=1e-12)
    assert qa["uniform_residual_floor_count"] == qa["loading_residual_floor_count"] == 0


def test_score_scale_invariance_including_extreme_finite_values():
    rng = np.random.default_rng(802)
    x = rng.normal(size=(12, 4, 6))
    ell = np.arange(1., 7.)
    original, _ = module.euclidean_log_residual_gain(x, ell)
    scales = np.logspace(-280, 280, 48).reshape(12, 4, 1)
    changed, _ = module.euclidean_log_residual_gain(x*scales, ell*1e250)
    np.testing.assert_allclose(original, changed, atol=2e-14)


def test_uniform_direction_and_zero_vectors_have_zero_gain():
    x = np.random.default_rng(803).normal(size=(12, 4, 6))
    x[2] = 0
    scores, qa = module.euclidean_log_residual_gain(x, np.ones(6))
    np.testing.assert_array_equal(scores, 0)
    assert qa["zero_vector_count"] == 4


def test_exact_projection_degeneracy_is_reported_not_silently_removed():
    # K=4 gives an exactly representable uniform unit direction and zero SSE.
    x = np.ones((12, 4, 4))
    scores, qa = module.euclidean_log_residual_gain(x, np.ones(4))
    assert np.isfinite(scores).all()
    assert qa["uniform_residual_floor_count"] == 48
    assert qa["loading_residual_floor_count"] == 48


@pytest.mark.parametrize("df", [None, 5., 1.5])
def test_down_covariance_mismatch_suppresses_false_loading_gate(df):
    x, root, inverse = mismatch_fixture(df)
    adopt, info = module.raw_loading_gate(x, root, inverse, include_old_gate=True)
    assert info["old_gate"]["adopt"], "Fixture must reproduce the old gate's mechanism failure"
    assert not adopt
    assert info["mean_log_residual_gain"] < 0
    assert info["all_inner_fits_converged"]
    assert info["numerical_failure_count"] == 0
    json.dumps(info, allow_nan=False)


def test_real_shared_loadings_retained_but_uniform_signal_not_adopted():
    noise, effect, ell, root, inverse = shared_fixture()
    common, common_info = module.raw_loading_gate(noise+effect, root, inverse)
    unequal, unequal_info = module.raw_loading_gate(noise+effect*ell, root, inverse,
                                                   include_old_gate=True)
    assert not common and common_info["all_inner_fits_converged"]
    assert unequal and unequal_info["old_gate"]["adopt"]
    assert unequal_info["mean_log_residual_gain"] > 1
    assert unequal_info["all_inner_fits_converged"]
    assert unequal_info["numerical_failure_count"] == 0


@pytest.mark.parametrize("kind,actual_rho,calibration_rho,expected_raw,expected_old", [
    ("mismatch", .1, .95, [False, False], [True, True]),
    ("shared", .8, .8, [False, True], [True, True]),
    ("uniform", .8, .8, [False, False], [False, False]),
])
def test_correlated_t5_fixture_keeps_false_negative_boundary_visible(
        kind, actual_rho, calibration_rho, expected_raw, expected_old):
    """One fixed sample, NOT FDR/power evidence; shared-fold0 is a known cost.

    It is intentionally NOT removed or repaired by changing the gate margins.
    All means, data, and fold choices here are used only for testing the gate.
    """
    rng = np.random.default_rng(771)
    study_root = np.linalg.cholesky(.35*np.eye(4)+.65*np.ones((4, 4)))
    base = np.einsum("st,gtk->gsk", study_root, rng.normal(size=(256, 4, 6)))
    radial = np.sqrt(3/rng.chisquare(5, 256))[:, None, None]
    mu = fixed_truth(256, 4, {"effect": 3.5, "replicated_fraction": .2}, .2)
    ell = np.array([.2, .5, .8, 1.2, 1.5, 1.8]) if kind == "shared" else np.ones(6)
    _, actual, _ = pipeline_roots(actual_rho, 6)
    _, root, inverse = pipeline_roots(calibration_rho, 6)
    x = (base @ actual)*radial + mu[..., None]*ell
    for fold in range(2):
        adopt, info = module.raw_loading_gate(x[np.arange(256) % 2 != fold], root, inverse,
                                             include_old_gate=True)
        assert adopt == expected_raw[fold]
        assert info["old_gate"]["adopt"] == expected_old[fold]
        assert info["all_inner_fits_converged"] and info["numerical_failure_count"] == 0
        if kind == "shared" and fold == 0:
            assert info["mean_log_residual_gain"] > .05
            assert info["mean_log_residual_gain"] < 2.58*info["gene_level_standard_error"]


def test_gene_score_is_mean_of_study_logs_not_log_of_sum():
    x, root, inverse = mismatch_fixture()
    _, info = module.raw_loading_gate(x, root, inverse)
    folds = np.arange(len(x)) % 3
    expected = np.empty(len(x))
    for entry in info["inner_folds"]:
        held = folds == entry["fold"]
        study_scores, _ = module.euclidean_log_residual_gain(x[held], entry["loading"])
        expected[held] = study_scores.mean(axis=1)
    np.testing.assert_array_equal(info["per_gene_log_residual_gain"], expected)
    assert info["mean_log_residual_gain"] == float(expected.mean())
    assert info["gene_level_standard_error"] == float(expected.std(ddof=1)/np.sqrt(len(x)))
    assert info["adopt"] == (expected.mean() > .05 and
                             expected.mean() > 2.58*expected.std(ddof=1)/np.sqrt(len(x)))


@pytest.mark.parametrize("kind", ["common", "unequal", "mismatch"])
def test_optional_old_gate_reproduces_legacy_and_does_not_change_raw(kind):
    noise, effect, ell, root, inverse = shared_fixture()
    x = noise + effect*(ell if kind == "unequal" else 1)
    if kind == "mismatch":
        x, root, inverse = mismatch_fixture()
    old_adopt, old_info = loading_gate(x, root, inverse)
    adopted, actual = module.raw_loading_gate(x, root, inverse, include_old_gate=True)
    raw_only, raw_info = module.raw_loading_gate(x, root, inverse)
    assert adopted == raw_only
    extra = actual.pop("old_gate")
    assert actual == raw_info
    assert extra["adopt"] == old_adopt
    for key in ["mean_log_residual_gain", "gene_level_standard_error", "inner_fits"]:
        assert extra[key] == old_info[key]


def test_only_three_old_learner_calls_and_each_excludes_inner_held(monkeypatch):
    x, root, inverse = mismatch_fixture()
    calls, actual_learner = [], module.learned_loading

    def recording(train, frozen_root):
        calls.append(train.copy())
        np.testing.assert_array_equal(frozen_root, root)
        return actual_learner(train, frozen_root)

    monkeypatch.setattr(module, "learned_loading", recording)
    module.raw_loading_gate(x, root, inverse, include_old_gate=True)
    assert len(calls) == 3
    for fold, call in enumerate(calls):
        np.testing.assert_array_equal(call, (x @ inverse)[np.arange(len(x)) % 3 != fold])


def test_changing_inner_held_does_not_change_its_loading_or_fit():
    x, root, inverse = mismatch_fixture()
    _, original = module.raw_loading_gate(x, root, inverse)
    changed = x.copy()
    changed[np.arange(len(x)) % 3 == 0] += np.arange(6)*8
    _, after = module.raw_loading_gate(changed, root, inverse)
    for key in ["loading", "loading_fit"]:
        assert original["inner_folds"][0][key] == after["inner_folds"][0][key]
    original_score = np.array(original["per_gene_log_residual_gain"])[::3]
    changed_score = np.array(after["per_gene_log_residual_gain"])[::3]
    assert not np.allclose(original_score, changed_score)  # scoring legitimately sees held data


def test_outer_held_data_never_enter_training_api():
    x, root, inverse = mismatch_fixture()
    outer_held = np.arange(len(x)) % 2 == 0
    before = module.raw_loading_gate(x[~outer_held], root, inverse)
    x[outer_held] += 1e5
    after = module.raw_loading_gate(x[~outer_held], root, inverse)
    assert before == after


def test_input_immutability_readonly_and_diagnostic_ownership():
    x, root, inverse = mismatch_fixture()
    folds = np.arange(len(x)) % 3
    snapshots = [a.copy() for a in [x, root, inverse, folds]]
    for value in [x, root, inverse, folds]:
        value.flags.writeable = False
    _, info = module.raw_loading_gate(x, root, inverse, include_old_gate=True, inner_fold_ids=folds)
    info["inner_fold_ids"][0] = 99
    for actual, expected in zip([x, root, inverse, folds], snapshots, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_default_fold_policy_is_explicit_and_repeat_deterministic():
    x, root, inverse = mismatch_fixture()
    first = module.raw_loading_gate(x, root, inverse)
    assert first == module.raw_loading_gate(x, root, inverse)
    assert first[1]["fold_policy"] == "input_row_index_modulo_3"
    assert first[1]["inner_fold_ids"] == (np.arange(len(x)) % 3).tolist()


def test_preserved_fold_labels_are_row_permutation_equivariant_to_roundoff():
    x, root, inverse = mismatch_fixture()
    folds = np.arange(len(x)) % 3
    adopt, before = module.raw_loading_gate(x, root, inverse, inner_fold_ids=folds)
    order = np.random.default_rng(904).permutation(len(x))
    adopted, after = module.raw_loading_gate(x[order], root, inverse, inner_fold_ids=folds[order])
    assert adopt == adopted
    np.testing.assert_allclose(np.array(after["per_gene_log_residual_gain"])[np.argsort(order)],
                               before["per_gene_log_residual_gain"], atol=2e-12, rtol=2e-11)
    for original, changed in zip(before["inner_folds"], after["inner_folds"], strict=True):
        np.testing.assert_allclose(original["loading"], changed["loading"], atol=2e-11)


def test_nonconvergence_flags_retained_and_prevent_adoption(monkeypatch):
    noise, effect, ell, root, inverse = shared_fixture()
    diagnostics = {"converged": False, "iterations": 100, "relative_change": .1,
                   "other_solver_diagnostic": {"kept": True}}
    snapshot = deepcopy(diagnostics)
    calls = []

    def failing_fit(*args):
        calls.append(True)
        return ell, diagnostics

    monkeypatch.setattr(module, "learned_loading", failing_fit)
    adopt, info = module.raw_loading_gate(noise+effect*ell, root, inverse, include_old_gate=True)
    assert len(calls) == 3 and not adopt and not info["old_gate"]["adopt"]
    assert info["mean_log_residual_gain"] > 1  # favorable scores cannot override failed fits
    assert all(fit == snapshot for fit in info["inner_fits"])
    assert all(entry["loading_fit"] == snapshot for entry in info["inner_folds"])
    assert diagnostics == snapshot
    info["inner_fits"][0]["other_solver_diagnostic"]["kept"] = False
    assert diagnostics == snapshot
    assert info["inner_folds"][0]["loading_fit"] == snapshot


def test_numerical_failures_preserved_all_folds_attempted_and_json_safe(monkeypatch):
    x, root, inverse = mismatch_fixture()
    calls = []

    def broken_fit(*args):
        calls.append(True)
        raise np.linalg.LinAlgError("fixture singular scatter")

    monkeypatch.setattr(module, "learned_loading", broken_fit)
    adopt, info = module.raw_loading_gate(x, root, inverse, include_old_gate=True)
    assert not adopt and len(calls) == 3 and info["numerical_failure_count"] == 3
    assert info["mean_log_residual_gain"] is None
    assert all(entry["error_message"] == "fixture singular scatter" for entry in info["inner_folds"])
    assert not info["old_gate"]["adopt"]
    json.dumps(info, allow_nan=False)


def test_zero_input_failure_is_not_silently_dropped():
    _, root, inverse = pipeline_roots(.8, 6)
    adopt, info = module.raw_loading_gate(np.zeros((24, 4, 6)), root, inverse)
    assert not adopt and info["numerical_failure_count"] == 3
    assert all(not fit["converged"] for fit in info["inner_fits"])
    json.dumps(info, allow_nan=False)


def test_optional_legacy_numeric_failure_does_not_change_raw_gate(monkeypatch):
    noise, effect, ell, root, inverse = shared_fixture()
    x = noise + effect*ell
    raw_adopt, raw_info = module.raw_loading_gate(x, root, inverse)

    def broken_legacy_score(*args):
        raise FloatingPointError("fixture legacy-only overflow")

    monkeypatch.setattr(module, "residual_profile", broken_legacy_score)
    adopt, info = module.raw_loading_gate(x, root, inverse, include_old_gate=True)
    assert adopt == raw_adopt and adopt
    assert info["mean_log_residual_gain"] == raw_info["mean_log_residual_gain"]
    assert info["numerical_failure_count"] == 0
    assert not info["old_gate"]["adopt"]
    assert info["old_gate"]["numerical_failure_count"] == 3


@pytest.mark.parametrize("bad", ["small", "studies", "pipelines", "nan", "complex",
                                  "root_shape", "root_not_spd", "not_inverse", "bool_option",
                                  "fold_shape", "fold_float", "fold_missing", "fold_empty"])
def test_invalid_inputs_rejected(bad):
    x, root, inverse = mismatch_fixture()
    kwargs = {}
    if bad == "small":
        x = x[:11]
    elif bad == "studies":
        x = x[:, :3]
    elif bad == "pipelines":
        x = x[..., :2]
    elif bad == "nan":
        x[0, 0, 0] = np.nan
    elif bad == "complex":
        x = x.astype(complex)
    elif bad == "root_shape":
        root = np.eye(5)
    elif bad == "root_not_spd":
        root = -root
    elif bad == "not_inverse":
        inverse *= 2
    elif bad == "bool_option":
        kwargs["include_old_gate"] = 1
    elif bad == "fold_shape":
        kwargs["inner_fold_ids"] = np.zeros(len(x)-1, dtype=int)
    elif bad == "fold_float":
        kwargs["inner_fold_ids"] = (np.arange(len(x)) % 3).astype(float)
    elif bad == "fold_missing":
        kwargs["inner_fold_ids"] = np.arange(len(x)) % 2
    elif bad == "fold_empty":
        kwargs["inner_fold_ids"] = np.full(len(x), 3, dtype=int)
    with pytest.raises((ValueError, TypeError)):
        module.raw_loading_gate(x, root, inverse, **kwargs)


@pytest.mark.parametrize("rho", [0., .1, .8])
def test_cs_noise_penalty_and_shared_signal_expected_sse_identity(rho):
    # Exact trace algebra, not a Monte Carlo estimate and not a log-gain proof.
    ell = np.array([.2, .5, .8, 1.2, 1.5, 1.8])
    u, v = ell/np.linalg.norm(ell), np.ones(6)/np.sqrt(6)
    covariance, _, _ = pipeline_roots(rho, 6)
    difference = np.trace((np.eye(6)-np.outer(v, v)) @ covariance) - np.trace(
        (np.eye(6)-np.outer(u, u)) @ covariance)
    penalty = rho*((np.ones(6) @ u)**2-6)
    assert difference == pytest.approx(penalty, abs=2e-14)
    assert penalty <= 0
    beta = 4
    signal_gain = beta**2*((u @ ell)**2-(v @ ell)**2)
    assert signal_gain + penalty > 0  # sufficiently strong shared ell can overcome cost


def test_new_audit_is_only_an_audit_not_a_geometry_or_testing_rule():
    x, root, inverse = mismatch_fixture()
    _, info = module.raw_loading_gate(x, root, inverse, include_old_gate=True)
    assert not info["intersection_applied"]
    assert not info["uses_gls_reconstruction_weights"]
    assert not info["refits_calibration_rho"]
    assert info["requires_outer_training_only"]
    assert not any(key in info for key in ["p_value", "e_value", "truth", "projection_weights"])
