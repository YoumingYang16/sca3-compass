"""Bounded mechanism/contract fixtures, not a power or FDR experiment."""

import inspect
import json
import math
import warnings
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_loading_audit_gate as raw_module
from sca3_compass import robustness_loading_focus as module
from sca3_compass.molecular_envelope_benchmark import fixed_truth
from sca3_compass.robustness_loading import pipeline_roots


@pytest.fixture(autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1):
        yield


def mechanism_fixture(kind="sparse", df=5.):
    """Pre-existing seed771, changed only by prespecified scenario switches."""
    rng = np.random.default_rng(771)
    root_study = np.linalg.cholesky(.35*np.eye(4) + .65*np.ones((4, 4)))
    base = np.einsum("st,gtk->gsk", root_study, rng.normal(size=(256, 4, 6)))
    radial = np.sqrt((df-2 if df > 2 else df)/rng.chisquare(df, 256))[:, None, None]
    mu = fixed_truth(256, 4, {"effect": 3.5,
                            "replicated_fraction": .02 if kind == "sparse" else .2}, .2)
    ell = np.array([.2, .5, .8, 1.2, 1.5, 1.8]) if kind in ("sparse", "shared") else np.ones(6)
    _, actual, _ = pipeline_roots(.1 if kind == "downcov" else .8, 6)
    _, root, inverse = pipeline_roots(.95 if kind == "downcov" else .8, 6)
    return (base @ actual)*radial + mu[..., None]*ell, root, inverse


def synthetic_receipt(x, ell, folds=None):
    """Testing ONLY: prescribed directions, no claim these loadings were fit."""
    folds = np.arange(len(x)) % 3 if folds is None else folds
    gains = np.empty(len(x))
    entries = []
    for fold in range(3):
        held = folds == fold
        per_study, quality = raw_module.euclidean_log_residual_gain(x[held], ell)
        gains[held] = per_study.mean(axis=1)
        entries.append({"fold": fold, "held_genes": int(held.sum()),
                        "training_genes": int((~held).sum()), "numerical_failure": False,
                        "loading": np.asarray(ell).tolist(), "raw_score_quality": quality,
                        "loading_fit": {"converged": True, "fixture_not_fitted": True}})
    return {"gate_version": "raw_euclidean_inner3_v1", "inner_fold_ids": folds.tolist(),
            "inner_folds": entries, "all_scores_finite": True,
            "all_inner_fits_converged": True, "numerical_failure_count": 0,
            "per_gene_log_residual_gain": gains.tolist()}


def simple_fixture(g=25):
    x = np.random.default_rng(981).normal(size=(g, 4, 6))
    root = np.eye(6)
    receipt = synthetic_receipt(x, np.arange(1., 7.))
    return x, root, root.copy(), receipt


@pytest.mark.parametrize("scale", [1., 1e-280, 1e280])
def test_raw_mean_log_norm_matches_unscaled_direct_math(scale):
    x, _, _, _ = simple_fixture()
    expected = np.log(np.linalg.norm(x.mean(axis=-1), axis=1)) + np.log(scale)
    actual, qa = module.raw_mean_log_norm(x*scale)
    np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=1e-12)
    assert not qa["uses_loading"] and not qa["raw_values_clipped_or_deleted"]


@pytest.mark.parametrize("k", [3, 4, 6, 10])
def test_raw_quadratic_form_and_gene_aggregation(k):
    x = np.random.default_rng(818).normal(size=(37, 4, k))
    ell = np.linspace(.2, 1.8, k)
    receipt = synthetic_receipt(x, ell)
    actual = module.focused_loading_audit(x, np.eye(k), np.eye(k), raw_audit=receipt)
    u, v = ell/np.linalg.norm(ell), np.ones(k)/np.sqrt(k)
    expected = np.empty((len(x), 4))
    for g in range(len(x)):
        for s in range(4):
            row = x[g, s]
            expected[g, s] = np.log((row @ (np.eye(k)-np.outer(v, v)) @ row) /
                                    (row @ (np.eye(k)-np.outer(u, u)) @ row))
    gene_scores = expected.mean(axis=1)
    selected = actual["selected_indices"]
    np.testing.assert_allclose(actual["per_gene_log_residual_gain"], gene_scores, atol=1e-14)
    assert actual["mean_log_residual_gain"] == pytest.approx(gene_scores[selected].mean(), abs=1e-14)
    assert actual["gene_level_standard_error"] == pytest.approx(
        gene_scores[selected].std(ddof=1)/np.sqrt(len(selected)), abs=1e-14)
    # Four within-gene scores are NOT four independent validation replicates.
    naive_se = expected[selected].std(ddof=1)/np.sqrt(4*len(selected))
    assert not np.isclose(actual["gene_level_standard_error"], naive_se)


@pytest.mark.parametrize("g", [12, 13, 17, 25, 128])
def test_exact_quartile_counts_not_rounded_global_count(g):
    x, root, inverse, receipt = simple_fixture(g)
    info = module.focused_loading_audit(x, root, inverse, raw_audit=receipt)
    expected = 0
    for fold in info["inner_folds"]:
        count = math.ceil(fold["held_genes"]/4)
        expected += count
        assert fold["selected_count"] == len(fold["selected_indices"]) == count
    assert info["selected_gene_count"] == expected
    assert info["selection_fraction"] == .25


def test_selection_matches_direct_norm_with_stable_ties():
    x, root, inverse, _ = simple_fixture(24)
    # Distinct amplitudes and exact repeated rows give reproducible ties.
    x[:] = np.arange(1., 7.)
    x *= np.repeat([1., 3., 3., 2.], 6)[:, None, None]
    receipt = synthetic_receipt(x, np.arange(1., 7.))
    info = module.focused_loading_audit(x, root, inverse, raw_audit=receipt)
    score = np.linalg.norm(x.mean(-1), axis=1)
    for entry in info["inner_folds"]:
        held = np.array(entry["held_indices"])
        expected = held[np.argsort(-score[held], kind="stable")[:entry["selected_count"]]]
        assert entry["selected_indices_in_rank_order"] == expected.tolist()
        assert entry["boundary_tied_count"] > 1
        assert entry["boundary_tied_selected_count"] == entry["selected_count"]


def test_all_zero_norm_selection_and_exact_alignment_events_preserved():
    x = np.zeros((24, 4, 4))
    receipt = synthetic_receipt(x, np.ones(4))
    info = module.focused_loading_audit(x, np.eye(4), np.eye(4), raw_audit=receipt)
    assert info["selected_indices"] == [0, 1, 2, 3, 4, 5]
    assert info["selection_log_norm_by_gene"] == [None]*24  # JSON-safe -inf
    assert info["selection_zero_mean_indices"] == list(range(24))
    assert info["mean_log_residual_gain"] == 0 and info["gene_level_standard_error"] == 0
    assert not info["positive_evidence"] and not info["negative_evidence"]
    x[:] = 1
    receipt = synthetic_receipt(x, np.ones(4))
    info = module.focused_loading_audit(x, np.eye(4), np.eye(4), raw_audit=receipt)
    assert sum(e["selected_raw_score_quality"]["uniform_residual_floor_count"]
               for e in info["inner_folds"]) == 4*info["selected_gene_count"]
    json.dumps(info, allow_nan=False)


def test_extreme_cancellation_and_subnormal_mean_do_not_become_zero():
    big, tiny = np.finfo(float).max, np.nextafter(0., 1.)
    x = np.array([[[big, tiny, -big, 0.]]]*12)
    score, qa = module.raw_mean_log_norm(x)
    expected = math.log(tiny) - math.log(4)
    np.testing.assert_allclose(score, expected, atol=2e-13)
    assert qa["normalization_underflow_value_count"] == 12
    assert qa["exact_binary_sum_fallback_study_vector_count"] == 12
    assert qa["zero_mean_gene_count"] == 0
    # Although the mathematical mean is below float64.tiny/subnormal, its
    # logarithm remains finite; zero-norm selection is not manufactured.
    assert np.isfinite(score).all()


def test_compensated_sum_preserves_small_component_and_pipeline_order():
    x = np.array([[[1e308, 1., -1e308]]]*12)
    score, _ = module.raw_mean_log_norm(x)
    np.testing.assert_allclose(score, -math.log(3), atol=2e-13)
    for order in ([2, 0, 1], [0, 2, 1], [1, 2, 0]):
        changed, _ = module.raw_mean_log_norm(x[..., order])
        np.testing.assert_array_equal(changed, score)


def test_selection_order_with_huge_and_tiny_vectors():
    base = np.broadcast_to(np.array([1., 2., 3., 4., 5., 6.]), (24, 4, 6))
    scales = np.logspace(-300, 300, 24)
    x = base * scales[:, None, None]
    receipt = synthetic_receipt(x, np.arange(6., 0., -1.))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        info = module.focused_loading_audit(x, np.eye(6), np.eye(6), raw_audit=receipt)
    assert info["audit_valid"]
    assert info["selected_indices"] == list(range(18, 24))
    assert np.isfinite(info["selection_log_norm_by_gene"]).all()
    json.dumps(info, allow_nan=False)


def test_selection_independent_of_candidate_direction():
    x, root, inverse, first = simple_fixture()
    second = synthetic_receipt(x, np.arange(6., 0., -1.))
    a = module.focused_loading_audit(x, root, inverse, raw_audit=first)
    b = module.focused_loading_audit(x, root, inverse, raw_audit=second)
    assert a["selected_indices"] == b["selected_indices"]
    assert a["selection_log_norm_by_gene"] == b["selection_log_norm_by_gene"]
    assert a["mean_log_residual_gain"] != b["mean_log_residual_gain"]


def test_reuse_calls_no_learner_and_preserves_same_inner_loading(monkeypatch):
    x, root, inverse = mechanism_fixture()
    train = x[1::2]
    _, raw = raw_module.raw_loading_gate(train, root, inverse, include_old_gate=True)

    def forbidden(*args, **kwargs):
        pytest.fail("Receipt reuse must perform no extra fits")

    monkeypatch.setattr(raw_module, "learned_loading", forbidden)
    monkeypatch.setattr(module, "raw_loading_gate", forbidden)
    info = module.focused_loading_audit(train, root, inverse, raw_audit=raw)
    assert not info["raw_audit_generated_here"] and info["raw_scores_match_receipt"]
    for entry in info["inner_folds"]:
        assert entry["source_raw_fold"] == raw["inner_folds"][entry["fold"]]
    assert info["raw_audit"] == raw


def test_generated_audit_calls_exact_three_train_excluding_fits(monkeypatch):
    x, root, inverse = mechanism_fixture()
    train = x[1::2]
    calls, learner = [], raw_module.learned_loading

    def recording(data, frozen):
        calls.append(data.copy())
        return learner(data, frozen)

    monkeypatch.setattr(raw_module, "learned_loading", recording)
    info = module.focused_loading_audit(train, root, inverse)
    assert len(calls) == 3 and info["audit_valid"]
    for fold, call in enumerate(calls):
        np.testing.assert_array_equal(call, (train @ inverse)[np.arange(len(train)) % 3 != fold])


def test_generated_and_reused_receipts_have_exact_same_focused_summary():
    x, root, inverse = mechanism_fixture()
    a = module.focused_loading_audit(x[1::2], root, inverse)
    b = module.focused_loading_audit(x[1::2], root, inverse, raw_audit=a["raw_audit"])
    a.pop("raw_audit_generated_here")
    b.pop("raw_audit_generated_here")
    assert a == b


def test_changed_inner_validation_cannot_change_its_fitted_direction():
    x, root, inverse = mechanism_fixture()
    train = x[1::2].copy()
    before = module.focused_loading_audit(train, root, inverse)
    train[np.arange(len(train)) % 3 == 0] += np.arange(6.)*8
    after = module.focused_loading_audit(train, root, inverse)
    for key in ("loading", "loading_fit"):
        assert before["inner_folds"][0]["source_raw_fold"][key] == after["inner_folds"][0]["source_raw_fold"][key]
    assert before["selection_log_norm_by_gene"] != after["selection_log_norm_by_gene"]


def test_outer_holdout_and_truth_absent_from_api_and_cannot_change_result():
    x, root, inverse = mechanism_fixture()
    before = module.focused_loading_audit(x[1::2], root, inverse)
    x[::2] *= 1e12
    after = module.focused_loading_audit(x[1::2], root, inverse)
    assert before == after
    assert list(inspect.signature(module.focused_loading_audit).parameters) == [
        "training", "root", "inverse_root", "raw_audit", "inner_fold_ids"]
    assert not after["adoption_rule_applied"] and not after["uses_truth_labels"]


def test_inputs_readonly_and_diagnostics_are_owned_copies():
    x, root, inverse, raw = simple_fixture()
    snapshot = deepcopy(raw)
    folds = np.arange(len(x)) % 3
    arrays = [x, root, inverse, folds]
    snapshots = [a.copy() for a in arrays]
    for a in arrays:
        a.flags.writeable = False
    info = module.focused_loading_audit(x, root, inverse, raw_audit=raw, inner_fold_ids=folds)
    info["raw_audit"]["inner_folds"][0]["loading"][0] = 900
    info["inner_folds"][1]["source_raw_fold"]["loading_fit"]["converged"] = False
    assert raw == snapshot
    for a, b in zip(arrays, snapshots, strict=True):
        np.testing.assert_array_equal(a, b)


def test_prespecified_folds_travel_with_permutation_without_score_ties():
    x, root, inverse, raw = simple_fixture(128)
    folds = np.arange(len(x)) % 3
    order = np.random.default_rng(991).permutation(len(x))
    a = module.focused_loading_audit(x, root, inverse, raw_audit=raw, inner_fold_ids=folds)
    permuted = synthetic_receipt(x[order], np.arange(1., 7.), folds[order])
    b = module.focused_loading_audit(x[order], root, inverse, raw_audit=permuted, inner_fold_ids=folds[order])
    assert sorted(order[b["selected_indices"]].tolist()) == a["selected_indices"]
    assert a["mean_log_residual_gain"] == pytest.approx(b["mean_log_residual_gain"], abs=1e-14)


@pytest.mark.parametrize("failure", ["fit", "raw_flag", "top_fit", "top_score", "missing_loading"])
def test_invalid_audit_cannot_produce_positive_or_negative_evidence(failure):
    x, root, inverse, raw = simple_fixture()
    if failure == "fit":
        raw["inner_folds"][0]["loading_fit"]["converged"] = False
        raw["inner_folds"][0]["loading_fit"]["retained_failure_details"] = {"score": .01}
    elif failure == "raw_flag":
        raw["inner_folds"][0]["numerical_failure"] = True
    elif failure == "top_fit":
        raw["all_inner_fits_converged"] = False
    elif failure == "top_score":
        raw["all_scores_finite"] = False
    else:
        del raw["inner_folds"][0]["loading"]
    info = module.focused_loading_audit(x, root, inverse, raw_audit=raw)
    assert not info["audit_valid"] and not info["positive_evidence"] and not info["negative_evidence"]
    assert info["raw_audit"] == raw
    assert info["selected_gene_count"] == sum((np.sum(np.arange(len(x)) % 3 == f)+3)//4 for f in range(3))
    json.dumps(info, allow_nan=False)


def test_mismatched_data_receipt_fails_closed_without_refitting():
    x, root, inverse, raw = simple_fixture()
    x[0] += 5
    info = module.focused_loading_audit(x, root, inverse, raw_audit=raw)
    assert not info["raw_scores_match_receipt"] and not info["audit_valid"]
    assert not info["positive_evidence"] and not info["negative_evidence"]


def test_failed_unselected_gene_not_removed_from_validity_check():
    x, root, inverse, raw = simple_fixture()
    original = module.focused_loading_audit(x, root, inverse, raw_audit=raw)
    unselected = next(i for i in range(len(x)) if i not in original["selected_indices"])
    raw["per_gene_log_residual_gain"][unselected] = None
    info = module.focused_loading_audit(x, root, inverse, raw_audit=raw)
    assert np.isfinite(info["mean_log_residual_gain"])
    assert not info["audit_valid"]
    assert info["selected_indices"] == original["selected_indices"]


def test_generated_whitening_overflow_is_explicit_fail_closed():
    x = np.full((24, 4, 6), np.finfo(float).max)
    root, inverse = np.eye(6)/2, np.eye(6)*2
    info = module.focused_loading_audit(x, root, inverse)
    assert not info["audit_valid"]
    assert info["raw_audit_generation_failure"]["error_type"] == "FloatingPointError"
    assert info["numerical_failure_count"] == 3
    assert all("Missing raw inner loading" in e["error_message"] for e in info["inner_folds"])
    json.dumps(info, allow_nan=False)


def test_generated_learner_failures_not_retried_or_omitted(monkeypatch):
    calls = []

    def failing(*args):
        calls.append(1)
        raise np.linalg.LinAlgError("fixture failure, retained")

    monkeypatch.setattr(raw_module, "learned_loading", failing)
    x, root, inverse, _ = simple_fixture()
    info = module.focused_loading_audit(x, root, inverse)
    assert len(calls) == 3 and not info["audit_valid"]
    assert info["raw_audit"]["numerical_failure_count"] == 3
    assert all(e["source_raw_fold"]["error_message"] == "fixture failure, retained" for e in info["inner_folds"])
    assert info["mean_log_residual_gain"] is None


@pytest.mark.parametrize("bad", ["type", "version", "scores_length", "compact", "fold_ids",
                                  "duplicate_fold", "fold_count", "missing_fit", "missing_failure",
                                  "top_score_flag", "fold_float", "fold_bool"])
def test_malformed_receipts_rejected(bad):
    x, root, inverse, raw = simple_fixture()
    if bad == "type":
        raw = []
    elif bad == "version":
        raw["gate_version"] = "another_gate"
    elif bad == "scores_length":
        raw["per_gene_log_residual_gain"].pop()
    elif bad == "compact":
        raw["per_gene_log_residual_gain"] = {"hash": "not_data"}
    elif bad == "fold_ids":
        raw["inner_fold_ids"][0] = 1
    elif bad == "duplicate_fold":
        raw["inner_folds"][1]["fold"] = 0
    elif bad == "fold_count":
        raw["inner_folds"][0]["held_genes"] += 1
    elif bad == "missing_fit":
        del raw["inner_folds"][0]["loading_fit"]
    elif bad == "missing_failure":
        del raw["inner_folds"][0]["numerical_failure"]
    elif bad == "top_score_flag":
        raw["all_scores_finite"] = 1
    elif bad == "fold_float":
        raw["inner_folds"][0]["fold"] = 0.
    else:
        raw["inner_folds"][0]["fold"] = False
    with pytest.raises((ValueError, TypeError)):
        module.focused_loading_audit(x, root, inverse, raw_audit=raw)


@pytest.mark.parametrize("bad", ["short", "studies", "pipelines", "nan", "inf", "complex", "string",
                                  "root_shape", "root_not_spd", "inverse", "fold_float", "fold_missing"])
def test_invalid_training_contract_rejected(bad):
    x, root, inverse, _ = simple_fixture()
    kwargs = {}
    if bad == "short":
        x = x[:11]
    elif bad == "studies":
        x = x[:, :3]
    elif bad == "pipelines":
        x = x[..., :2]
    elif bad in ("nan", "inf"):
        x[0, 0, 0] = np.nan if bad == "nan" else np.inf
    elif bad == "complex":
        x = x.astype(complex)
    elif bad == "string":
        x = x.astype(str)
    elif bad == "root_shape":
        root = np.eye(5)
    elif bad == "root_not_spd":
        root *= -1
    elif bad == "inverse":
        inverse *= 2
    elif bad == "fold_float":
        kwargs["inner_fold_ids"] = (np.arange(len(x)) % 3).astype(float)
    else:
        kwargs["inner_fold_ids"] = np.arange(len(x)) % 2
    with pytest.raises((ValueError, TypeError)):
        module.focused_loading_audit(x, root, inverse, **kwargs)


@pytest.mark.parametrize("x", [[], np.empty((0, 4, 6)), np.zeros((4, 6)),
                               np.zeros((2, 4, 6), dtype=complex), np.full((2, 4, 6), np.inf)])
def test_invalid_mean_norm_input(x):
    with pytest.raises(ValueError):
        module.raw_mean_log_norm(x)


@pytest.mark.parametrize("fold", [0, 1])
def test_known_sparse_loading_is_positive_after_focus(fold):
    x, root, inverse = mechanism_fixture("sparse")
    train = x[np.arange(256) % 2 != fold]
    _, raw = raw_module.raw_loading_gate(train, root, inverse, include_old_gate=True)
    info = module.focused_loading_audit(train, root, inverse, raw_audit=raw)
    assert raw["old_gate"]["adopt"] and raw["mean_log_residual_gain"] < -.05
    if fold == 0:
        assert raw["mean_log_residual_gain"] < -2.58*raw["gene_level_standard_error"]
    assert info["audit_valid"] and info["positive_evidence"] and not info["negative_evidence"]
    assert info["selected_gene_count"] == 33
    assert info["mean_log_residual_gain"] == pytest.approx([.4946519031, .5310898590][fold], abs=1e-8)


@pytest.mark.parametrize("df", [5., 1.5])
@pytest.mark.parametrize("fold", [0, 1])
def test_known_downcov_loading_remains_negative_after_focus(df, fold):
    x, root, inverse = mechanism_fixture("downcov", df)
    train = x[np.arange(256) % 2 != fold]
    _, raw = raw_module.raw_loading_gate(train, root, inverse, include_old_gate=True)
    info = module.focused_loading_audit(train, root, inverse, raw_audit=raw)
    assert raw["old_gate"]["adopt"]
    assert raw["mean_log_residual_gain"] < -max(.05, 2.58*raw["gene_level_standard_error"])
    assert info["audit_valid"] and info["negative_evidence"] and not info["positive_evidence"]


@pytest.mark.parametrize("fold", [0, 1])
def test_shared_loading_false_negative_boundary_remains_recorded(fold):
    x, root, inverse = mechanism_fixture("shared")
    train = x[np.arange(256) % 2 != fold]
    _, raw = raw_module.raw_loading_gate(train, root, inverse, include_old_gate=True)
    info = module.focused_loading_audit(train, root, inverse, raw_audit=raw)
    assert raw["adopt"] == (fold == 1)
    assert info["positive_evidence"]
    assert info["raw_audit"] == raw  # old false negative not overwritten


def test_exact_binary_fallback_agrees_with_fraction_reference():
    tiny = np.nextafter(0., 1.)
    x = np.array([[[1e308, tiny, -1e308], [1e300, 2*tiny, -1e300]]])
    actual, qa = module.raw_mean_log_norm(x)
    squares = sum((sum((Fraction(float(v)) for v in row), Fraction())/3)**2 for row in x[0])
    expected = .5*(math.log(squares.numerator)-math.log(squares.denominator))
    assert actual[0] == pytest.approx(expected, abs=5e-13)
    assert qa["exact_binary_sum_fallback_study_vector_count"] == 2


def test_numpy_boolean_convergence_receipt_is_supported():
    x, root, inverse, raw = simple_fixture()
    for entry in raw["inner_folds"]:
        entry["loading_fit"]["converged"] = np.bool_(True)
    info = module.focused_loading_audit(x, root, inverse, raw_audit=raw)
    assert info["audit_valid"]
    json.dumps(info, allow_nan=False)


def test_original_failure_counter_alone_cannot_be_hidden():
    x, root, inverse, raw = simple_fixture()
    raw["numerical_failure_count"] = 1
    info = module.focused_loading_audit(x, root, inverse, raw_audit=raw)
    assert not info["bulk_raw_audit_valid"] and not info["audit_valid"]


@pytest.mark.parametrize("kind", ["generation", "score"])
def test_runtime_warnings_preserved_and_fail_closed(monkeypatch, kind):
    x, root, inverse, raw = simple_fixture()
    kwargs = {}
    if kind == "generation":
        def warned_gate(*args, **kwargs):
            warnings.warn("fixture learner roundoff", RuntimeWarning, stacklevel=1)
            return False, raw
        monkeypatch.setattr(module, "raw_loading_gate", warned_gate)
    else:
        scorer = module.euclidean_log_residual_gain

        def warned_score(*args, **kwargs):
            warnings.warn("fixture score roundoff", RuntimeWarning, stacklevel=1)
            return scorer(*args, **kwargs)
        monkeypatch.setattr(module, "euclidean_log_residual_gain", warned_score)
        kwargs["raw_audit"] = raw
    info = module.focused_loading_audit(x, root, inverse, **kwargs)
    assert not info["audit_valid"]
    assert info["runtime_warning_count"] == (1 if kind == "generation" else 6)
    if kind == "generation":
        assert info["raw_audit_generation_warnings"][0]["message"] == "fixture learner roundoff"
    else:
        assert info["inner_folds"][0]["score_warnings"][0]["message"] == "fixture score roundoff"


def test_warning_before_scoring_exception_is_not_lost(monkeypatch):
    x, root, inverse, raw = simple_fixture()

    def broken_score(*args, **kwargs):
        warnings.warn("roundoff then exception", RuntimeWarning, stacklevel=1)
        raise FloatingPointError("fixture scoring exception")

    monkeypatch.setattr(module, "euclidean_log_residual_gain", broken_score)
    info = module.focused_loading_audit(x, root, inverse, raw_audit=raw)
    assert info["runtime_warning_count"] == 3
    assert info["numerical_failure_count"] == 3
    assert not info["audit_valid"]
    assert all(e["score_warnings"][0]["message"] == "roundoff then exception" for e in info["inner_folds"])


def test_column_matrix_receipt_cannot_broadcast_into_false_agreement():
    x, root, inverse, raw = simple_fixture()
    raw["per_gene_log_residual_gain"] = np.array(raw["per_gene_log_residual_gain"])[:, None]
    with pytest.raises(ValueError, match="scalar per gene"):
        module.focused_loading_audit(x, root, inverse, raw_audit=raw)


@pytest.mark.parametrize("kind", ["sparse", "shared"])
@pytest.mark.parametrize("fold", [0, 1])
def test_infinite_variance_focus_remains_inconclusive_not_a_positive_success(kind, fold):
    x, root, inverse = mechanism_fixture(kind, 1.5)
    train = x[np.arange(256) % 2 != fold]
    _, raw = raw_module.raw_loading_gate(train, root, inverse, include_old_gate=True)
    info = module.focused_loading_audit(train, root, inverse, raw_audit=raw)
    assert raw["old_gate"]["adopt"] and info["audit_valid"]
    assert not info["positive_evidence"] and not info["negative_evidence"]
    if kind == "sparse":
        assert info["mean_log_residual_gain"] < 0
        assert raw["mean_log_residual_gain"] < -2.58*raw["gene_level_standard_error"]
    assert info["statistical_validation"] == "not_validated"
