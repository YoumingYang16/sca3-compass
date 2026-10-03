"""Training-only top-quartile raw loading audit; no adoption rule or FDR claim.

The caller supplies ONLY outer-training genes and frozen calibration roots.
Selection uses the norm of the raw pipeline-mean vector, not the candidate
loading. Inner validation genes do not fit their loading. Existing full raw
audit receipts can supply those same loadings, avoiding additional fitting.
"""

import math
import warnings
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256

import numpy as np

from .robustness_loading_audit_gate import (
    _checked_inputs,
    _json_safe,
    euclidean_log_residual_gain,
    raw_loading_gate,
)


def raw_mean_log_norm(training):
    """Return log(||mean_pipeline(x_g)||_2) without squaring raw magnitudes.

    Exactly zero mean vectors return -inf (a valid lowest selection score).
    Normalize each study vector before compensated summation. If a nonzero
    input underflows during normalization, or compensated cancellation is
    exactly zero, use an exact rational sum of the supplied binary floats.
    This rare fallback changes neither the observations nor the estimand.
    All fallback/underflow counts are reported. Final logs are float64.
    """
    x = np.asarray(training)
    if x.dtype.kind not in "iuf":
        raise ValueError("Finite real G x S x K training required")
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or min(x.shape) < 1 or not np.isfinite(x).all():
        raise ValueError("Finite nonempty G x S x K training required")
    logs = np.full(x.shape[:2], -np.inf)
    underflows = exact = cancellations = zeros = 0
    for g, gene in enumerate(x):
        for s, row in enumerate(gene):
            scale = float(np.max(np.abs(row)))
            if scale == 0:
                zeros += 1
                continue
            with np.errstate(under="ignore"):
                normalized = row / scale
            lost = int(np.sum((row != 0) & (normalized == 0)))
            underflows += lost
            total = math.fsum(normalized.tolist())
            if lost or total == 0:
                exact += 1
                value = sum((Fraction(float(v)) for v in row), Fraction())
                if value == 0:
                    cancellations += 1
                    continue
                logs[g, s] = (math.log(abs(value.numerator))
                              - math.log(value.denominator) - math.log(x.shape[-1]))
            else:
                logs[g, s] = math.log(scale) + math.log(abs(total)) - math.log(x.shape[-1])
    scores = np.full(len(x), -np.inf)
    for g, row in enumerate(logs):
        finite = row[np.isfinite(row)]
        if len(finite):
            peak = float(finite.max())
            # Underflow here drops contributions below float64 resolution,
            # not observations from the selected-gene count.
            scores[g] = peak + .5 * math.log(math.fsum(
                math.exp(2 * (float(value) - peak)) for value in finite))
    if np.isnan(scores).any() or np.isposinf(scores).any():
        raise FloatingPointError("Invalid raw mean log-norm")
    return scores, {
        "score": "log_l2_norm_of_raw_pipeline_mean_vector_across_studies",
        "zero_input_study_vector_count": zeros,
        "exact_zero_cancellation_study_vector_count": cancellations,
        "normalization_underflow_value_count": underflows,
        "exact_binary_sum_fallback_study_vector_count": exact,
        "zero_mean_gene_count": int(np.sum(np.isneginf(scores))),
        "uses_loading": False,
        "raw_values_clipped_or_deleted": False,
    }


def _select_quartiles(scores, folds):
    selected = np.zeros(len(scores), dtype=bool)
    entries = []
    for fold in range(3):
        indices = np.flatnonzero(folds == fold)
        count = (len(indices) + 3) // 4
        # indices are ascending; stable sorting resolves exact ties by input
        # row index. No tolerance or loading-dependent tie-breaking.
        order = np.argsort(-scores[indices], kind="stable")
        chosen = indices[order[:count]]
        selected[chosen] = True
        boundary = scores[chosen[-1]]
        ties = scores[indices] == boundary
        entries.append({
            "fold": fold, "held_indices": indices.tolist(),
            "held_genes": len(indices), "selected_count": count,
            "selected_indices": sorted(chosen.tolist()),
            "selected_indices_in_rank_order": chosen.tolist(),
            "boundary_log_norm": float(boundary),
            "boundary_tied_count": int(ties.sum()),
            "boundary_tied_selected_count": int(np.sum(scores[chosen] == boundary)),
        })
    return selected, entries


def _checked_receipt(receipt, x, folds):
    """Check structure, not unobservable provenance of old learned loadings."""
    if not isinstance(receipt, Mapping):
        raise TypeError("raw_audit must be a full raw_loading_gate receipt")
    if receipt.get("gate_version") != "raw_euclidean_inner3_v1":
        raise ValueError("Unsupported or compact raw audit receipt")
    for field in ("all_scores_finite", "all_inner_fits_converged"):
        if not isinstance(receipt.get(field), (bool, np.bool_)):
            raise TypeError(f"Raw audit lacks boolean {field}")
    failure_count = receipt.get("numerical_failure_count")
    if (isinstance(failure_count, (bool, np.bool_))
            or not isinstance(failure_count, (int, np.integer)) or failure_count < 0):
        raise ValueError("Raw numerical-failure count must be a nonnegative integer")
    if not np.array_equal(np.asarray(receipt.get("inner_fold_ids")), folds):
        raise ValueError("Raw audit inner-fold IDs differ from the supplied training contract")
    gains = receipt.get("per_gene_log_residual_gain")
    if not isinstance(gains, (list, tuple, np.ndarray)) or len(gains) != len(x):
        raise ValueError("Full per-gene raw scores required; compact receipts are not reusable")
    if np.asarray(gains, dtype=float).shape != (len(x),):
        raise ValueError("Raw scores must be a scalar per gene, not a broadcastable matrix")
    entries = receipt.get("inner_folds")
    if not isinstance(entries, (list, tuple)) or len(entries) != 3:
        raise ValueError("Exactly three raw inner-fold records required")
    by_fold = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Invalid raw inner-fold record")
        fold = entry.get("fold")
        if isinstance(fold, (bool, np.bool_)) or not isinstance(fold, (int, np.integer)):
            raise TypeError("Raw fold number must be an integer")
        if fold not in (0, 1, 2) or fold in by_fold:
            raise ValueError("Raw inner folds must be distinct 0, 1, 2")
        if (entry.get("held_genes") != int(np.sum(folds == fold))
                or entry.get("training_genes") != int(np.sum(folds != fold))):
            raise ValueError("Raw inner-fold counts do not match training")
        if not isinstance(entry.get("loading_fit"), Mapping):
            raise TypeError("Every raw inner fold must retain its loading-fit diagnostics")
        if not isinstance(entry.get("numerical_failure"), (bool, np.bool_)):
            raise TypeError("Every raw inner fold must retain its numerical-failure flag")
        by_fold[fold] = entry
    return deepcopy(dict(receipt)), by_fold


def _warning_receipt(caught):
    return [{"category": value.category.__name__, "message": str(value.message),
             "filename": value.filename, "lineno": value.lineno,
             "runtime_warning": issubclass(value.category, RuntimeWarning)} for value in caught]


def focused_loading_audit(training, root, inverse_root, *, raw_audit=None,
                          inner_fold_ids=None):
    """Return focused positive/negative heuristic evidence and a full receipt.

    Within each of three inner-validation folds select ceil(N_fold/4) genes
    by descending raw pipeline-mean norm. Average the existing raw Euclidean
    study log-SSE ratios WITHIN each selected gene, then pool selected genes
    with equal gene weights. The sample SD/sqrt(N_selected) is a descriptive
    clustered SE, not a calibrated inferential standard error.

    Either generate the ordinary raw audit on ONLY these outer-training
    genes or reuse a full supplied receipt from exactly the same inputs and
    frozen roots. Reuse performs zero fits and cross-checks recomputed raw
    scores. Old receipts contain no data/root hash: the caller remains
    responsible for their provenance; score agreement cannot prove it.

    Positive: mean>.05 and mean>2.58*SE. Negative: mean<-.05 and
    mean<-2.58*SE. Both require audit_valid. Any failed/nonconverged raw fit
    or raw-score inconsistency invalidates the whole audit; no bad fold is
    dropped and no missing loading is refitted. This module DOES NOT combine
    bulk and focused evidence or modify a loading-adoption decision.
    """
    x, root, inverse, folds, fold_policy = _checked_inputs(
        training, root, inverse_root, inner_fold_ids, False)
    selection_score, selection_quality = raw_mean_log_norm(x)
    selected, selection_entries = _select_quartiles(selection_score, folds)
    generated = raw_audit is None
    generation_failure = None
    generation_warnings = []
    if generated:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                _, raw_audit = raw_loading_gate(x, root, inverse, inner_fold_ids=folds)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
                generation_failure = {"error_type": type(error).__name__,
                                      "error_message": str(error)}
        generation_warnings = _warning_receipt(caught)
    if generation_failure is None:
        source, by_fold = _checked_receipt(raw_audit, x, folds)
    else:
        source, by_fold = None, {}
    scores = np.full(len(x), np.nan)
    entries, failures, matches, fits_ok = [], 0, True, True
    for selection_entry in selection_entries:
        fold = selection_entry["fold"]
        held = folds == fold
        entry = deepcopy(selection_entry)
        raw = by_fold.get(fold)
        entry["source_raw_fold"] = deepcopy(raw)
        entry["numerical_failure"] = False
        converged = raw["loading_fit"].get("converged") if raw is not None else False
        fit_ok = (isinstance(converged, (bool, np.bool_)) and bool(converged)
                  and raw is not None
                  and not raw["numerical_failure"])
        fits_ok = fits_ok and fit_ok
        caught = []
        try:
            if raw is None or "loading" not in raw:
                raise ValueError("Missing raw inner loading; no refit attempted")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    study_scores, quality = euclidean_log_residual_gain(x[held], raw["loading"])
                    scores[held] = study_scores.mean(axis=1)
                    _, selected_quality = euclidean_log_residual_gain(
                        x[selected & held], raw["loading"])
            # Do not silently reuse loadings with another data ordering. The
            # tolerance allows only log-score roundoff, not a different fit.
            previous = np.asarray(source["per_gene_log_residual_gain"], dtype=float)[held]
            match = bool(np.isfinite(previous).all() and np.allclose(
                scores[held], previous, rtol=2e-12, atol=2e-12))
            matches = matches and match
            entry.update(raw_scores_match_receipt=match,
                         max_raw_score_absolute_difference=(float(np.max(np.abs(
                             scores[held] - previous))) if np.isfinite(previous).all() else None),
                         all_held_raw_score_quality=quality,
                         selected_raw_score_quality=selected_quality)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            entry.update(numerical_failure=True, error_type=type(error).__name__,
                         error_message=str(error))
            failures += 1
            matches = False
        finally:
            entry["score_warnings"] = _warning_receipt(caught)
        entries.append(entry)
    finite = bool(np.isfinite(scores).all())
    source_valid = bool(source is not None and source["all_scores_finite"]
                        and source["all_inner_fits_converged"]
                        and source["numerical_failure_count"] == 0)
    runtime_warnings = sum(e["runtime_warning"] for e in generation_warnings)
    runtime_warnings += sum(w["runtime_warning"] for e in entries for w in e.get("score_warnings", []))
    valid = bool(source_valid and fits_ok and finite and matches and failures == 0
                 and runtime_warnings == 0)
    focused = scores[selected]
    # Even if a selected subset has finite scores, never silently omit a
    # failed unselected gene or report a partial-denominator audit as valid.
    summary_finite = bool(np.isfinite(focused).all())
    mean = float(focused.mean()) if summary_finite else None
    se = float(focused.std(ddof=1) / np.sqrt(len(focused))) if summary_finite else None
    return _json_safe({
        "audit_version": "raw_euclidean_focus_quartile_inner3_v1",
        "audit_valid": valid,
        "positive_evidence": bool(valid and mean > .05 and mean > 2.58 * se),
        "negative_evidence": bool(valid and mean < -.05 and mean < -2.58 * se),
        "mean_log_residual_gain": mean,
        "gene_level_standard_error": se,
        "min_gain": .05, "standard_error_multiplier": 2.58,
        "selected_gene_count": int(selected.sum()),
        "training_gene_count": len(x),
        "selected_gene_fraction": float(selected.mean()),
        "selected_indices": np.flatnonzero(selected).tolist(),
        "selection_fraction": .25,
        "selection_count_rule": "ceil(inner_validation_gene_count / 4)",
        "selection_tie_rule": "exact_float_score_tie_then_ascending_input_row_index",
        "selection_log_norm_by_gene": selection_score.tolist(),
        "selection_zero_mean_indices": np.flatnonzero(np.isneginf(selection_score)).tolist(),
        "selection_quality": selection_quality,
        "per_gene_log_residual_gain": scores.tolist(),
        "inner_fold_ids": folds.tolist(), "fold_policy": fold_policy,
        "inner_folds": entries, "raw_audit": source,
        "raw_audit_generated_here": generated,
        "raw_audit_generation_failure": generation_failure,
        "raw_audit_generation_warnings": generation_warnings,
        "runtime_warning_count": runtime_warnings,
        "raw_scores_match_receipt": matches,
        "bulk_raw_audit_valid": source_valid,
        "all_inner_fits_converged": bool(fits_ok),
        "all_scores_finite": finite,
        "numerical_failure_count": failures,
        "training_float64_sha256": sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),
        "root_float64_sha256": sha256(np.ascontiguousarray(root).tobytes()).hexdigest(),
        "inverse_root_float64_sha256": sha256(np.ascontiguousarray(inverse).tobytes()).hexdigest(),
        "reused_receipt_provenance": "caller-asserted; structure and raw scores checked, not cryptographically authenticated",
        "requires_outer_training_only": True,
        "held_gene_excluded_from_its_inner_loading": True,
        "selection_uses_loading": False,
        "uses_truth_labels": False,
        "adoption_rule_applied": False,
        "interpretation": "selected-training-gene heuristic; SE is descriptive, not a significance claim or FDR proof",
        "statistical_validation": "not_validated",
    })
