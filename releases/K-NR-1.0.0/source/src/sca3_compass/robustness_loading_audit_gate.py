"""Training-only raw Euclidean audit of the existing loading proposal.

This module does not replace the loading learner, fit a covariance, construct
test statistics, or apply the proposed intersection with the old gate. The
caller must pass ONLY outer-training genes and independently frozen roots.
The numerical SE margin below is a heuristic, not a hypothesis test or FDR
guarantee. See docs/robustness_loading_audit_gate.md for the mechanism/limits.
"""

from copy import deepcopy

import numpy as np

from .robustness_loading import learned_loading, residual_profile


def euclidean_log_residual_gain(training, loading):
    """Return per-gene, per-study log(SSE_uniform / SSE_loading), plus QA.

    Both SSEs are orthogonal projections in RAW Euclidean pipeline space:
    SSE_u(x) = ||x - (x @ u) u||^2, u = loading / ||loading||.
    There are no GLS weights here. Per-vector scaling cancels from the
    ratio, so residuals are evaluated after safe Euclidean normalization.
    Exactly zero input vectors contribute zero gain. Residual fractions
    below float64.tiny are floored for a finite logarithm and counted.
    """
    x = np.asarray(training)
    ell = np.asarray(loading)
    if x.dtype.kind not in "iuf" or ell.dtype.kind not in "iuf":
        raise ValueError("Finite real training vectors and positive loading required")
    x, ell = np.asarray(x, dtype=float), np.asarray(ell, dtype=float)
    if (x.ndim != 3 or x.shape[-1] < 2 or ell.shape != (x.shape[-1],)
            or not np.isfinite(x).all() or not np.isfinite(ell).all()
            or np.any(ell <= 0)):
        raise ValueError("Finite G x S x K training and positive length-K loading required")
    scale = np.max(np.abs(x), axis=-1, keepdims=True)
    nonzero = scale[..., 0] > 0
    normalized = np.divide(x, scale, out=np.zeros_like(x), where=scale > 0)
    norm = np.sqrt(np.sum(normalized**2, axis=-1, keepdims=True))
    normalized = np.divide(normalized, norm, out=np.zeros_like(x), where=norm > 0)
    unit = ell / np.max(ell)
    unit /= np.linalg.norm(unit)
    uniform = np.ones(x.shape[-1]) / np.sqrt(x.shape[-1])
    logs, clipped = [], []
    for direction in (uniform, unit):
        residual = normalized - (normalized @ direction)[..., None] * direction
        fraction = np.sum(residual**2, axis=-1)
        clipped.append(int(np.sum(nonzero & (fraction < np.finfo(float).tiny))))
        logs.append(np.log(np.maximum(fraction, np.finfo(float).tiny)))
    gain = np.where(nonzero, logs[0] - logs[1], 0.)
    if not np.isfinite(gain).all():
        raise FloatingPointError("Nonfinite raw Euclidean residual gain")
    return gain, {
        "zero_vector_count": int(np.sum(~nonzero)),
        "uniform_residual_floor_count": clipped[0],
        "loading_residual_floor_count": clipped[1],
        "residual_fraction_floor": float(np.finfo(float).tiny),
    }


def _checked_inputs(training, root, inverse_root, inner_fold_ids, include_old_gate):
    values = [np.asarray(value) for value in (training, root, inverse_root)]
    if any(value.dtype.kind not in "iuf" for value in values):
        raise ValueError("Finite real training array and frozen roots required")
    x, root, inverse = [np.asarray(value, dtype=float) for value in values]
    if (x.ndim != 3 or x.shape[0] < 12 or x.shape[1] != 4 or x.shape[2] < 3
            or not np.isfinite(x).all()):
        raise ValueError("Finite TRAIN G x 4 x K with G>=12, K>=3 required")
    k = x.shape[-1]
    for matrix in (root, inverse):
        if (matrix.shape != (k, k) or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12)
                or np.linalg.eigvalsh(matrix).min() <= 0):
            raise ValueError("Finite symmetric positive definite frozen roots required")
    if not np.allclose(root @ inverse, np.eye(k), rtol=1e-9, atol=1e-9):
        raise ValueError("Frozen root and inverse_root must be inverses")
    if not isinstance(include_old_gate, (bool, np.bool_)):
        raise TypeError("include_old_gate must be boolean")
    if inner_fold_ids is None:
        folds = np.arange(len(x)) % 3
        policy = "input_row_index_modulo_3"
    else:
        folds = np.asarray(inner_fold_ids)
        if (folds.shape != (len(x),) or folds.dtype.kind not in "iu"
                or not np.isin(folds, [0, 1, 2]).all()):
            raise ValueError("inner_fold_ids must be integer length-G labels 0, 1, 2")
        policy = "caller_supplied_0_1_2_labels_fixed_without_outcomes"
    if any(np.sum(folds == f) < 2 or np.sum(folds != f)*4 <= k for f in range(3)):
        raise ValueError("Each inner fold needs >=2 held genes and >K training vectors")
    return x, root, inverse, folds, policy


def _summary(gain, fits_ok):
    finite = bool(np.isfinite(gain).all())
    mean = float(gain.mean()) if finite else None
    se = float(gain.std(ddof=1)/np.sqrt(len(gain))) if finite else None
    adopt = bool(finite and fits_ok and mean > .05 and mean > 2.58*se)
    return {
        "adopt": adopt, "mean_log_residual_gain": mean,
        "gene_level_standard_error": se, "min_gain": .05,
        "standard_error_multiplier": 2.58, "all_scores_finite": finite,
        "all_inner_fits_converged": bool(fits_ok),
        "interpretation": "training-only prediction heuristic; not a significance claim or null proof",
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def raw_loading_gate(training, root, inverse_root, *, include_old_gate=False,
                     inner_fold_ids=None):
    """Return (raw_adopt, diagnostics); optionally report the old gate too.

    Three inner gene folds fit the SAME old learned_loading on whitened
    inner-training vectors. Raw Euclidean residual gains are computed on
    inner-held genes, averaged over studies WITHIN each gene, then assessed
    with mean > .05 and mean > 2.58 * SE_across_genes. Every fit must converge.

    Default folds are row_index % 3, deliberately matching loading_gate.
    A permutation that changes these memberships can change the result.
    Preassigned inner_fold_ids can travel with rows, but must not use labels
    or outcomes to choose folds. Inner-held data affect scores, never their
    fitted direction. No outer-held observations should be supplied at all.

    include_old_gate reuses these fits to reproduce the old whitened scoring
    rule, log(sum_study SSE_uniform / sum_study SSE_loading). It does NOT
    change raw_adopt or apply an intersection. Numerical failures are kept
    in the receipt and prevent adoption; unsuccessful fits are not omitted.
    """
    x, root, inverse, folds, policy = _checked_inputs(
        training, root, inverse_root, inner_fold_ids, include_old_gate,
    )
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        whitened = x @ inverse
    gain = np.full(len(x), np.nan)
    old_gain = np.full(len(x), np.nan)
    entries, fits, failures, old_failures = [], [], 0, 0
    for fold in range(3):
        held = folds == fold
        entry = {"fold": fold, "training_genes": int(np.sum(~held)),
                 "held_genes": int(np.sum(held)), "numerical_failure": False}
        fit = {"converged": False, "not_returned_by_learner": True}
        loading = None
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                loading, fit = learned_loading(whitened[~held], root)
                fit = deepcopy(fit)
                scores, quality = euclidean_log_residual_gain(x[held], loading)
                gain[held] = scores.mean(axis=1)
                entry.update(loading=np.asarray(loading).tolist(), raw_score_quality=quality)
        except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
            entry.update(numerical_failure=True, error_type=type(error).__name__,
                         error_message=str(error))
            failures += 1
        if include_old_gate and loading is not None:
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    old = residual_profile(whitened[held], np.ones(x.shape[-1]), inverse)
                    new = residual_profile(whitened[held], loading, inverse)
                    old_gain[held] = np.log(old/new)
                    if not np.isfinite(old_gain[held]).all():
                        raise FloatingPointError("Nonfinite legacy whitened gain")
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as error:
                entry.update(old_gate_numerical_failure=True,
                             old_gate_error_type=type(error).__name__,
                             old_gate_error_message=str(error))
                old_failures += 1
        entry["loading_fit"] = deepcopy(fit)
        entries.append(entry)
        fits.append(deepcopy(fit))
    fits_ok = failures == 0 and all(fit.get("converged", False) for fit in fits)
    info = _summary(gain, fits_ok)
    info.update(
        gate_version="raw_euclidean_inner3_v1", inner_folds=entries,
        inner_fits=fits, numerical_failure_count=failures,
        inner_fold_ids=folds.tolist(), fold_policy=policy,
        per_gene_log_residual_gain=gain.tolist(),
        score="mean_study_log_raw_orthogonal_SSE_ratio_per_gene",
        uses_gls_reconstruction_weights=False, refits_calibration_rho=False,
        requires_outer_training_only=True,
        held_gene_excluded_from_its_inner_loading=True,
        intersection_applied=False,
    )
    if include_old_gate:
        old_info = _summary(old_gain, fits_ok and old_failures == 0)
        old_info.update(inner_fits=deepcopy(fits),
                        numerical_failure_count=old_failures,
                        per_gene_log_residual_gain=old_gain.tolist(),
                        score="log_ratio_of_summed_whitened_study_residuals_per_gene")
        info["old_gate"] = old_info
    return info["adopt"], _json_safe(info)
