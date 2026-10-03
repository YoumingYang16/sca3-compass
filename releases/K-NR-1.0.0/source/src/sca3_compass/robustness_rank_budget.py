"""Exploratory TRAIN-only categorical count-reliability calibration.

No HELD probabilities or labels enter fitting. The iid categorical working
model predicts threshold counts, NOT true positives or the full focused-eBH
utility. Conditional superuniformity of the HELD p-value is still required;
this module neither establishes fitted-nuisance FDR nor claims originality.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .robustness_calibrators import focused_calibrator

CONCENTRATION_FRACTION = .8
JEFFREYS_PSEUDOCOUNT = .5
POLICY = "training_categorical_rank_budget_v1"

__all__ = [
    "CONCENTRATION_FRACTION",
    "JEFFREYS_PSEUDOCOUNT",
    "POLICY",
    "apply_rank_budget",
    "categorical_count_law",
    "fit_rank_budget",
    "rank_budget_calibrator",
    "rank_threshold",
]


def _array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot be masked")
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise TypeError(f"{name} must contain real, non-boolean numbers")
    if not raw.size or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
        raise ValueError(f"{name} must be nonempty, finite probabilities in [0, 1]")
    with np.errstate(over="ignore", invalid="ignore"):
        array = raw.astype(np.float64, copy=False)
    if not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be nonempty and finite in float64")
    if np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return array


def _integer(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not boolean")
    number = int(value)
    if number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _genes(value: Any) -> int:
    number = _integer(value, "family_gene_count")
    if number > min((2**53 - 1) // 2, (np.iinfo(np.intp).max - 1) // 2):
        raise ValueError("family_gene_count exceeds exact float64/indexable family size")
    return number


def _level(value: Any) -> float:
    if np.ma.isMaskedArray(value):
        raise ValueError("alpha cannot be masked")
    raw = np.asarray(value)
    if raw.ndim or raw.dtype.kind not in "fiu":
        raise TypeError("alpha must be a real, non-boolean scalar")
    number = float(raw)
    if not np.isfinite(number) or not 0 < number < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    return number


def _configuration(genes: Any, alpha: Any) -> tuple[int, int, float]:
    g, level = _genes(genes), _level(alpha)
    cap = max(1, g // 4)
    harmonic = np.sum(1. / np.arange(1, cap + 1))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scale = 2 * g * harmonic / level
    if not np.isfinite(scale):
        raise ValueError("family/alpha gives an unsafe float64 calibrator scale")
    return g, cap, level


def _threshold(rank: int, genes: int, alpha: float) -> tuple[float, dict[str, Any]]:
    family, fraction = 2 * genes, CONCENTRATION_FRACTION
    nominal = alpha * fraction * rank / family
    target = family / (alpha * rank)  # Match the existing eBH operation order.
    bound = Fraction.from_float(alpha) * Fraction.from_float(fraction) * rank / family
    tau, corrections = nominal, 0
    if not 0 < tau < 1 or not np.isfinite(target):
        raise ValueError("No safe positive float64 rank threshold")
    # No held values, epsilon inflation of evidence, or label-dependent fix.
    # Round down from the nominal expression until BOTH conditions hold.
    while Fraction.from_float(tau) > bound or fraction / tau < target:
        tau = float(np.nextafter(tau, 0.))
        corrections += 1
        if tau <= 0 or corrections > 16:
            raise ArithmeticError("Unable to represent a safe rank threshold within 16 ulps")
    height = fraction / tau
    if not np.isfinite(height):
        raise ValueError("Nonfinite concentrated evidence at rank threshold")
    return tau, {
        "rank": rank, "threshold": tau, "nominal_threshold": nominal,
        "threshold_hex": tau.hex(), "nominal_threshold_hex": nominal.hex(),
        "ulp_downward_corrections": corrections,
        "step_evidence_height": height, "ebh_rank_cutoff": target,
        "exact_float_input_rational_upper_bound_enforced": True,
        "rounding_rule": "nextafter_toward_zero_only_until_rational_bound_and_eheight_hold",
    }


def rank_threshold(
    rank: int, family_gene_count: int, alpha: float = .05,
) -> tuple[float, dict[str, Any]]:
    """Frozen rank boundary, including a downward-only float64 correction.

    Ranks are 1..max(1,G//4). The mathematical bound uses the exact rational
    values of the supplied float64 alpha and the fixed float64 fraction .8.
    The returned step height also meets the actual existing float64 eBH cutoff.
    """
    genes, cap, level = _configuration(family_gene_count, alpha)
    number = _integer(rank, "rank")
    if number > cap:
        raise ValueError("rank exceeds max(1, family_gene_count//4)")
    return _threshold(number, genes, level)


def _support(q: NDArray[np.float64], genes: int) -> NDArray[np.bool_]:
    """Exact combinatorial support, including the {0,2} parity-only case."""
    possible = np.flatnonzero(q > 0)
    indices = np.arange(2 * genes + 1)
    if len(possible) == 1:
        return indices == genes * possible[0]
    mask = (indices >= genes * possible[0]) & (indices <= genes * possible[-1])
    if np.array_equal(possible, [0, 2]):
        mask &= indices % 2 == 0
    return mask


def categorical_count_law(
    q: ArrayLike, family_gene_count: int,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Coefficients of (q0+q1*x+q2*x**2)**G by direct convolution.

    Algebraically exact finite distribution, evaluated in float64, NOT exact
    rational arithmetic. No FFT, negative-coefficient clipping or Monte Carlo.
    Mass is normalized at every multiplication; all drifts, numerical zeros
    on the mathematical support and subnormal coefficients are disclosed.
    Returned index k is Pr(sum of G iid counts equals k), length 2G+1.
    """
    genes = _genes(family_gene_count)
    probabilities = _array(q, "q")
    if probabilities.shape != (3,):
        raise ValueError("q must have shape (3,) for counts 0, 1, 2")
    total = math.fsum(probabilities.tolist())
    if abs(total - 1.) > 32 * np.finfo(float).eps:
        raise ValueError("q must sum to one within 32 float64 epsilons")
    input_q = probabilities.tolist()
    probabilities = probabilities / total
    operations: list[dict[str, Any]] = []

    def multiply(left, left_n, right, right_n):
        raw = np.convolve(left, right)
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ArithmeticError("Convolution produced negative or nonfinite coefficients")
        mass = math.fsum(raw.tolist())
        if mass <= 0 or abs(mass - 1.) > 1e-10:
            raise ArithmeticError("Convolution probability mass outside numerical tolerance")
        result = raw / mass
        n = left_n + right_n
        possible = _support(probabilities, n)
        operations.append({
            "left_gene_count": left_n, "right_gene_count": right_n,
            "output_gene_count": n, "mass_before_normalization": mass,
            "normalization_divisor": mass,
            "mass_after_normalization": math.fsum(result.tolist()),
            "zero_coefficients_on_mathematical_support": int(np.sum(possible & (result == 0))),
            "subnormal_positive_coefficients": int(np.sum((result > 0) & (result < np.finfo(float).tiny))),
        })
        return result

    law, law_n = np.ones(1), 0
    power, power_n = probabilities, 1
    remaining = genes
    while remaining:
        if remaining & 1:
            law = multiply(law, law_n, power, power_n)
            law_n += power_n
        remaining >>= 1
        if remaining:
            power = multiply(power, power_n, power, power_n)
            power_n *= 2
    if law_n != genes or len(law) != 2 * genes + 1:
        raise ArithmeticError("Internal polynomial degree mismatch")
    possible = _support(probabilities, genes)
    zero_count = int(np.sum(possible & (law == 0)))
    indices = np.arange(len(law), dtype=float)
    theoretical_mean = genes * (probabilities[1] + 2 * probabilities[2])
    numeric_mean = math.fsum((indices * law).tolist())
    return law, {
        "algorithm": "direct_nonnegative_convolution_binary_exponentiation_float64",
        "family_gene_count": genes, "q_input": input_q,
        "q_normalized": probabilities.tolist(), "q_input_mass": total,
        "operations": operations, "convolution_count": len(operations),
        "normalization_performed_each_convolution": True,
        "max_absolute_mass_drift": max(abs(row["mass_before_normalization"] - 1.) for row in operations),
        "mass": math.fsum(law.tolist()), "minimum_coefficient": float(law.min()),
        "negative_coefficient_count": int(np.sum(law < 0)),
        "structural_zero_count": int(np.sum(~possible)),
        "zero_coefficients_on_mathematical_support": zero_count,
        "intermediate_underflow_zero_count_max": max(row["zero_coefficients_on_mathematical_support"] for row in operations),
        "underflow_detected": zero_count > 0 or any(row["zero_coefficients_on_mathematical_support"] > 0 for row in operations),
        "underflow_detection_scope": "zero_coefficients_on_known_support_not_all_rounded_terms",
        "subnormal_positive_coefficients": int(np.sum((law > 0) & (law < np.finfo(float).tiny))),
        "mean": numeric_mean, "theoretical_mean": float(theoretical_mean),
        "absolute_mean_error": abs(numeric_mean - float(theoretical_mean)),
        "uses_fft": False, "negative_coefficients_clipped": False,
    }


def _bh_count(training: NDArray[np.float64], alpha: float) -> int:
    ordered = np.sort(training.ravel())
    passing = np.flatnonzero(ordered <= alpha * np.arange(1, training.size + 1) / training.size)
    return int(passing[-1] + 1) if len(passing) else 0


def _training_hash(training: NDArray[np.float64]) -> str:
    value = np.ascontiguousarray(training, dtype="<f8")
    digest = sha256(json.dumps([list(value.shape), value.dtype.str]).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def fit_rank_budget(
    training_p: ArrayLike, family_gene_count: int, alpha: float = .05,
) -> dict[str, Any]:
    """Fit on TRAIN (N,2) only; G is the ACTUAL deployed gene family size.

    For every allowed rank r, form C_i in {0,1,2} at tau_r, smooth the three
    counts by +.5 each, and evaluate E[R*I(R>=r)] for G iid categorical genes.
    Exact float64 utility ties prefer smaller r (no tolerance). The zero-BH
    pilot veto disables concentration even if smoothing gives positive utility.
    TRAIN p-values may be in-sample; utility is not scientific validation.
    """
    training = _array(training_p, "training_p")
    if training.ndim != 2 or training.shape[1] != 2:
        raise ValueError("training_p must have shape (N, 2) with N>=1")
    genes, cap, level = _configuration(family_gene_count, alpha)
    pilot = _bh_count(training, level)
    fallback_threshold = .5 * level * max(1, pilot) / training.size
    if not 0 < fallback_threshold < 1 or not np.isfinite(1. / fallback_threshold):
        raise ValueError("Unsafe legacy pilot fallback threshold")
    cache = {}
    table = []
    for rank in range(1, cap + 1):
        tau, boundary = _threshold(rank, genes, level)
        counts = np.bincount(np.sum(training <= tau, axis=1), minlength=3)
        key = tuple(int(x) for x in counts)
        q = (counts + JEFFREYS_PSEUDOCOUNT) / (len(training) + 3 * JEFFREYS_PSEUDOCOUNT)
        reused = key in cache
        if not reused:
            cache[key] = categorical_count_law(q, genes)
        law, diagnostics = cache[key]
        # fsum avoids a cumsum rounding a small tail away by subtraction from 1.
        tail = math.fsum(law[rank:].tolist())
        utility = math.fsum((np.arange(rank, len(law), dtype=float) * law[rank:]).tolist())
        table.append({
            **boundary, "category_counts": list(key), "q": q.tolist(),
            "mean_count": diagnostics["mean"],
            "tail_probability": tail, "expected_count_above_rank": utility,
            "utility": utility, "law_diagnostics": deepcopy(diagnostics),
            "law_reused_for_identical_category_counts": reused,
        })
    winner = min(table, key=lambda row: (-row["utility"], row["rank"]))
    return {
        "version": POLICY, "status": "exploratory_training_utility_only",
        "training_gene_count": len(training), "training_signed_count": training.size,
        "training_sha256": _training_hash(training),
        "training_hash_rule": "sha256_json_shape_dtype_then_C_order_little_endian_float64_bytes",
        "family_gene_count": genes, "family": 2 * genes, "cap": cap,
        "alpha": level, "pilot_level": level, "pilot_discoveries": pilot,
        "pseudocount_per_category": JEFFREYS_PSEUDOCOUNT,
        "smoothing": "Jeffreys_Dirichlet_posterior_mean_plug_in_not_posterior_predictive",
        "requested_concentration_fraction": CONCENTRATION_FRACTION,
        "concentration_fraction": CONCENTRATION_FRACTION if pilot else 0.,
        "zero_count_fallback": pilot == 0,
        "selected_rank": winner["rank"] if pilot else None,
        "threshold": winner["threshold"] if pilot else fallback_threshold,
        "selected_utility": winner["utility"] if pilot else None,
        "selected_has_no_training_threshold_exceedances": (
            sum(winner["category_counts"][1:]) == 0 if pilot else None
        ),
        "all_rank_candidates_have_zero_training_threshold_exceedances": all(
            sum(row["category_counts"][1:]) == 0 for row in table
        ),
        "candidates": table, "distinct_count_laws_evaluated": len(cache),
        "utility_rule": "E[R * I(R >= r)]_under_G_iid_three_category_counts",
        "tie_break": "exact_float64_utility_tie_then_lower_rank_no_tolerance",
        "zero_pilot_rule": "disable_concentration_preserve_legacy_capped_BY_no_smoothing_only_choice",
        "uses_held_probabilities_for_selection": False, "uses_truth_labels": False,
        "training_probabilities_may_be_in_sample": True,
        "models_within_gene_two_sign_count_dependence": True,
        "models_between_gene_dependence": False,
        "includes_capped_component_in_utility": False,
        "integrates_categorical_parameter_uncertainty": False,
        "independent_scientific_validation": False,
        "fitted_pvalue_fdr_proven": False, "originality_claimed": False,
    }


def _application(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise TypeError("fitted_receipt must be a mapping")
    required = (
        "version", "family_gene_count", "family", "cap", "alpha", "pilot_level",
        "training_gene_count", "training_signed_count", "pilot_discoveries",
        "selected_rank", "threshold", "concentration_fraction", "zero_count_fallback",
        "uses_held_probabilities_for_selection", "uses_truth_labels",
    )
    if any(key not in receipt for key in required):
        raise ValueError("Incomplete rank-budget fitted receipt")
    if (receipt["version"] != POLICY
            or receipt["uses_held_probabilities_for_selection"] is not False
            or receipt["uses_truth_labels"] is not False):
        raise ValueError("Unsupported policy or prohibited selection provenance flags")
    genes, cap, alpha = _configuration(receipt["family_gene_count"], receipt["alpha"])
    n = _integer(receipt["training_gene_count"], "training_gene_count")
    count = _integer(receipt["pilot_discoveries"], "pilot_discoveries", 0)
    fallback = count == 0
    if (_integer(receipt["family"], "family") != 2 * genes
            or _integer(receipt["cap"], "cap") != cap
            or _integer(receipt["training_signed_count"], "training_signed_count") != 2 * n
            or count > 2 * n or _level(receipt["pilot_level"]) != alpha
            or not isinstance(receipt["zero_count_fallback"], bool)
            or receipt["zero_count_fallback"] != fallback):
        raise ValueError("Inconsistent rank-budget dimensions or BH receipt")
    if fallback:
        if receipt["selected_rank"] is not None:
            raise ValueError("Zero-pilot fallback cannot select a concentrated rank")
        tau, boundary = .5 * alpha / (2 * n), None
        fraction = 0.
    else:
        rank = _integer(receipt["selected_rank"], "selected_rank")
        if rank > cap:
            raise ValueError("selected_rank exceeds rank cap")
        tau, boundary = _threshold(rank, genes, alpha)
        fraction = CONCENTRATION_FRACTION
    # Do not trust stored thresholds/fractions or silently repair modified receipts.
    stored_tau = _array(receipt["threshold"], "threshold")
    stored_fraction = _array(receipt["concentration_fraction"], "concentration_fraction")
    if (stored_tau.ndim or stored_fraction.ndim
            or float(stored_tau) != tau or float(stored_fraction) != fraction
            or not 0 < tau < 1 or not np.isfinite(1. / tau)):
        raise ValueError("Inconsistent or unsafe fitted threshold/fraction")
    return {
        "version": POLICY, "family_gene_count": genes, "family": 2 * genes,
        "cap": cap, "alpha": alpha, "threshold": tau,
        "concentration_fraction": fraction, "zero_count_fallback": fallback,
        "selected_rank": receipt["selected_rank"], "boundary": boundary,
        "uses_held_probabilities_for_selection": False, "uses_truth_labels": False,
        "provenance_authenticated": False,
    }


def apply_rank_budget(
    held_p: ArrayLike, fitted_receipt: Mapping[str, Any],
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Apply a frozen fitted policy; no threshold reselection from HELD.

    G comes from fitting, not held-array length. Caller retains all 2G claims
    in the eventual eBH family. Receipt consistency is checked, not its external
    provenance or utility optimality. No mutations of inputs or fitted receipt.
    """
    info = _application(fitted_receipt)
    held = _array(held_p, "held_p")
    values = focused_calibrator(
        held, info["family"], info["threshold"], info["concentration_fraction"],
        info["alpha"], cap=info["cap"],
    )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ArithmeticError("Nonfinite or negative calibrated evidence")
    return values, info


def rank_budget_calibrator(
    held_p: ArrayLike, training_p: ArrayLike, family_gene_count: int, alpha: float = .05,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Convenience wrapper: finish TRAIN fitting before inspecting HELD input."""
    fitted = fit_rank_budget(training_p, family_gene_count, alpha)
    values, application = apply_rank_budget(held_p, fitted)
    return values, {"selection": fitted, "application": application}
