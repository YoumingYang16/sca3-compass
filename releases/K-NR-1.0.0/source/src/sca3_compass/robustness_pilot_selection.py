"""Bounded, training-only sensitivity for the existing pilot calibrator.

Only multipliers (.25, .5, .65, .8) are supported; the concentration fraction
is always .8. This is a sidecar, not a replacement or an originality claim.
Inner validation uses precomputed TRAIN p-values, not refitted nuisance models
and not independent scientific confirmation. Conditional validity requires a
superuniform HELD p-value given *all* information used to choose its calibrator.
Neither cross-fitting nor this module proves that premise for fitted p-values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .molecular_methods import ebh
from .robustness_calibrators import focused_calibrator
from .robustness_pilot import pilot_calibrator

__all__ = [
    "CONCENTRATION_FRACTION",
    "MULTIPLIERS",
    "apply_pilot_multiplier",
    "pilot_multiplier_calibrator",
    "pilot_training_receipt",
    "select_pilot_multiplier",
    "selected_pilot_calibrator",
]

MULTIPLIERS = (.25, .5, .65, .8)
CONCENTRATION_FRACTION = .8


def _probabilities(values: ArrayLike, name: str) -> NDArray[np.float64]:
    if np.ma.isMaskedArray(values):
        raise ValueError(f"{name} cannot be masked")
    array = np.asarray(values)
    if array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must have a real, non-boolean numeric dtype")
    if not array.size or not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError(f"{name} must be nonempty, finite probabilities in [0, 1]")
    return array.astype(np.float64, copy=False)


def _scalar(value: Any, name: str) -> float:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{name} cannot be masked")
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "fiu":
        raise TypeError(f"{name} must be a real, non-boolean scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _level(value: Any) -> float:
    result = _scalar(value, "alpha")
    if not 0 < result < 1:
        raise ValueError("alpha must be in (0, 1)")
    return result


def _integer(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer, not boolean")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _multiplier(value: Any) -> float:
    result = _scalar(value, "multiplier")
    if result not in MULTIPLIERS:
        raise ValueError(f"multiplier must be one of {MULTIPLIERS}")
    return result


def _threshold(multiplier: float, alpha: float, count: int, size: int) -> float:
    # Preserve the legacy order of operations for exact multiplier=.5 parity.
    threshold = multiplier * alpha * max(1, count) / size
    if not 0 < threshold < 1 or not np.isfinite(1. / threshold):
        raise ValueError("alpha/training size gives an unsafe float64 threshold")
    return threshold


def pilot_training_receipt(training_p: ArrayLike, alpha: float = .05) -> dict[str, Any]:
    """Compute the existing BH pilot receipt once, without any HELD input.

    The fixed-calibrator API accepts any nonempty training shape, matching
    the legacy flattened BH family. The selector additionally requires (N, 2).
    The dummy probability is constant; it has no role in the pilot count.
    """
    training = _probabilities(training_p, "training_p")
    level = _level(alpha)
    # Fail early even when the pilot has no discoveries (its receipt still
    # contains a threshold). Never silently clip tiny levels or probabilities.
    _threshold(.25, level, 0, training.size)
    if not np.isfinite(2. / level):
        raise ValueError("alpha is too small for finite float64 calibration")
    _, receipt = pilot_calibrator(
        np.ones(1), training, 2, fraction=CONCENTRATION_FRACTION, alpha=level,
    )
    return receipt


def _multiplier_receipt(receipt: Mapping[str, Any], multiplier: float) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise TypeError("training_receipt must be a mapping")
    required = (
        "training_signed_count", "pilot_discoveries", "pilot_level",
        "uses_held_probabilities_for_selection", "uses_truth_labels",
    )
    if any(key not in receipt for key in required):
        raise ValueError("Incomplete training BH receipt")
    if (receipt["uses_held_probabilities_for_selection"] is not False
            or receipt["uses_truth_labels"] is not False):
        raise ValueError("Receipt must explicitly exclude HELD selection and truth labels")
    size = _integer(receipt["training_signed_count"], "training_signed_count", 1)
    count = _integer(receipt["pilot_discoveries"], "pilot_discoveries", 0)
    if count > size:
        raise ValueError("pilot_discoveries cannot exceed the training signed family")
    level = _level(receipt["pilot_level"])
    if "zero_count_fallback" in receipt:
        fallback = receipt["zero_count_fallback"]
        if not isinstance(fallback, (bool, np.bool_)) or bool(fallback) != (count == 0):
            raise ValueError("Inconsistent zero-count fallback receipt")
    # Counts and level are the sufficient pilot receipt. Ignore an input
    # threshold/fraction: these are recomputed for the requested multiplier.
    threshold = _threshold(multiplier, level, count, size)
    return {
        "training_signed_count": size,
        "pilot_discoveries": count,
        "pilot_level": level,
        "threshold": threshold,
        "concentration_fraction": CONCENTRATION_FRACTION if count else 0.,
        "zero_count_fallback": count == 0,
        "threshold_rule": (
            ".5*alpha*Rpilot/training_family" if multiplier == .5
            else f"{multiplier:g}*alpha*Rpilot/training_family"
        ),
        "uses_held_probabilities_for_selection": False,
        "uses_truth_labels": False,
        "multiplier": multiplier,
        "requested_concentration_fraction": CONCENTRATION_FRACTION,
        "zero_count_threshold_uses_max_one": True,
    }


def apply_pilot_multiplier(
    held_p: ArrayLike, training_receipt: Mapping[str, Any], family: int,
    multiplier: float = .5,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Apply a fixed multiplier to a reusable training BH receipt.

    Threshold = multiplier * alpha * max(1, R_train) / m_train. If R_train
    is zero the concentration fraction is zero, exactly the old capped-BY
    fallback. Otherwise fraction=.8. ``family`` is the eventual signed family
    used for this calibrator, not necessarily the length of this HELD fold.
    The receipt must come from the permitted training information; provenance
    cannot be authenticated from its numerical fields alone.
    """
    selected = _multiplier(multiplier)
    info = _multiplier_receipt(training_receipt, selected)
    size = _integer(family, "family", 2)
    held = _probabilities(held_p, "held_p")
    cap = max(1, size // 8)
    harmonic = np.sum(1. / np.arange(1, cap + 1))
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        scale = size * harmonic / info["pilot_level"]
    if not np.isfinite(scale):
        raise ValueError("family/alpha gives an unsafe float64 calibrator scale")
    values = focused_calibrator(
        held, size, info["threshold"], info["concentration_fraction"],
        info["pilot_level"], cap=cap,
    )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ArithmeticError("Nonfinite or negative calibrated evidence")
    info.update({"family": size, "cap": cap})
    return values, info


def pilot_multiplier_calibrator(
    held_p: ArrayLike, training_p: ArrayLike, family: int,
    multiplier: float = .5, alpha: float = .05,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Convenience fixed-multiplier API; .5 is bitwise old fraction=.8.

    Use pilot_training_receipt followed by apply_pilot_multiplier to reuse
    exactly the same training BH count across the four sensitivity candidates.
    """
    receipt = pilot_training_receipt(training_p, alpha)
    return apply_pilot_multiplier(held_p, receipt, family, multiplier)


def select_pilot_multiplier(
    training_p: ArrayLike, alpha: float = .05,
) -> tuple[float, dict[str, Any]]:
    """Select solely from precomputed outer-TRAIN (N, 2) probabilities.

    Split complete gene rows by their zero-based parity. For each candidate
    and each validation parity, fit the pilot on the opposite parity and run
    e-BH on that validation half with its OWN actual 2*n_half family. Maximize
    the unweighted mean of the two discovery proportions. Exact integer
    scores distinguish ties, then prefer closest to .5, then smaller value.
    Refit the pilot on all supplied TRAIN rows. N>=2 is required; no silent
    tiny-sample split replacement or resampling. No HELD inputs or labels.
    """
    training = _probabilities(training_p, "training_p")
    if training.ndim != 2 or training.shape[1] != 2 or len(training) < 2:
        raise ValueError("Selector requires training_p shape (N, 2) with N>=2")
    level = _level(alpha)
    indices = [np.arange(parity, len(training), 2) for parity in (0, 1)]
    pilots = [pilot_training_receipt(training[index], level) for index in indices]
    sizes = [2 * len(index) for index in indices]
    denominator = 2 * sizes[0] * sizes[1]
    candidates: list[dict[str, Any]] = []
    for multiplier in MULTIPLIERS:
        folds = []
        counts = []
        for validation_parity in (0, 1):
            train_parity = 1 - validation_parity
            validation_index = indices[validation_parity]
            values, application = apply_pilot_multiplier(
                training[validation_index], pilots[train_parity],
                sizes[validation_parity], multiplier,
            )
            rejected = ebh(values, level)
            count = int(rejected.sum())
            counts.append(count)
            folds.append({
                "validation_parity": validation_parity,
                "training_row_indices": indices[train_parity].tolist(),
                "validation_row_indices": validation_index.tolist(),
                "training_bh_receipt": dict(pilots[train_parity]),
                "calibrator_receipt": application,
                "validation_signed_family": sizes[validation_parity],
                "ebh_level": level,
                "discoveries": count,
                "discovery_proportion": count / sizes[validation_parity],
                "validation_evalues": values.tolist(),
                "validation_rejections": rejected.tolist(),
            })
        numerator = counts[0] * sizes[1] + counts[1] * sizes[0]
        candidates.append({
            "multiplier": multiplier,
            "mean_discovery_proportion": numerator / denominator,
            "utility_numerator": numerator,
            "utility_denominator": denominator,
            "folds": folds,
        })
    winner = min(candidates, key=lambda item: (
        -item["utility_numerator"], abs(item["multiplier"] - .5), item["multiplier"],
    ))
    selected = winner["multiplier"]
    all_training = pilot_training_receipt(training, level)
    refit = _multiplier_receipt(all_training, selected)
    return selected, {
        "version": "pilot_multiplier_inner_parity_v1",
        "selection_status": "training_utility_only_not_scientific_validation",
        "training_gene_count": len(training),
        "training_signed_count": training.size,
        "alpha": level,
        "multipliers": list(MULTIPLIERS),
        "concentration_fraction": CONCENTRATION_FRACTION,
        "split": "zero_based_gene_row_parity_no_shuffle",
        "utility": "unweighted_mean_discovery_proportion_over_two_validation_halves",
        "tie_break": "exact_integer_utility_then_closest_to_0.5_then_lower",
        "candidates": candidates,
        "selected_multiplier": selected,
        "selected_mean_discovery_proportion": winner["mean_discovery_proportion"],
        "all_training_bh_receipt": all_training,
        "refit_calibrator_receipt": refit,
        "all_inner_pilots_zero": all(info["pilot_discoveries"] == 0 for info in pilots),
        "all_utilities_zero": all(item["utility_numerator"] == 0 for item in candidates),
        "zero_count_fallback": refit["zero_count_fallback"],
        "uses_held_probabilities_for_selection": False,
        "uses_truth_labels": False,
        "nuisances_refitted_inside_inner_split": False,
        "precomputed_training_pvalues_may_make_utility_optimistic": True,
        "fitted_pvalue_fdr_proven": False,
        "originality_claimed": False,
    }


def selected_pilot_calibrator(
    held_p: ArrayLike, training_p: ArrayLike, family: int, alpha: float = .05,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Select without HELD access, then apply the all-TRAIN refitted rule.

    Selection is independent of both held_p and the outer application family.
    Selection diagnostics remain identical if either is changed. No e-BH is
    run on HELD here: return e-values for the caller's eventual signed family.
    """
    selected, selection = select_pilot_multiplier(training_p, alpha)
    values, application = apply_pilot_multiplier(
        held_p, selection["all_training_bh_receipt"], family, selected,
    )
    return values, {"selection": selection, "application": application}
