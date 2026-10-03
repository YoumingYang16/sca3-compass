"""TRAIN-simulated utility selection of a two-direction family budget.

Standalone DEVELOPMENT component. No data simulator, gamma optimizer, held
observations, real labels, nuisance fit or baseline-specific policy lives here.
The caller supplies already TRAIN-generated (family, gene, +/-) e-tensors and
their simulated truth, after any frozen expert/gamma choice. Both directions
remain in each 2G e-BH family, including directions assigned zero weight.

Multiplying by a weight >1 need not preserve individual e-validity. Under
per-null conditional e-validity given that claim's OWN training information,
TRAIN-measurable nonnegative weights summing to two PER GENE preserve the
aggregate expected null budget <=2G. This suffices for the e-BH self-consistency
bound. It neither proves fitted-nuisance validity nor permits conditioning on
all cross-fold training sets simultaneously. See the companion note.
"""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np

from .molecular_methods import ebh

POSITIVE_WEIGHT_GRID = (0., .25, .5, .75, 1., 1.25, 1.5, 1.75, 2.)
POLICY = "training_predictive_direction_budget_v1"


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers (not bool/complex/text)")
    array = np.asarray(raw, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _hash_tensor(array, dtype):
    value = np.ascontiguousarray(array, dtype=dtype)
    digest = sha256()
    digest.update(json.dumps([list(value.shape), value.dtype.str]).encode("ascii"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _standard_error(values):
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def directional_budget(
    simulated_e, simulated_truth, *, alpha=.05,
    positive_weight_grid=POSITIVE_WEIGHT_GRID,
):
    """Return (weights[positive, negative], JSON-safe complete selection receipt).

    simulated_e: finite nonnegative B x G x 2 array, generated using ONLY
    the outer fold's TRAIN information and independent/fixed simulation
        randomness. G is the full deployed gene count, not the held-fold size.
        This may be the OLD training-mass-gated mixture, raw projection,
        raw Simes, cone, or a predictive-gamma mixture. No specific gamma
        policy is required or optimized by this budget selector.
    simulated_truth: boolean array of exactly the same shape, truth of the
        GENERATED effects, not real training labels or actual held labels.
        Both directions can be true (e.g. two positive and two negative studies).
    positive_weight_grid: prespecified strictly increasing finite grid in
        [0,2] containing 1. Negative weight is exactly 2 minus positive weight.
        Defaults to nine possibilities; (1.,) is the no-allocation ablation.
        Caller must freeze any custom grid and offer it equally to baselines.

    Every candidate is applied uniformly to all simulated genes, separately
    within each predictive family. Select maximum INTEGER pooled TP, with ties
    closest to equal weights, then smaller positive weight. Equal weights are
    therefore selected whenever they tie for best, including all-null truth.
    Simulated FDP is reported but never optimized or used as a tie-break.

    Apply the returned pair uniformly to all ACTUAL held genes in THIS outer
    fold, without refitting on them, re-normalizing across folds, dropping zero-
    budget claims or changing the final e-BH family size from 2G. This function
    cannot authenticate provenance: its training-only contract must be enforced
    by the caller. A selected predictive gain is not an independent performance
    estimate or a fitted-model FDR guarantee.
    Supply unbudgeted component/convex-mixture e-values and apply this budget
    once: stacking independently normalized allocations need not preserve it.
    """
    evidence = _real_array(simulated_e, "simulated_e")
    if (evidence.ndim != 3 or evidence.shape[-1] != 2
            or not evidence.shape[0] or not evidence.shape[1] or np.any(evidence < 0)):
        raise ValueError("simulated_e must be nonnegative with nonempty shape (B,G,2)")
    truth = np.asarray(simulated_truth)
    if truth.dtype.kind != "b" or truth.shape != evidence.shape:
        raise ValueError("simulated_truth must be boolean and exactly match (B,G,2)")
    grid = _real_array(positive_weight_grid, "positive_weight_grid")
    if (grid.ndim != 1 or not grid.size or np.any(grid < 0) or np.any(grid > 2)
            or np.any(np.diff(grid) <= 0) or not np.any(grid == 1.)):
        raise ValueError("A strictly increasing grid in [0,2] containing 1 is required")
    if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
            or not isinstance(alpha, (int, float, np.integer, np.floating))
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("alpha must be a finite real scalar in (0,1)")
    alpha = float(alpha)
    families, genes, _ = evidence.shape
    # ebh uses finite doubles. Reject unrepresentable thresholds explicitly,
    # rather than silently accept overflow warnings as an all-zero result.
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        try:
            np.float64(2 * genes) / alpha
        except FloatingPointError as error:
            raise ValueError("e-BH threshold is outside floating-point range") from error
    true_by_family = truth.sum(axis=(1, 2))
    total_true = int(true_by_family.sum())
    table = []
    for index, positive in enumerate(grid):
        weights = np.array([positive, 2. - positive])
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            weighted = evidence * weights
        # Do NOT flatten predictive families into one multiple-testing family.
        rejection = np.stack([ebh(family, alpha) for family in weighted])
        true_rejection, false_rejection = rejection & truth, rejection & ~truth
        tp = true_rejection.sum(axis=(1, 2))
        fp = false_rejection.sum(axis=(1, 2))
        fdp = fp / np.maximum(1, tp + fp)
        utility = float(tp.sum() / max(1, total_true))
        residual_se = _standard_error(tp - utility * true_by_family)
        power_se = (residual_se / float(true_by_family.mean())
                    if residual_se is not None and total_true else None)
        table.append({
            "grid_index": index, "weights": weights.tolist(),
            "distance_from_equal": float(abs(positive - 1.)),
            "total_true_positives": int(tp.sum()),
            "total_false_positives": int(fp.sum()),
            "pooled_true_positive_rate": utility,
            "true_positives_by_family": tp.tolist(),
            "false_positives_by_family": fp.tolist(),
            "rejections_by_family": (tp + fp).tolist(),
            "true_positives_by_sign": true_rejection.sum(axis=(0, 1)).tolist(),
            "false_positives_by_sign": false_rejection.sum(axis=(0, 1)).tolist(),
            "simulated_fdp_by_family": fdp.tolist(),
            "mean_simulated_fdp": float(fdp.mean()),
            "descriptive_mc_se_pooled_power": power_se,
            "descriptive_mc_se_mean_fdp": _standard_error(fdp),
        })
    selected = min(range(len(table)), key=lambda j: (
        -table[j]["total_true_positives"], table[j]["distance_from_equal"],
        table[j]["weights"][0],
    ))
    equal_index = int(np.flatnonzero(grid == 1.)[0])
    best, equal = table[selected], table[equal_index]
    evidence_hash, truth_hash = _hash_tensor(evidence, "<f8"), _hash_tensor(truth, "u1")
    inputs_hash = sha256(json.dumps({
        "evidence_sha256": evidence_hash, "truth_sha256": truth_hash,
        "positive_weight_grid": grid.tolist(), "alpha": alpha,
    }, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return np.array(best["weights"]), {
        "policy": POLICY, "development_only": True,
        "families": families, "family_gene_count": genes,
        "signed_family_size": 2 * genes, "alpha": alpha,
        "numpy_version": np.__version__, "direction_order": ["positive", "negative"],
        "positive_weight_grid": grid.tolist(),
        "default_grid_used": tuple(grid) == POSITIVE_WEIGHT_GRID,
        "selected_weights": best["weights"].copy(),
        "selected_grid_index": selected, "equal_weight_grid_index": equal_index,
        "selected_equal_weights": selected == equal_index,
        "selected_total_true_positives": best["total_true_positives"],
        "total_simulated_true_claims": total_true,
        "simulated_true_counts_by_family_and_sign": truth.sum(axis=1).tolist(),
        "selected_utility_name": "pooled_simulated_signed_true_positive_rate",
        "selected_utility": best["pooled_true_positive_rate"],
        "selected_descriptive_mc_se": best["descriptive_mc_se_pooled_power"],
        "equal_weight_total_true_positives": equal["total_true_positives"],
        "selected_minus_equal_total_true_positives": (
            best["total_true_positives"] - equal["total_true_positives"]),
        "selected_minus_equal_tp_by_sign": [
            b - e for b, e in zip(best["true_positives_by_sign"], equal["true_positives_by_sign"])],
        "tied_best_grid_points": sum(
            row["total_true_positives"] == best["total_true_positives"] for row in table),
        "selection_rule": "maximize integer pooled TP; ties closest to equal, then smaller positive weight",
        "utility_table": table, "inputs_sha256": inputs_hash,
        "simulated_e_sha256": evidence_hash, "simulated_truth_sha256": truth_hash,
        "per_gene_budget": 2., "same_weights_for_all_genes_in_fold": True,
        "zero_budget_claims_retained_in_family": True,
        "weights_renormalized_across_genes_or_folds": False,
        "uses_real_truth_labels": False, "uses_simulated_truth_for_utility": True,
        "held_observation_access": False, "provenance_verified": False,
        "uses_simulated_fdp_for_selection": False, "gamma_reselected": False,
        "individual_weighted_e_validity_claimed": False,
        "validity_status": "conditional component e-validity required separately given each claim's own training; aggregate null budget only; fitted nuisance not certified",
        "uncertainty_interpretation": "descriptive predictive-family MC SE, not postselection CI, independent validation or FDR certificate",
    }
