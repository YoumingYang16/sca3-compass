"""Training-only empirical-Bayes predictive utility for convex e-mixtures.

DEVELOPMENT mechanism, not posterior-FDR testing or an FDR certificate.
The caller supplies only OTHER genes' fitted pattern weights, paired scale
and own-Q weights, frozen nuisance parameters, profiles and pilot receipts.
No held observations, real truth labels or case identifier are accepted.

Sixteen independent predictive families and a fixed seed compare all 25
sign-specific mixtures using common random numbers. Simulated locations
provide simulated truth. Selection maximizes pooled simulated true-positive
rate, not simulated FDP. A wrong predictive model can choose a poor mixture.
Known conditional e-validity is preserved by ANY training-only convex weight;
this does not establish validity of fitted nuisance-dependent e-values.
See docs/robustness_predictive_gate.md for the precise conditional argument.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from itertools import product
import json

import numpy as np

from .molecular_methods import checked_probabilities, ebh
from .robustness_calibrators import focused_calibrator
from .robustness_cone import cone_partial_conjunction
from .robustness_patterns import pattern_library
from .robustness_projection import projection_pc


DEFAULT_SEED = 719991
DEFAULT_FAMILIES = 16
GAMMA_GRID = (0., .25, .5, .75, 1.)
POLICY = "training_predictive_pooled_tp_grid25_v1"


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _hash_arrays(*arrays):
    digest = sha256()
    for array in arrays:
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(json.dumps(list(canonical.shape)).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _frozen_pilot(receipt, alpha, training_signed_count=None):
    """Validate a receipt from pilot_calibrator; never refit on simulations."""
    if not isinstance(receipt, Mapping):
        raise ValueError("A frozen training pilot receipt is required")
    required = {
        "training_signed_count", "pilot_discoveries", "pilot_level", "threshold",
        "concentration_fraction", "zero_count_fallback", "threshold_rule",
        "uses_held_probabilities_for_selection", "uses_truth_labels",
    }
    if not required.issubset(receipt):
        raise ValueError("Incomplete frozen training pilot receipt")
    count = _integer(receipt["training_signed_count"], "training_signed_count", 2)
    discoveries = _integer(receipt["pilot_discoveries"], "pilot_discoveries", 0)
    if count % 2 or discoveries > count or (training_signed_count is not None and count != training_signed_count):
        raise ValueError("Pilot training family must match the paired signed training genes")
    level, threshold, fraction = (float(receipt[k]) for k in
                                  ("pilot_level", "threshold", "concentration_fraction"))
    if not np.isfinite([level, threshold, fraction]).all() or level != alpha or not 0 <= fraction <= 1:
        raise ValueError("Invalid or mismatched frozen pilot level/fraction")
    expected = .5 * alpha * max(1, discoveries) / count
    if threshold != expected or (discoveries == 0 and fraction != 0.):
        raise ValueError("Frozen pilot threshold/fallback disagrees with its training count")
    if receipt["threshold_rule"] != ".5*alpha*Rpilot/training_family":
        raise ValueError("Unrecognized pilot threshold rule")
    if (receipt["zero_count_fallback"] is not (discoveries == 0)
            or receipt["uses_held_probabilities_for_selection"] is not False
            or receipt["uses_truth_labels"] is not False):
        raise ValueError("Pilot receipt must declare training-only selection and the correct fallback")
    return {
        "training_signed_count": count, "pilot_discoveries": discoveries,
        "pilot_level": level, "threshold": threshold, "concentration_fraction": fraction,
        "zero_count_fallback": discoveries == 0,
        "threshold_rule": ".5*alpha*Rpilot/training_family",
        "uses_held_probabilities_for_selection": False, "uses_truth_labels": False,
    }


def apply_frozen_pilot(pvalues, receipt, family_gene_count, *, alpha=.05):
    """Apply an already selected pilot to any p-array, with the DEPLOYED 2G.

    Array length is not the testing family size: simulations may be batched.
    No pilot is recomputed here. Exactly reuses focused_calibrator, including
    the zero-discovery fallback and max(1, (2G)//8) cap used by pilot_calibrator.
    """
    genes = _integer(family_gene_count, "family_gene_count")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0,1)")
    frozen = _frozen_pilot(receipt, alpha)
    return focused_calibrator(
        pvalues, 2 * genes, frozen["threshold"], frozen["concentration_fraction"],
        alpha, cap=max(1, (2 * genes) // 8),
    )


def _sample_predictive(weights, paired_training, shape, df, genes, families, seed):
    """Internal sampler; locations/truth are GENERATED, not external labels."""
    patterns, amplitudes = pattern_library()
    rng = np.random.Generator(np.random.PCG64(seed))
    components = rng.choice(len(weights), size=(families, genes), p=weights)
    # Independent of components, while preserving the empirical V/W pairing.
    training_indices = rng.integers(len(paired_training), size=(families, genes))
    sampled_pair = paired_training[training_indices]
    variance, q_weight = sampled_pair[..., 0], sampled_pair[..., 1]
    locations = patterns[components] * amplitudes[components, None]
    truth = np.stack(((locations > 0).sum(-1) >= 2, (locations < 0).sum(-1) >= 2), axis=-1)
    noise = rng.standard_normal((families, genes, 4)) @ np.linalg.cholesky(shape).T
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        if not np.isinf(df):
            radial = rng.chisquare(df, size=(families, genes)) / df
            # No tail clipping, resampling until finite, or seed replacement.
            if not np.isfinite(radial).all() or np.any(radial <= 0):
                raise FloatingPointError("Predictive Student radial draw is not representable")
            noise /= np.sqrt(radial)[..., None]
        means = locations + np.sqrt(variance)[..., None] * noise
    if not np.isfinite(means).all():
        raise FloatingPointError("Predictive observation is outside floating-point range")
    return means, variance, q_weight, truth, components, training_indices


def _standard_error(values):
    if len(values) < 2:
        return None
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _grid_utilities(cone_e, candidate_e, truth, alpha):
    """Score identical predictive families for every pair; integer TP ties."""
    true_by_family = truth.sum(axis=(1, 2))
    total_true = int(true_by_family.sum())
    table = []
    for positive, negative in product(GAMMA_GRID, repeat=2):
        gamma = np.array([positive, negative])
        mixture = (1 - gamma) * cone_e + gamma * candidate_e
        rejection = np.stack([ebh(family, alpha) for family in mixture])
        tp = (rejection & truth).sum(axis=(1, 2))
        fp = (rejection & ~truth).sum(axis=(1, 2))
        count = tp + fp
        fdp = fp / np.maximum(1, count)
        pooled = float(tp.sum() / max(1, total_true))
        # Cluster-by-family delta-method MC SE for the pooled ratio, ONLY
        # descriptive under this fitted predictive distribution. No selection CI.
        ratio_residual = tp - pooled * true_by_family
        ratio_se = _standard_error(ratio_residual)
        if ratio_se is not None and total_true:
            ratio_se /= float(true_by_family.mean())
        else:
            ratio_se = None
        table.append({
            "grid_index": len(table),
            "grid_indices": [GAMMA_GRID.index(positive), GAMMA_GRID.index(negative)],
            "gamma": [positive, negative], "pooled_true_positive_rate": pooled,
            "total_true_positives": int(tp.sum()), "total_false_positives": int(fp.sum()),
            "true_positives_by_family": tp.tolist(), "false_positives_by_family": fp.tolist(),
            "true_positives_by_sign": (rejection & truth).sum(axis=(0, 1)).tolist(),
            "simulated_fdp_by_family": fdp.tolist(), "mean_simulated_fdp": float(fdp.mean()),
            "descriptive_mc_se_pooled_power": ratio_se,
            "descriptive_mc_se_mean_fdp": _standard_error(fdp),
        })
    # Same denominator for every pair. Compare integer counts, not rounded
    # rates or an arbitrary floating-point utility-tie tolerance.
    selected = min(range(len(table)), key=lambda j: (
        -table[j]["total_true_positives"], sum(table[j]["gamma"]), *table[j]["gamma"],
    ))
    return np.array(table[selected]["gamma"]), table, selected


def simulate_predictive_e(
    pattern_weights, training_variance, training_q_weights, study_shape, df,
    profiles, family_gene_count, cone_pilot, candidate_pilot, *,
    candidate="projection", families=DEFAULT_FAMILIES, seed=DEFAULT_SEED, alpha=.05,
):
    """Return TRAIN-simulated cone, candidate, truth tensors and model receipt.

    pattern_weights: 241 fitted weights in pattern_library() order.
    training_variance, training_q_weights: paired positive N-vectors from
        OTHER genes. Variance is the conditional Student SCALE multiplier,
        not the marginal variance. Q weights are NOT renormalized here.
    study_shape: frozen SPD 4x4 scale shape (need not have unit diagonal).
    df: frozen conditional Student df, or +inf for exactly Gaussian noise.
    profiles: frozen nonnegative 2x4 candidate profiles; zero subsets use
        the existing projection_pc uniform-subset fallback.
    family_gene_count: actual complete deployed G, NOT the training fold N.
    cone_pilot, candidate_pilot: already selected pilot_calibrator receipts,
        typically fraction=.8, both formed on the same 2N training family.
    candidate: 'projection' or 'support_simes', with identical information,
        simulation and search opportunity; only the candidate PC test differs.

    Must be called once per outer training fold. This API does not enforce
    provenance cryptographically: the caller must provide ONLY training fits.
    Fitting uncertainty, effect/scale dependence and gene dependence are NOT
    integrated into this plug-in predictive model. No hidden fallback occurs.
    """
    genes = _integer(family_gene_count, "family_gene_count")
    families = _integer(families, "families")
    seed = _integer(seed, "seed", 0)
    if candidate not in {"projection", "support_simes"}:
        raise ValueError("candidate must be projection or support_simes")
    if not np.isscalar(alpha) or not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0,1)")
    weights = np.asarray(pattern_weights, dtype=float)
    patterns, amplitudes = pattern_library()
    if (weights.shape != amplitudes.shape or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12)):
        raise ValueError("241 finite nonnegative pattern weights summing to one required")
    original_weight_sum = float(weights.sum())
    weights = weights / original_weight_sum  # simplex roundoff only
    variance, q_weights = np.asarray(training_variance, float), np.asarray(training_q_weights, float)
    if (variance.ndim != 1 or not variance.size or q_weights.shape != variance.shape
            or not np.isfinite(variance).all() or np.any(variance <= 0)
            or not np.isfinite(q_weights).all() or np.any(q_weights <= 0)):
        raise ValueError("Paired finite positive training variance/Q-weight vectors required")
    shape, profiles = np.asarray(study_shape, float), np.asarray(profiles, float)
    if (shape.shape != (4, 4) or not np.isfinite(shape).all()
            or not np.allclose(shape, shape.T, rtol=1e-12, atol=1e-12)):
        raise ValueError("Finite symmetric positive definite 4x4 study shape required")
    try:
        np.linalg.cholesky(shape)
    except np.linalg.LinAlgError as error:
        raise ValueError("Positive definite study shape required") from error
    if (not np.isscalar(df) or np.isnan(df) or df <= 0
            or profiles.shape != (2, 4) or not np.isfinite(profiles).all() or np.any(profiles < 0)):
        raise ValueError("Positive df (or +inf) and finite nonnegative 2x4 profiles required")
    df = float(df)
    cone_receipt = _frozen_pilot(cone_pilot, alpha, 2 * len(variance))
    candidate_receipt = _frozen_pilot(candidate_pilot, alpha, 2 * len(variance))
    # Canonical pair ordering makes arbitrary training-row permutations exact.
    pair_order = np.lexsort((q_weights, variance))
    paired = np.column_stack((variance[pair_order], q_weights[pair_order]))
    means, simulated_variance, simulated_weight, truth, components, indices = _sample_predictive(
        weights, paired, shape, df, genes, families, seed,
    )
    flat_means = means.reshape(-1, 4)
    signed = np.stack((flat_means, -flat_means), axis=1)
    flat_variance = simulated_variance.reshape(-1)
    flat_weight = simulated_weight.reshape(-1, 1)
    cone_p = cone_partial_conjunction(signed, flat_variance, [shape, shape], df)
    simes_diagnostics = {} if candidate == "support_simes" else None
    candidate_p = projection_pc(
        signed, flat_variance, shape, df, profiles,
        support_simes=candidate == "support_simes", diagnostics=simes_diagnostics,
    )
    # Avoid overflow for extremely small W without modifying any valid p/W:
    # division is needed only when p<W; otherwise min(1,p/W) is exactly one.
    def weighted_p(p):
        p = checked_probabilities(p)
        result = np.ones_like(p)
        np.divide(p, flat_weight, out=result, where=p < flat_weight)
        return result

    cone_e = apply_frozen_pilot(weighted_p(cone_p), cone_receipt, genes, alpha=alpha).reshape(families, genes, 2)
    candidate_e = apply_frozen_pilot(weighted_p(candidate_p), candidate_receipt, genes, alpha=alpha).reshape(families, genes, 2)
    prior_hash = _hash_arrays(patterns, amplitudes, weights)
    predictive_model_hash = sha256((
        _hash_arrays(patterns, amplitudes, weights, paired, shape, np.array([df]))
        + json.dumps([genes, families, seed], separators=(",", ":"))
    ).encode("utf-8")).hexdigest()
    inputs_hash = sha256((
        _hash_arrays(weights, paired, shape, np.array([df]), profiles)
        + json.dumps([cone_receipt, candidate_receipt, genes, families, seed, alpha, candidate],
                     sort_keys=True, separators=(",", ":"), allow_nan=False)
    ).encode("utf-8")).hexdigest()
    return cone_e,candidate_e,truth,{
        "policy": "training_predictive_e_components_v1", "development_only": True, "candidate": candidate,
        "families": families, "family_gene_count": genes, "signed_family_size": 2 * genes,
        "training_gene_count": len(variance), "seed": seed, "rng": "NumPy PCG64",
        "numpy_version": np.__version__, "alpha": float(alpha),
        "gaussian": bool(np.isinf(df)), "conditional_df": None if np.isinf(df) else df,
        "prior_sha256": prior_hash, "inputs_sha256": inputs_hash,
        "predictive_model_sha256": predictive_model_hash,
        "simulation_sha256": _hash_arrays(means, simulated_variance, simulated_weight, truth),
        "component_indices_sha256": _hash_arrays(components),
        "training_pair_indices_sha256": _hash_arrays(indices),
        "simulated_component_counts": np.bincount(components.ravel(), minlength=len(weights)).tolist(),
        "simulated_true_counts_by_family_and_sign": truth.sum(axis=1).tolist(),
        "total_simulated_true_claims": int(truth.sum()),
        "original_pattern_weight_sum": original_weight_sum,
        "training_pair_sha256": _hash_arrays(paired),
        "q_weights_renormalized": False,
        "cone_pilot": cone_receipt, "candidate_pilot": candidate_receipt,
        "support_simes_diagnostics": simes_diagnostics,
        "uses_real_truth_labels": False, "uses_simulated_truth_for_utility": True,
        "held_observation_access": False, "uses_simulated_fdp_for_selection": False,
        "parameter_posterior_integrated": False,
        "uncertainty_interpretation": "descriptive predictive-family MC SE; not a postselection confidence interval or FDR certificate",
        "validity_status": "conditional on valid component e-values given training; fitted-nuisance validity not established here",
    }


def select_predictive_gamma(cone_e,candidate_e,truth,simulation_receipt):
    """Apply the unchanged gamma policy to the simulator's already frozen output.

    Internal composition interface; supplied tensors MUST come from the TRAIN
    simulator. This is not an API for selecting on real held labels.
    """
    gamma,table,selected=_grid_utilities(cone_e,candidate_e,truth,simulation_receipt['alpha'])
    best_count=table[selected]['total_true_positives']
    return gamma,{**simulation_receipt,'policy':POLICY,
        "gamma_grid": list(GAMMA_GRID), "selected_gamma": gamma.tolist(),
        "selected_grid_index": selected,
        "selected_grid_indices": table[selected]["grid_indices"].copy(),
        "selected_utility": table[selected]["pooled_true_positive_rate"],
        "selected_utility_name": "pooled_simulated_signed_true_positive_rate",
        "selected_total_true_positives": best_count,
        "selected_descriptive_mc_se": table[selected]["descriptive_mc_se_pooled_power"],
        "tied_best_grid_pairs": sum(row["total_true_positives"] == best_count for row in table),
        "selection_rule": "maximize pooled TP; ties minimize gamma+ + gamma-, then gamma+, then gamma-",
        "utility_table": table}


def predictive_gate(
    pattern_weights, training_variance, training_q_weights, study_shape, df,
    profiles, family_gene_count, cone_pilot, candidate_pilot, *,
    candidate="projection", families=DEFAULT_FAMILIES, seed=DEFAULT_SEED, alpha=.05,
):
    """Original gamma policy/receipt, unchanged after simulator extraction."""
    simulation=simulate_predictive_e(pattern_weights,training_variance,training_q_weights,
        study_shape,df,profiles,family_gene_count,cone_pilot,candidate_pilot,
        candidate=candidate,families=families,seed=seed,alpha=alpha)
    return select_predictive_gamma(*simulation)
