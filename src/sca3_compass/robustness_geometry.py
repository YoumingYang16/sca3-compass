"""Training-only loading geometry extraction, without priors or testing rules.

The working signal is beta_gs * ell_k for one shared positive ell. Learned
geometry and downstream fitted inference carry no exact plug-in guarantee.
"""

from collections.abc import Mapping

import numpy as np

from .robustness_loading import (
    learned_loading,
    loading_gate,
    pipeline_roots,
    positive_geometry,
)
from .robustness_methods import tyler_shape


def _checked_inputs(z, diagnostics, force):
    z = np.asarray(z)
    if z.dtype.kind not in "iuf":
        raise ValueError("Finite real G x 4 x K targets required")
    z = np.asarray(z, dtype=float)
    if (
        z.ndim != 3 or z.shape[0] < 24 or z.shape[1] != 4 or z.shape[2] < 3
        or not np.isfinite(z).all()
    ):
        raise ValueError("Finite G x 4 x K targets with G>=24 and K>=3 required")
    if (
        not isinstance(diagnostics, Mapping)
        or not isinstance(diagnostics.get("fit"), Mapping)
        or "rho" not in diagnostics["fit"]
    ):
        raise ValueError("diagnostics['fit']['rho'] must contain the frozen pipeline rho")
    rho = np.asarray(diagnostics["fit"]["rho"])
    if rho.ndim != 0 or rho.dtype.kind not in "iuf":
        raise ValueError("Frozen fit rho must be a finite real scalar")
    rho = float(rho)
    if not np.isfinite(rho) or not -1 / (z.shape[2] - 1) < rho < 1:
        raise ValueError("Frozen fit rho must define a positive definite pipeline covariance")
    if not isinstance(force, (bool, np.bool_)):
        raise TypeError("force must be boolean")
    return z, rho


def _json_safe(value):
    """Keep the small helper diagnostics strict-JSON serializable."""
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def fit_fold_geometries(z, diagnostics, force=False, *, audit_mode='legacy'):
    """Return two parity-indexed geometries (or None) and a compact receipt.

    For fold f, held = arange(G) % 2 == f. Only z[~held] fits the existing
    loading gate, positive loading and Tyler study shape. Every gene is then
    projected using that fold's fixed parameters. Adopted records contain
    means (G, 4), raw q (G,), scalar dimension and projection_variance,
    study_shape (4, 4), and fold. No prior or p-value is computed.

    A None slot instructs the caller to retain its existing path exactly;
    this function does not reconstruct fallback geometry. force still runs
    and records the current gate, then adopts both folds as an ablation.
    Only diagnostics['fit']['rho'] is read, and it is never refitted. Its
    provenance must be ensured by the caller. Inputs are not mutated.
    audit_mode='intersection' additionally requires positive raw Euclidean
    validation. 'veto' instead rejects an old adoption only with sufficiently
    negative raw validation (or audit failure); absence of positive evidence
    is not automatically treated as evidence against the old proposal.
    'focused_veto' requires BOTH bulk and signal-focused raw audits to give
    negative evidence before vetoing. Inconclusive focus is not positive
    validation. An invalid focused audit fails closed and is retained.
    """
    z, rho = _checked_inputs(z, diagnostics, force)
    if audit_mode not in ('legacy','intersection','veto','focused_veto'):
        raise ValueError('Unknown loading audit mode')
    g, s, k = z.shape
    covariance, root, inverse_root = pipeline_roots(rho, k)
    geometries = [None, None]
    infos = []
    for fold in range(2):
        held = np.arange(g) % 2 == fold
        training = z[~held]
        raw_info=None
        if audit_mode=='legacy':
            adopt, gate = loading_gate(training, root, inverse_root)
        else:
            from .robustness_loading_audit_gate import raw_loading_gate
            raw_adopt,raw_info=raw_loading_gate(training,root,inverse_root,include_old_gate=True)
            gate=raw_info['old_gate']
            valid=raw_info['all_scores_finite'] and raw_info['all_inner_fits_converged']
            veto=bool(valid and raw_info['mean_log_residual_gain']<-.05 and
                raw_info['mean_log_residual_gain']<-2.58*raw_info['gene_level_standard_error'])
            if audit_mode=='focused_veto':
                from .robustness_loading_focus import focused_loading_audit
                try:
                    focus=focused_loading_audit(training,root,inverse_root,raw_audit=raw_info)
                except (ArithmeticError,ValueError,TypeError,np.linalg.LinAlgError) as error:
                    focus={'audit_valid':False,'negative_evidence':False,
                        'positive_evidence':False,'numerical_failure_count':1,
                        'error_type':type(error).__name__,'error_message':str(error),
                        'interpretation':'invalid focus audit; no silent adoption or refit'}
                raw_info['bulk_veto_triggered']=veto
                raw_info['focused_audit']=focus
                valid=bool(valid and focus['audit_valid'])
                veto=bool(valid and veto and focus['negative_evidence'])
            adopt=gate['adopt'] and (raw_adopt if audit_mode=='intersection' else valid and not veto)
            raw_info.update(veto_triggered=veto,audit_mode=audit_mode,
                rule='intersection: positive raw evidence; veto: sufficiently negative raw evidence or audit failure',
                final_unforced_adoption=bool(adopt))
            if audit_mode=='focused_veto':
                raw_info['rule']='legacy adoption AND valid audits AND NOT(bulk negative AND focused negative); inconclusive focus is not validation'
        adopt = bool(adopt or force)
        info = {
            "fold": fold, "training_genes": int((~held).sum()),
            "held_genes": int(held.sum()), "gate": gate,
            "fallback_to_fixed_geometry": not adopt,
            "forced_ablation": bool(force),
        }
        if raw_info is not None:
            info['raw_loading_audit']=raw_info
        if adopt:
            loading, load_info = learned_loading(training @ inverse_root, root)
            weights, basis, variance = positive_geometry(
                loading, covariance, root, inverse_root,
            )
            train_residual = (training @ inverse_root) @ basis
            shape, shape_info = tyler_shape(
                train_residual.transpose(0, 2, 1).reshape(-1, s),
            )
            residual = (z @ inverse_root) @ basis
            q = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual)
            means = z @ weights
            dimension = s * basis.shape[1]
            if (
                not np.isfinite(means).all() or not np.isfinite(q).all()
                or np.any(q < 0) or not np.isfinite(variance) or variance <= 0
            ):
                raise FloatingPointError("Nonfinite or invalid projected fold geometry")
            geometries[fold] = {
                "means": means, "q": q, "dimension": dimension,
                "projection_variance": variance, "study_shape": shape,
                "fold": fold,
            }
            info.update(
                loading=loading.tolist(), projection_weights=weights.tolist(),
                residual_dimension=dimension, projection_variance=variance,
                loading_fit=load_info, study_shape=shape_info,
                study_shape_condition_number=float(np.linalg.cond(shape)),
            )
        infos.append(info)
    receipt = {
        "folds": infos,
        "fallback_gene_fraction": sum(
            info["held_genes"] for info in infos if info["fallback_to_fixed_geometry"]
        ) / g,
        "frozen_fit_rho": rho,
        "pipeline_condition_number": float(np.linalg.cond(covariance)),
        "uses_truth_labels": False,
        "uses_own_gene_to_learn_loading": False,
        "uses_own_gene_to_learn_study_shape": False,
        "refits_calibration_rho": False,
        "projection_scope": "all genes under each adopted fold's fixed geometry",
        "conditions": (
            "shared positive loading ell across genes and studies; separable "
            "pipeline/study noise geometry; caller-supplied frozen rho; "
            "held genes excluded from their fold's geometry learning"
        ),
        "limitations": (
            "does not cover arbitrary gene-specific loading heterogeneity; "
            "cross-fitting does not create cross-gene independence; "
            "learned geometry and fitted inference remain empirical"
        ),
    }
    if audit_mode!='legacy':
        receipt['audit_mode']=audit_mode
    return geometries, _json_safe(receipt)
