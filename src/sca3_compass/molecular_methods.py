"""Auditable reference procedures. No algorithm in this module is claimed novel.

Axes: genes x studies x prespecified analytic pipelines. Two signs are tested
as separate hypotheses. See docs/MOLECULAR_METHODS_PROTOCOL.md for assumptions.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def checked_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Probabilities must be finite, nonempty and in [0, 1]")
    return values


def fdr_adjust(pvalues: np.ndarray, method: str = "BY") -> np.ndarray:
    p = checked_probabilities(pvalues)
    if method not in {"BH", "BY"}:
        raise ValueError("Unknown FDR procedure")
    flat = p.ravel()
    order = np.argsort(flat, kind="stable")
    m = flat.size
    harmonic = np.sum(1 / np.arange(1, m + 1)) if method == "BY" else 1.0
    ordered = np.minimum.accumulate((flat[order] * m * harmonic / np.arange(1, m + 1))[::-1])[::-1]
    output = np.empty(m)
    output[order] = np.minimum(ordered, 1)
    return output.reshape(p.shape)


def partial_conjunction(pvalues: np.ndarray, r: int) -> np.ndarray:
    """Bonferroni PC p=(n-r+1)*p_(r); arbitrary study dependence."""
    p = checked_probabilities(pvalues)
    if p.ndim < 1 or not isinstance(r, int) or not 1 <= r <= p.shape[-1]:
        raise ValueError("r must be an integer in [1, number of studies]")
    return np.minimum(1, (p.shape[-1] - r + 1) * np.partition(p, r - 1, axis=-1)[..., r - 1])


def ebh(evalues: np.ndarray, alpha: float) -> np.ndarray:
    e = np.asarray(evalues, dtype=float)
    if not e.size or not np.isfinite(e).all() or np.any(e < 0) or not 0 < alpha < 1:
        raise ValueError("Invalid e-values or alpha")
    flat = e.ravel()
    order = np.argsort(-flat, kind="stable")
    candidates = flat[order] >= flat.size / (alpha * np.arange(1, flat.size + 1))
    rejected = np.zeros(flat.size, dtype=bool)
    if candidates.any():
        k = np.flatnonzero(candidates)[-1] + 1
        rejected[order[:k]] = True
    return rejected.reshape(e.shape)


def covariance_root(covariance: np.ndarray) -> np.ndarray:
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.isfinite(cov).all():
        raise ValueError("Covariance must be finite and square")
    if not np.allclose(cov, cov.T) or not np.allclose(np.diag(cov), 1):
        raise ValueError("Expected a symmetric unit-diagonal correlation matrix")
    eigenvalues, vectors = np.linalg.eigh(cov)
    if eigenvalues.min() < -1e-9:
        raise ValueError("Covariance must be positive semidefinite")
    return vectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))


def equicorrelation(dimension: int, rho: float) -> np.ndarray:
    if dimension < 1 or not 0 <= rho <= 1:
        raise ValueError("Unsupported equicorrelation")
    return (1 - rho) * np.eye(dimension) + rho * np.ones((dimension, dimension))


class GaussianMaxT:
    """Finite Monte Carlo rank calibration under a specified Gaussian null.

    Reusing a bank across hypotheses allows dependence; BY does not require
    independent p-values. Monte Carlo validity is unconditional over the bank.
    It does NOT extend to an arbitrarily misspecified/estimated covariance.
    """

    def __init__(self, covariance: np.ndarray, draws: int, seed: int):
        if draws < 99:
            raise ValueError("At least 99 null draws are required")
        root = covariance_root(covariance)
        rng = np.random.default_rng(seed)
        draws_z = rng.normal(size=(draws, len(root))) @ root.T
        self.reference = np.sort(draws_z.max(axis=1))
        self.pipelines = len(root)
        self.minimum_p = 1 / (draws + 1)

    def pvalues(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        if z.shape[-1] != self.pipelines or not np.isfinite(z).all():
            raise ValueError("Invalid statistics or pipeline count")
        observed = z.max(axis=-1)
        greater_equal = len(self.reference) - np.searchsorted(self.reference, observed, side="left")
        return (1 + greater_equal) / (len(self.reference) + 1)


def p_to_e_mixture(p: np.ndarray) -> np.ndarray:
    """Prespecified convex mixture of standard power calibrators, not learned."""
    p = np.maximum(checked_probabilities(p), np.finfo(float).tiny)
    return np.mean([(1 - k) * p ** (-k) for k in (0.25, 0.5, 0.75)], axis=0)


def evaluate_methods(z: np.ndarray, r: int, alpha: float, calibration: GaussianMaxT) -> dict[str, np.ndarray]:
    if z.ndim != 3 or not np.isfinite(z).all() or not 0 < alpha < 1:
        raise ValueError("Expected finite genes x studies x pipelines and valid alpha")
    # 2G hypotheses: rows are genes and columns are positive/negative claims.
    signed = np.stack((z, -z), axis=1)
    per_pipeline = ndtr(-signed)
    selected = per_pipeline.min(axis=-1)
    bonf = np.minimum(1, selected * z.shape[-1])
    calibrated = calibration.pvalues(signed)
    fixed = per_pipeline[..., 0]
    results = {
        "naive_min_PC_BH": fdr_adjust(partial_conjunction(selected, r), "BH") <= alpha,
        "pipeline_bonferroni_PC_BY": fdr_adjust(partial_conjunction(bonf, r)) <= alpha,
        "fixed_pipeline_PC_BY": fdr_adjust(partial_conjunction(fixed, r)) <= alpha,
        "maxT_PC_BY": fdr_adjust(partial_conjunction(calibrated, r)) <= alpha,
    }
    # Under a directional null, each prespecified pipeline supplies a valid p.
    # Average pipeline e-values; smallest n-r+1 study mean is an e-PC value.
    study_e = p_to_e_mixture(per_pipeline).mean(axis=-1)
    count = z.shape[1] - r + 1
    if count < 1:
        raise ValueError("r exceeds study count")
    pc_e = np.sort(study_e, axis=-1)[..., :count].mean(axis=-1)
    results["mixture_e_PC_eBH"] = ebh(pc_e, alpha)
    return results
