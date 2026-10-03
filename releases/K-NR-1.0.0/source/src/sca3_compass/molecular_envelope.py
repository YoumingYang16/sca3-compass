"""Finite-calibration confidence envelope; see the complete assumption ledger.

This is a candidate synthesis of classical results, not a certified new method.
The probability guarantee is unconditional over an independent MC null bank.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri
from scipy.stats import chi2

from .molecular_methods import checked_probabilities, covariance_root


@dataclass(frozen=True)
class CorrelationEnvelope:
    lower_matrix: np.ndarray
    rho_lower: float
    fallback: bool
    n: int
    delta: float


def correlation_envelope(calibration: np.ndarray, delta: float) -> CorrelationEnvelope:
    """Simultaneous lower bounds for correlations, with KNOWN unit variances.

    For each pair, sum of squared centered differences / (2-2*rho) is
    chi-square(n-1). Bonferroni allocates delta across pairs. Sample-standardizing
    calibration columns first would invalidate this exact pivot and is forbidden.
    delta is this study's share of the global confidence-failure budget.
    """
    x = np.asarray(calibration, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all() or not 0 < delta < 1:
        raise ValueError("Need finite n x K calibration, n,K>=2 and delta in (0,1)")
    n, k = x.shape
    i, j = np.triu_indices(k, 1)
    differences = x[:, i] - x[:, j]
    differences -= differences.mean(axis=0)
    q = np.sum(differences**2, axis=0)
    upper_variance = q / chi2.ppf(delta / len(i), n - 1)
    lower = np.maximum(-1.0, 1 - upper_variance / 2)
    bounds = np.eye(k)
    bounds[i, j] = bounds[j, i] = lower
    rho_lower = float(lower.min())
    # Negative lower bounds must NOT be clamped to zero for Slepian calibration.
    return CorrelationEnvelope(bounds, rho_lower, rho_lower < 0, n, delta)


def rank_tail(reference: np.ndarray, observed: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, float)
    obs = np.asarray(observed, float)
    if ref.ndim != 1 or not ref.size or not np.isfinite(ref).all() or not np.isfinite(obs).all():
        raise ValueError("Finite nonempty null bank required")
    ordered = np.sort(ref)
    return (1 + ref.size - np.searchsorted(ordered, obs, side="left")) / (ref.size + 1)


def gaussian_bank(covariance: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Independent Gaussian maximum bank, not a bootstrap from observed genes."""
    if draws < 99:
        raise ValueError("At least 99 draws required")
    root = covariance_root(covariance)
    return (rng.normal(size=(draws, len(root))) @ root.T).max(axis=1)


def equicorrelated_bank(k: int, rho: float, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Exact iid draws of max of an equicorrelated Gaussian vector in O(B).

    max(epsilon_1,...,epsilon_K) has CDF Phi(t)^K. Inverse sampling avoids
    generating an unnecessary B x K matrix. Clip only machine endpoints.
    """
    if not isinstance(k, int) or k < 1 or not 0 <= rho <= 1 or draws < 99:
        raise ValueError("Invalid equicorrelation bank parameters")
    u = rng.random(draws)
    probability = np.exp(np.log(np.maximum(u, np.finfo(float).tiny)) / k)
    probability = np.minimum(probability, np.nextafter(1.0, 0.0))
    return np.sqrt(rho) * rng.normal(size=draws) + np.sqrt(1-rho) * ndtri(probability)


def envelope_pvalues(z: np.ndarray, envelope: CorrelationEnvelope, draws: int,
                     rng: np.random.Generator) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.shape[-1] != len(envelope.lower_matrix) or not np.isfinite(z).all():
        raise ValueError("Statistics must match calibration dimension")
    if envelope.fallback:
        return np.minimum(1, z.shape[-1] * ndtr(-z.max(axis=-1)))
    bank = equicorrelated_bank(z.shape[-1], envelope.rho_lower, draws, rng)
    return rank_tail(bank, z.max(axis=-1))


def efilter_adjusted(e_selection: np.ndarray, e_filtering: np.ndarray) -> np.ndarray:
    """Python port of author's e_filter(..., TypeI='FDR'), MIT attribution.

    Author: Trambak Banerjee, 2025; external/efilter/LICENSE.
    Commit 918ad5c7d31c42e82af33c0aa36fe147b04e0969, funcs.R.
    NOT ordinary valid e-values after the filtering adjustment.
    """
    s, f = np.asarray(e_selection, float), np.asarray(e_filtering, float)
    if (s.ndim != 1 or not s.size or s.shape != f.shape or not np.isfinite(s).all()
            or not np.isfinite(f).all() or np.any(s <= 0) or np.any(f < s)):
        raise ValueError("Need finite positive eS and eF>=eS")
    order = np.argsort(-s, kind="stable")
    m_ord = len(s) - np.searchsorted(np.sort(f), s[order], side="left")
    temp = s[order] * np.arange(1, len(s)+1) / m_ord
    adjusted = np.empty_like(s)
    adjusted[order] = np.maximum.accumulate(temp[::-1])[::-1]
    return adjusted


def efilter(p: np.ndarray, r: int, alpha: float, kappa: float = .5) -> np.ndarray:
    """One directional family. No universal validity assertion for this wrapper."""
    p = checked_probabilities(p)
    if p.ndim != 2 or not 2 <= r <= p.shape[1] or not 0 < alpha < 1 or not 0 < kappa < 1:
        raise ValueError("Invalid partial-conjunction parameters")
    ordered = np.sort(p, axis=1)
    count = p.shape[1] - r + 1
    selection = np.clip(count * ordered[:, r-1], np.finfo(float).tiny, 1)
    filtering = np.clip(count * ordered[:, r-2], np.finfo(float).tiny, 1)
    return efilter_adjusted(kappa * selection**(kappa-1), kappa * filtering**(kappa-1)) > 1/alpha
