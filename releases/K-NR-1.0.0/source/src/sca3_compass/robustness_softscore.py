"""Diagnostic classical monotone soft likelihood-ratio mixtures for r=2 of 4.

For known Gaussian shape Sigma, exp(lambda * v'x - lambda**2 / 2) is
a likelihood ratio for the shift lambda * Sigma @ v, where v'Sigma v=1.
Mixing positive subset-sum directions gives a coordinatewise monotone score,
so zero is least favorable under the component null (all three means <= 0).
This uses classical likelihood mixtures and Monte Carlo rank tests; no novelty
or unconditional validity theorem for estimated/plugin shapes is claimed.

For Student data the same score is ONLY a statistic, not a likelihood ratio or
an e-value. It is calibrated against an independent multivariate Student bank
with the supplied shape and degrees of freedom. Rank validity requires that
this bank match the boundary-null law, conditionally if nuisance inputs are
random. Cross-fitting alone does not establish that matching assumption.
"""

import time
from itertools import combinations
from numbers import Integral

import numpy as np
from scipy.special import logsumexp

from .molecular_methods import covariance_root
from .robustness_mc import projection_weights

_NAMES = ("strength2", "strength4", "mixture")
_TRIPLES = tuple(combinations(range(4), 3))
_LOG_WEIGHTS = np.log([.45 / 3] * 3 + [.1 / 3] * 3 + [.45])
_CHUNK_ROWS = 8192


def _soft_log_scores(projected):
    """Seven projections in size/combination order -> three log scores."""
    score2 = logsumexp(2 * projected - 2 + _LOG_WEIGHTS, axis=-1)
    score4 = logsumexp(4 * projected - 8 + _LOG_WEIGHTS, axis=-1)
    mixture = np.logaddexp(score2, score4) - np.log(2.)
    return np.stack((score2, score4, mixture), axis=-1)


def _rank_tail_sorted(ordered, observed):
    """Conservative plus-one rank: reference ties count in the upper tail."""
    return (1 + len(ordered) - np.searchsorted(ordered, observed, side="left")) / (
        len(ordered) + 1
    )


def _geometry(shape):
    # Normalize only to reuse the existing correlation-root validator; shape
    # itself may have a nonunit diagonal. Do not whiten the observed vectors:
    # whitening could destroy coordinatewise monotonicity.
    if shape.shape != (4, 4) or not np.isfinite(shape).all():
        raise ValueError("Two finite 4 x 4 shapes required")
    if np.any(np.diag(shape) <= 0):
        raise ValueError("Shapes must have positive diagonal entries")
    scale = np.sqrt(np.diag(shape))
    root = scale[:, None] * covariance_root(shape / np.outer(scale, scale))
    weights = []
    for subset in _TRIPLES:
        with np.errstate(divide="ignore", invalid="ignore"):
            w = projection_weights(shape, subset)
        if not np.isfinite(w).all():
            raise ValueError("Every positive subset sum must have positive variance")
        weights.append(w)
    return root, weights


def softscore_pc(mean, variance, shapes, df, rng, draws=131071):
    """Return ``(pvalues, diagnostics)`` for two prespecified signs per gene.

    ``mean`` is finite G x 2 x 4, ``variance`` is a positive G vector, and
    ``shapes`` contains two symmetric positive semidefinite 4 x 4 matrices.
    Every subset-sum variance must be positive. Rows g use shapes[g % 2], as
    in robustness_mc. For each triple, x = mean / sqrt(variance) is projected
    on v_A = 1_A / sqrt(1_A' shape 1_A) for its seven nonempty subsets.

    Priors on subset sizes 1/2/3 are (.45, .1, .45), uniform within each size.
    ``strength2`` and ``strength4`` use fixed lambda=2 and 4; ``mixture`` is
    their equal likelihood-scale mixture, calibrated as its own statistic.
    Each returned p-value array is G x 2 and takes the maximum rank p over
    all four triples, testing the partial-conjunction null of at most one
    positive study. Signs and variants are separate outputs, with no further
    multiplicity adjustment or data-dependent choice among them.

    ``df`` is a positive scalar, or +inf for Gaussian calibration. Student
    bank rows use one shared independent chi-square scale across all studies;
    ``variance`` supplies the scale, not the marginal Student variance.
    The caller supplies an np.random.Generator independent of observed data.
    Each fold's bank is reused across triples, signs, genes, and strengths.

    Score work is chunked; at the default draws retained banks/scores take
    about 7 MiB, with small fixed-size scratch, below a 200 MB working-memory
    budget excluding caller inputs and required G x 2 outputs. Storage is
    O(draws + G), not O(draws * G). The caller controls BLAS thread limits.
    Diagnostics report elapsed_seconds, draws per bank, bank_count, min_p,
    and the calibration family.
    """
    started = time.perf_counter()
    mean = np.asarray(mean, dtype=float)
    variance = np.asarray(variance, dtype=float)
    shapes = np.asarray(shapes, dtype=float)
    df = np.asarray(df, dtype=float)
    if mean.ndim != 3 or mean.shape[1:] != (2, 4) or not np.isfinite(mean).all():
        raise ValueError("Finite G x 2 x 4 means required")
    if (variance.shape != (len(mean),) or not np.isfinite(variance).all()
            or np.any(variance <= 0)):
        raise ValueError("A finite positive G-vector of variances is required")
    if shapes.shape != (2, 4, 4):
        raise ValueError("Two finite 4 x 4 shapes required")
    if df.ndim != 0 or np.isnan(df) or df <= 0:
        raise ValueError("Positive scalar df or +inf required")
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, Integral) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an independent np.random.Generator")
    geometry = [_geometry(shape) for shape in shapes]
    df = float(df)
    draws = int(draws)
    result = {name: np.zeros(mean.shape[:2]) for name in _NAMES}
    bank_count = 0
    for fold, (root, weights) in enumerate(geometry):
        if fold >= len(mean):
            continue
        bank = rng.normal(size=(draws, 4)) @ root.T
        if np.isfinite(df):
            bank /= np.sqrt(rng.chisquare(df, size=(draws, 1)) / df)
        if not np.isfinite(bank).all():
            raise ValueError("Nonfinite null bank; df/shape exceed numerical range")
        bank_count += 1
        reference = np.empty((draws, len(_NAMES)))
        for w in weights:
            for start in range(0, draws, _CHUNK_ROWS):
                rows = slice(start, start + _CHUNK_ROWS)
                reference[rows] = _soft_log_scores(bank[rows] @ w)
            if not np.isfinite(reference).all():
                raise ValueError("Nonfinite reference scores")
            reference.sort(axis=0)
            for start in range(fold, len(mean), 2 * _CHUNK_ROWS):
                rows = slice(start, min(start + 2 * _CHUNK_ROWS, len(mean)), 2)
                x = mean[rows] / np.sqrt(variance[rows, None, None])
                observed = _soft_log_scores(x @ w)
                if not np.isfinite(observed).all():
                    raise ValueError("Nonfinite observed scores")
                for j, name in enumerate(_NAMES):
                    p = _rank_tail_sorted(reference[:, j], observed[..., j])
                    np.maximum(result[name][rows], p, out=result[name][rows])
        del bank, reference
    return result, {
        "elapsed_seconds": time.perf_counter() - started,
        "draws": draws,
        "bank_count": bank_count,
        "min_p": 1 / (draws + 1),
        "calibration": "gaussian" if np.isinf(df) else "student",
    }
