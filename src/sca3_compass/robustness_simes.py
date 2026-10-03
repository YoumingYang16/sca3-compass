"""Predeclared shape-only Simes eligibility and Bonferroni fallback.

For known Gaussian noise, or a common independent Student denominator,
ordinary one-sided Simes is used only on a nonnegative SPD retained shape.
Simes values above one half are extended to one BEFORE any weighting.
Any negative retained off-diagonal selects ordinary Bonferroni instead.
Fitted shape eligibility does not certify true noise or plug-in inference.
"""
from itertools import combinations

import numpy as np

SIMES_POLICY = "retained_nonnegative_spd_simes_halflevel_else_bonferroni_v1"


def simes_eligibility(shape):
    """Inspect only the supplied retained shape, never observed p-values.

    Positive diagonal scaling is removed for correlation diagnostics only;
    input geometry is never refitted, ridged, or mutated. Invalid shapes
    raise instead of asserting valid marginal noise. Negative entries have
    NO tolerance: even a tiny negative selects the predeclared fallback.
    """
    shape = np.asarray(shape, dtype=float)
    if (shape.ndim != 2 or shape.shape[0] == 0 or shape.shape[0] != shape.shape[1]
            or not np.isfinite(shape).all()
            or not np.allclose(shape, shape.T, rtol=1e-12, atol=1e-12)
            or np.any(np.diag(shape) <= 0)):
        raise ValueError("Finite symmetric positive definite retained study shape required")
    scale = np.sqrt(np.diag(shape))
    correlation = shape / scale[:, None] / scale[None, :]
    try:
        np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError as error:
        raise ValueError("Positive definite retained study shape required") from error
    off = ~np.eye(len(shape), dtype=bool)
    # Check original entries too: normalization must not hide underflowed negatives.
    negative = bool(np.any(shape[off] < 0))
    return {
        "policy": SIMES_POLICY,
        "retained_count": len(shape),
        "simes_eligible": not negative,
        "method": "bonferroni" if negative else "simes_halflevel",
        "reason": "negative_retained_correlation" if negative else "nonnegative_spd_retained_shape",
        "minimum_retained_correlation": float(correlation[off].min()) if off.any() else None,
        "halflevel_extension": not negative,
        "uses_pvalues_for_selection": False,
    }


def _checked_pvalues(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if (p.ndim < 1 or p.shape[-1] == 0 or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError("Finite probabilities with a nonempty study axis required")
    return p


def _intersection(p, eligible):
    count = p.shape[-1]
    if not eligible:
        # Bonferroni is valid on the full unit interval; do not weaken it by
        # imposing the half-level extension needed only for Simes.
        return np.minimum(1., count * p.min(axis=-1))
    # Preserve the historical support-Simes arithmetic exactly.
    value = np.min(np.sort(p, axis=-1) * count / np.arange(1, count + 1), axis=-1)
    return np.where(value <= .5, value, 1.)


def simes_intersection(pvalues, shape, *, diagnostics=None):
    """Combine marginals on exactly the supplied retained study subshape."""
    info = simes_eligibility(shape)
    p = _checked_pvalues(pvalues)
    if p.shape[-1] != info["retained_count"]:
        raise ValueError("Marginals and retained study shape must match")
    if diagnostics is not None:
        diagnostics.update(info)
    return _intersection(p, info["simes_eligible"])


def simes_pc(pvalues, shape, r=2, *, diagnostics=None):
    """PC = maximum over ALL size S-r+1 intersections, with local fallback.

    Different intersections may use different predeclared rules. Sorting all
    studies and dropping r-1 entries is valid only when every intersection
    uses Simes; otherwise explicitly retain study identities and take max.
    """
    p = _checked_pvalues(pvalues)
    shape = np.asarray(shape, dtype=float)
    studies = p.shape[-1]
    if shape.shape != (studies, studies):
        raise ValueError("Marginals and study shape must match")
    simes_eligibility(shape)  # Reject invalid full geometry, even for singleton PC.
    if isinstance(r, (bool, np.bool_)) or not isinstance(r, (int, np.integer)) or not 1 <= r <= studies:
        raise ValueError("Integer replication count in [1, studies] required")
    count = studies - r + 1
    intersections = []
    for indices in combinations(range(studies), count):
        info = simes_eligibility(shape[np.ix_(indices, indices)])
        intersections.append({"studies": list(indices), **info})
    fallback_count = sum(not info["simes_eligible"] for info in intersections)
    if diagnostics is not None:
        diagnostics.update({
            "policy": SIMES_POLICY, "replication_count": int(r),
            "intersection_count": len(intersections),
            "fallback_intersection_count": fallback_count,
            "fallback_intersection_fraction": fallback_count / len(intersections),
            "intersections": intersections, "uses_pvalues_for_selection": False,
        })
    if fallback_count == 0:
        # Keep legacy fixed/prior/loading unitdiag-positive results bitwise.
        ordered = np.sort(p, axis=-1)[..., r - 1:]
        value = (ordered * (count / np.arange(1, count + 1))).min(axis=-1)
        return np.where(value <= .5, value, 1.)
    result = np.zeros(p.shape[:-1])
    for info in intersections:
        value = _intersection(p[..., info["studies"]], info["simes_eligible"])
        result = np.maximum(result, value)
    return result
