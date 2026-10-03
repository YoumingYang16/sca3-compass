"""Optional leave-GENE-out BH threshold e-values for two dependent signs.

Each row is one gene; both of its p-values are replaced by zero to choose
its threshold, with family size still 2*N. Validity requires conditional
superuniformity and independence across genes, not across signs. These
assumptions cannot be verified from an observed p-value array.

This is a specific block adaptation of classical leave-one-out reasoning,
not a claim of a new theorem or BH equivalence. In particular, complementary
pairs cannot yield standalone e-BH rejections at the construction level
alpha < 1/2. See docs/robustness_block_e.md for derivations and primary sources.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["block_bh_evalues", "block_bh_evalues_reference"]

_CONDITIONS = (
    "Condition on the prespecified training/external information and known nuisance.",
    "Every true signed-null p-value is superuniform given that information.",
    ("Gene blocks are mutually independent given that information, including "
     "null and alternative genes; within-gene signs may be dependent."),
    ("The family, gene pairing, and alpha are fixed given that information; "
     "do not tune them on the held fold."),
)


def _validate(
    p: ArrayLike, alpha: float,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    if np.ma.isMaskedArray(p) or np.ma.isMaskedArray(alpha):
        raise ValueError("Masked p-values or alpha are not supported")
    values = np.asarray(p)
    if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
        raise ValueError("p must have nonempty shape (N, 2), one gene per row")
    if values.dtype.kind not in "fiu":
        raise TypeError("p must have a real numeric, non-boolean dtype")
    if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("p must contain finite probabilities in [0, 1]")
    level = np.asarray(alpha)
    if level.ndim != 0 or level.dtype.kind not in "fiu":
        raise TypeError("alpha must be a real, non-boolean scalar")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    # Multiply before dividing: alpha / m could underflow unnecessarily.
    # Rank 1 can be subnormal; only ranks >= 2 are used as denominators.
    with np.errstate(under="ignore", over="ignore", divide="ignore"):
        values = values.astype(np.float64, copy=False)
        level = float(level)
        cuts = level * np.arange(1, values.size + 1, dtype=np.float64) / values.size
        finite_reciprocal = np.isfinite(1.0 / cuts[1])
    if cuts[1] <= 0 or not finite_reciprocal:
        raise ValueError("alpha is too small for finite float64 e-values at this family size")
    return values, level, cuts


def _result(
    p: NDArray[np.float64], alpha: float, cuts: NDArray[np.float64],
    counts: NDArray[np.intp], implementation: str,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    tau = cuts[counts - 1]
    e = (p <= tau[:, None]) / tau[:, None]
    return e, {
        "tau": tau,
        "R": counts,
        "alpha": alpha,
        "n_genes": p.shape[0],
        "family_size": p.size,
        "excluded_signs_per_gene": 2,
        "conditions": _CONDITIONS,
        "conditions_verified": False,
        "arbitrary_gene_dependence_valid": False,
        "implementation": implementation,
    }


def block_bh_evalues_reference(
    p: ArrayLike, alpha: float = .05,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """O(N**2) reference: exclude each sorted pair, prepend zeros, scan BH.

    Sorting once lets each literal block-zeroed BH scan take O(N) time;
    temporary space is O(N). Contract and conditions match block_bh_evalues.
    """
    values, level, cuts = _validate(p, alpha)
    order = np.argsort(values, axis=None, kind="stable")
    sorted_p = values.ravel()[order]
    owners = order // 2
    counts = np.empty(values.shape[0], dtype=np.intp)
    for gene in range(values.shape[0]):
        zeroed = np.concatenate((np.zeros(2), sorted_p[owners != gene]))
        counts[gene] = np.flatnonzero(zeroed <= cuts)[-1] + 1
    return _result(values, level, cuts, counts, "quadratic_reference")


def block_bh_evalues(
    p: ArrayLike, alpha: float = .05,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    """Return (e, diagnostics) for an (N, 2) array of signed p-values.

    For gene i, replace BOTH signs by zero, run BH(alpha) on all 2*N values,
    and let R[i] >= 2 be its largest passing rank. Set tau[i] = alpha*R[i]/(2*N)
    and e[i,s] = I(p[i,s] <= tau[i])/tau[i]. Neither own sign influences tau[i].
    BH and indicator comparisons use inclusive <= with no tie tolerance.

    The optimized implementation takes O(N log N) time and O(N) space,
    using one sort and three rank segments per excluded pair. No N-by-N
    allocation or iterative convergence criterion is needed.

    Inputs must be nonempty finite real numeric probabilities in [0,1] and
    scalar 0 < alpha < 1. Computation is float64; alpha must permit a finite
    reciprocal at the smallest possible threshold alpha/N. Invalid shapes,
    bounds, nonnumeric/boolean dtypes, masks, and unsafe levels are rejected.
    Inputs are not mutated. Diagnostics include length-N arrays tau and R,
    the family size, and unverified validity conditions. See the module docs:
    arbitrary dependence of the resulting e-values does not remove the
    gene-independence requirement of this construction.

    alpha is the construction level, not the caller's final e-BH level.
    A prespecified alpha=.025 followed by e-BH at .05 avoids the universal
    same-level obstruction for complementary pairs. It does not guarantee
    power; see the documented fixed-mixture examples and remaining failures.
    """
    values, level, cuts = _validate(p, alpha)
    m = values.size
    order = np.argsort(values, axis=None, kind="stable")
    sorted_p = values.ravel()[order]
    ranks = np.arange(1, m + 1, dtype=np.intp)
    original_ranks = np.empty(m, dtype=np.intp)
    original_ranks[order] = ranks
    pair_ranks = original_ranks.reshape(-1, 2)
    a = pair_ranks.min(axis=1)
    b = pair_ranks.max(axis=1)

    # After deleting original ranks a < b and prepending two zeros, an
    # original rank j maps to j+2 (j<a), j+1 (a<j<b), or j (j>b).
    # Prefix maxima record the LAST passing mapped rank, not the number of
    # passing comparisons. A BH comparison may fail before a later success.
    prefix_two = np.zeros(m + 1, dtype=np.intp)
    prefix_two[1:m - 1] = np.where(sorted_p[:-2] <= cuts[2:], ranks[:-2] + 2, 0)
    np.maximum.accumulate(prefix_two, out=prefix_two)
    left = prefix_two[a - 1]

    prefix_one = np.zeros(m + 1, dtype=np.intp)
    prefix_one[1:m] = np.where(sorted_p[:-1] <= cuts[1:], ranks[:-1] + 1, 0)
    np.maximum.accumulate(prefix_one, out=prefix_one)
    middle = prefix_one[b - 1]
    middle = np.where(middle > a + 1, middle, 0)

    last_unshifted = np.max(np.where(sorted_p <= cuts, ranks, 0))
    right = np.where(last_unshifted > b, last_unshifted, 0)
    counts = np.maximum(2, np.maximum(left, np.maximum(middle, right)))
    return _result(values, level, cuts, counts, "sorted_rank_segments")
