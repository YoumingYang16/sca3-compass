"""DEVELOPMENT: training-only, known-shape empirical-Bayes pattern mixture.

This is a classical finite location mixture, not a new testing procedure. It
produces neither p-values nor an FDR guarantee. The caller must supply ONLY
outer-training genes, including training-only nuisance parameters. There is
no fold assignment or access to held genes in this module.

Related precedent for EB learning of sharing patterns: Urbut et al. (2019),
Nature Genetics 51, 187-195, and https://stephenslab.github.io/mashr/ . This
finite signed location mixture is not a reproduction of mash's covariance
mixture or its local false-sign posterior. Any downstream training-only
positive projections and signed partial-conjunction tests are outside this
module; learning a pattern here does not establish validity of such tests.

For component c, M_i | c ~ t_df(a_c b_c, variance_i * shape), with a Gaussian
branch at df=inf. ``variance`` denotes the positive scale multiplier, NOT the
Student marginal variance (which need not exist). Locations are in the input
means' units, not multiplied by sqrt(variance_i). The fixed library is zero
once plus all 80 nonzero signed supports at amplitudes 2.5, 3.5, and 4.5.

The fitted objective is sum_i log(sum_c w_c f_ic) + a sum_c log(w_c).
Pseudocount a=.02 means Dirichlet concentration 1.02, not .02. It makes the
objective strictly concave on the simplex interior. EM updates are
w_c <- (sum_i responsibility_ic + a)/(N+C*a). A small change in objective
alone is NOT a convergence certificate: we report the concave-objective
duality gap max(gradient) - dot(w, gradient), an upper bound on the remaining
objective improvement. A bounded SLSQP polish can follow 500 EM iterations.

Density convention: scipy.stats.multivariate_t documentation,
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_t.html
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.linalg import solve_triangular,cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.special import logsumexp, xlogy


def pattern_library():
    """Return fresh component-aligned (241 x 4 supports, 241 amplitudes).

    Index zero is the null with amplitude zero. Nonzero supports are in
    lexicographic order, each followed by ascending amplitude. A support
    containing two positive AND two negative entries belongs to both
    directional replicated-alternative summaries; those masses can overlap.
    """
    supports = np.array([b for b in product((-1, 0, 1), repeat=4) if any(b)], dtype=int)
    patterns = np.vstack((np.zeros((1, 4), dtype=int), np.repeat(supports, 3, axis=0)))
    amplitudes = np.r_[0., np.tile([2.5, 3.5, 4.5], len(supports))]
    return patterns, amplitudes


def _checked_inputs(means, variance, shape, df):
    means = np.asarray(means, dtype=float)
    variance = np.asarray(variance, dtype=float)
    shape = np.asarray(shape, dtype=float)
    if means.ndim != 2 or means.shape[1] != 4 or len(means) == 0:
        raise ValueError("means must be a nonempty N x 4 training array")
    if not np.isfinite(means).all():
        raise ValueError("means must be finite")
    if variance.shape != (len(means),) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("variance must be a finite positive vector with N entries")
    if (
        shape.shape != (4, 4) or not np.isfinite(shape).all()
        or not np.allclose(shape, shape.T, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError("shape must be a finite symmetric positive definite 4 x 4 matrix")
    try:
        root = np.linalg.cholesky(shape)
    except np.linalg.LinAlgError as error:
        raise ValueError("shape must be positive definite") from error
    if not np.isscalar(df) or np.isnan(df) or df <= 0:
        raise ValueError("df must be positive or positive infinity")
    return means, variance, root, float(df)


def _logpdf(means, variance, root, df, patterns, amplitudes):
    location = patterns * amplitudes[:, None]
    whitened_means = solve_triangular(root, means.T, lower=True, check_finite=False).T
    whitened_location = solve_triangular(root, location.T, lower=True, check_finite=False).T
    residual = whitened_means[:, None, :] - whitened_location[None, :, :]
    if not np.isfinite(residual).all():
        raise FloatingPointError("Whitened locations exceed floating-point range")
    # Scaled squares retain finite Student log densities even for observations
    # too extreme to square. No clipping or removal of heavy-tail observations.
    magnitude = np.max(np.abs(residual), axis=-1)
    scaled = residual / np.where(magnitude > 0, magnitude, 1.)[..., None]
    with np.errstate(divide="ignore"):
        log_q = (
            2 * np.log(magnitude) + np.log(np.einsum("ncd,ncd->nc", scaled, scaled))
            - np.log(variance[:, None])
        )
    constant = -2 * np.log(2 * np.pi) - np.log(np.diag(root)).sum() - 2 * np.log(variance)
    if np.isinf(df):
        with np.errstate(over="ignore"):
            kernel = .5 * np.exp(log_q)
        result = constant[:, None] - kernel
    else:
        # In dimension four, Gamma(df/2+2)/Gamma(df/2)=(df/2)*(df/2+1).
        # This exact simplification avoids catastrophic cancellation at large df.
        correction = np.logaddexp(0., np.log(2.) - np.log(df))
        kernel = (df / 2 + 2) * np.logaddexp(0., log_q - np.log(df))
        result = constant[:, None] + correction - kernel
    if np.isnan(result).any() or np.isposinf(result).any():
        raise FloatingPointError("Non-finite component density calculation")
    return result


def pattern_logpdf(means, variance, shape, df):
    """N x 241 normalized log densities in ``pattern_library`` order.

    Uses the supplied shape as-is, with no refitting or diagonal rescaling.
    Gaussian densities beyond representable log range may be -inf; fitting
    rejects a row for which every component is -inf rather than dropping it.
    """
    means, variance, root, df = _checked_inputs(means, variance, shape, df)
    patterns, amplitudes = pattern_library()
    return _logpdf(means, variance, root, df, patterns, amplitudes)


def _state(log_density, weights, pseudocount):
    log_weights = np.log(weights)
    joint = log_density + log_weights
    normalizer = logsumexp(joint, axis=1)
    counts = np.exp(joint - normalizer[:, None]).sum(axis=0)
    objective = float(normalizer.sum() + pseudocount * log_weights.sum())
    gradient = (counts + pseudocount) / weights
    # Concavity gives F(w*)-F(w) <= max(gradient)-w.dot(gradient).
    gap = max(0., float(gradient.max() - np.dot(weights, gradient)))
    return objective, counts, gap


def _newton_polish(log_density,weights,pseudocount,tol,max_iter):
    """Equality-constrained Newton recovery for the SAME concave objective.

    Diagonal preconditioning is numerical, not a changed penalty or model.
    No relaxed gap threshold: original normalized duality gap is required.
    """
    weights=weights.copy()
    n,c=log_density.shape
    total=n+c*pseudocount
    density=np.exp(log_density)
    objective,counts,gap=_state(log_density,weights,pseudocount)
    receipt={'attempted':True,'iterations':0,'accepted_steps':0,'line_search_evaluations':0,
        'initial_relative_gap':gap/total,'objective_history':[objective],
        'failure':None,'converged':False}
    for iteration in range(max_iter):
        if gap/total<=tol:
            break
        denominator=density@weights
        ratios=density/denominator[:,None]
        gradient=ratios.sum(0)+pseudocount/weights
        hessian=ratios.T@ratios+np.diag(pseudocount/weights**2)
        scale=1/np.sqrt(np.diag(hessian))
        try:
            factor=cho_factor(hessian*scale[:,None]*scale[None,:],lower=True,check_finite=True)
            inverse_gradient=cho_solve(factor,scale*gradient)*scale
            inverse_one=cho_solve(factor,scale)*scale
        except (np.linalg.LinAlgError,ValueError) as error:
            receipt['failure']=type(error).__name__+': '+str(error)
            break
        direction=inverse_gradient-inverse_one*(inverse_gradient.sum()/inverse_one.sum())
        step=min(1.,.99*float(np.min(-weights[direction<0]/direction[direction<0]))) if np.any(direction<0) else 1.
        accepted=False
        for _ in range(50):
            candidate=weights+step*direction
            candidate/=candidate.sum()
            value,new_counts,new_gap=_state(log_density,candidate,pseudocount)
            receipt['line_search_evaluations']+=1
            rounding=32*np.finfo(float).eps*max(1.,abs(objective),abs(value))
            if value>=objective-rounding and (value>objective+rounding or new_gap<gap):
                weights,objective,counts,gap=candidate,value,new_counts,new_gap
                receipt['objective_history'].append(objective)
                receipt['accepted_steps']+=1
                accepted=True
                break
            step*=.5
        receipt['iterations']=iteration+1
        if not accepted:
            receipt['failure']='No non-inferior lower-gap feasible Newton step'
            break
    receipt.update(converged=bool(gap/total<=tol),final_relative_gap=gap/total)
    return weights,receipt


def _replicated_summary(weights, patterns, amplitudes, sign):
    indices = np.flatnonzero((patterns == sign).sum(axis=1) >= 2)
    mass = float(weights[indices].sum())
    normalized = weights[indices] / mass
    component_dominant = int(indices[np.argmax(normalized)])
    component_entropy = float(-np.sum(xlogy(normalized, normalized)))
    supports, membership = np.unique(patterns[indices], axis=0, return_inverse=True)
    support_mass = np.bincount(membership, weights=weights[indices], minlength=len(supports))
    support_weights = support_mass / mass
    dominant = int(np.argmax(support_mass))
    same_support = indices[membership == dominant]
    amplitude_weights = weights[same_support] / support_mass[dominant]
    entropy = float(-np.sum(xlogy(support_weights, support_weights)))
    return {
        "sign": sign,
        "component_indices": indices.copy(),
        "patterns": patterns[indices].copy(),
        "amplitudes": amplitudes[indices].copy(),
        "conditional_weights": normalized,
        "mass": mass,
        "support_patterns": supports,
        "support_masses": support_mass,
        "support_conditional_weights": support_weights,
        "dominant_pattern": supports[dominant].copy(),
        "dominant_amplitude": float(amplitudes[same_support[np.argmax(amplitude_weights)]]),
        "dominant_amplitudes": amplitudes[same_support].copy(),
        "dominant_amplitude_weights": amplitude_weights,
        "dominant_mass": float(support_mass[dominant]),
        "dominant_mass_fraction": float(support_weights[dominant]),
        "entropy": entropy,
        "normalized_entropy": float(entropy / np.log(len(supports))),
        "effective_count": float(np.exp(entropy)),
        "component_dominant_index": component_dominant,
        "component_dominant_pattern": patterns[component_dominant].copy(),
        "component_dominant_amplitude": float(amplitudes[component_dominant]),
        "component_dominant_mass": float(weights[component_dominant]),
        "component_dominant_mass_fraction": float(weights[component_dominant] / mass),
        "component_entropy": component_entropy,
        "component_normalized_entropy": float(component_entropy / np.log(len(indices))),
        "component_effective_count": float(np.exp(component_entropy)),
    }


def fit_pattern_mixture(
    means, variance, shape, df, *, pseudocount=.02, max_iter=500,
    tol=1e-7, slsqp_max_iter=200, newton_max_iter=30,
):
    """Fit classical MAP mixture weights on caller-selected training genes.

    ``tol`` bounds the final duality gap divided by N+241*pseudocount.
    ``converged`` is based on that certificate, NOT an optimizer success flag.
    Set both polishing iteration limits to zero to disable all polishing.
    Newton recovery is attempted only after an otherwise monotone fit misses
    the SAME duality-gap criterion. Unconverged finite fits are
    returned with explicit diagnostics; the caller must not treat them as
    converged. No random initialization, truth labels, held data, global fit
    cache, automatic testing threshold, or posterior testing decision is used.

    Rows are canonically sorted together with variance before reductions, so
    arbitrary permutations of training rows give identical numerical output.
    Inputs are never mutated. Reported objective history contains every EM
    iterate followed, if improved, by the best feasible SLSQP iterate. Raw
    SLSQP iterates need not be monotone and are separately reported. Objective
    ties at floating-point rounding precision are broken by smaller duality
    gap; all monotonicity assertions therefore allow roundoff, not true descent.

    In each ``replicated`` sign summary, dominant pattern, entropy and
    effective_count aggregate the three amplitudes per support. The selected
    dominant_amplitude is the mode WITHIN that dominant support. Explicit
    ``component_*`` fields report the alternative component-level summary.
    """
    means, variance, root, df = _checked_inputs(means, variance, shape, df)
    if not np.isscalar(pseudocount) or not np.isfinite(pseudocount) or pseudocount <= 0:
        raise ValueError("pseudocount must be finite and strictly positive")
    if not np.isscalar(tol) or not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and strictly positive")
    for name, value in (("max_iter", max_iter), ("slsqp_max_iter", slsqp_max_iter),('newton_max_iter',newton_max_iter)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    order = np.lexsort((variance, means[:, 3], means[:, 2], means[:, 1], means[:, 0]))
    means, variance = means[order], variance[order]
    patterns, amplitudes = pattern_library()
    log_density = _logpdf(means, variance, root, df, patterns, amplitudes)
    offset = log_density.max(axis=1)
    if not np.isfinite(offset).all():
        raise FloatingPointError("At least one training row has no representable component density")
    log_density -= offset[:, None]
    objective_offset = float(offset.sum())
    if not np.isfinite(objective_offset):
        raise FloatingPointError("Training log likelihood exceeds floating-point range")
    n, components = log_density.shape
    normalizing_count = n + components * pseudocount
    if not np.isfinite(normalizing_count) or pseudocount / normalizing_count <= 0:
        raise ValueError("pseudocount is outside the representable optimization range")
    weights = np.full(components, 1 / components)
    objective, counts, gap = _state(log_density, weights, pseudocount)
    history = [objective]
    em_iterations = 0
    em_monotone = True
    for iteration in range(max_iter):
        if gap / normalizing_count <= tol:
            break
        updated = (counts + pseudocount) / normalizing_count
        updated /= updated.sum()
        new_objective, new_counts, new_gap = _state(log_density, updated, pseudocount)
        if new_objective < objective - 1e-11 * max(1., abs(objective)):
            # An actual EM descent is a numerical failure, not convergence.
            em_monotone = False
            break
        weights, objective, counts, gap = updated, new_objective, new_counts, new_gap
        history.append(objective)
        em_iterations = iteration + 1
    em_gap = gap
    slsqp = {"attempted": False, "success": None, "iterations": 0, "message": None,
             "objective_history": [], "accepted": False}
    if gap / normalizing_count > tol and slsqp_max_iter:
        slsqp["attempted"] = True
        likelihood = np.exp(log_density)
        # Every MAP optimum has w_c >= a/(N+C*a), by the EM stationarity
        # identity. These bounds do not exclude the positive-prior optimum.
        lower = pseudocount / normalizing_count
        best_weights, best_objective, best_gap = weights.copy(), objective, gap

        # A fixed linear diagonal rescaling keeps this a convex problem but
        # avoids a badly conditioned SLSQP Hessian at tiny mixture weights.
        denominator = np.einsum("nc,c->n", likelihood, weights)
        curvature = (likelihood / denominator[:, None]) ** 2
        diagonal = curvature.sum(axis=0) + pseudocount / weights ** 2
        coordinate_scale = np.sqrt(normalizing_count / diagonal)

        def objective_and_gradient(coordinates):
            w = coordinates * coordinate_scale
            denominator = np.einsum("nc,c->n", likelihood, w)
            if np.any(w <= 0) or np.any(denominator <= 0):
                return np.inf, np.zeros_like(w)
            value = np.log(denominator).sum() + pseudocount * np.log(w).sum()
            gradient = (likelihood / denominator[:, None]).sum(axis=0) + pseudocount / w
            return -float(value) / normalizing_count, -gradient * coordinate_scale / normalizing_count

        def consider(coordinates):
            nonlocal best_weights, best_objective, best_gap
            w = coordinates * coordinate_scale
            if (
                not np.isfinite(w).all() or np.any(w < lower - 1e-8)
                or abs(w.sum() - 1) > 1e-7
            ):
                return
            # Correct only tiny SLSQP feasibility rounding before evaluation.
            candidate = np.maximum(w, lower)
            candidate = candidate / candidate.sum()
            value, _, candidate_gap = _state(log_density, candidate, pseudocount)
            slsqp["objective_history"].append(value + objective_offset)
            rounding = 8 * np.finfo(float).eps * max(1., abs(best_objective))
            if value > best_objective + rounding or (
                value >= best_objective - rounding and candidate_gap < best_gap
            ):
                best_weights, best_objective, best_gap = candidate.copy(), value, candidate_gap

        fit = minimize(
            objective_and_gradient, weights / coordinate_scale, method="SLSQP", jac=True,
            bounds=[(lower / scale, 1. / scale) for scale in coordinate_scale],
            constraints={"type": "eq", "fun": lambda u: np.dot(u, coordinate_scale) - 1.,
                         "jac": lambda u: coordinate_scale.copy()},
            callback=consider,
            options={"maxiter": slsqp_max_iter, "ftol": 1e-15},
        )
        consider(fit.x)
        slsqp.update(success=bool(fit.success), iterations=int(fit.nit), message=str(fit.message),
                     linear_diagonal_preconditioning=True)
        rounding = 8 * np.finfo(float).eps * max(1., abs(objective))
        if best_objective > objective or (
            best_objective >= objective - rounding and best_gap < gap
        ):
            weights = best_weights
            objective, counts, gap = _state(log_density, weights, pseudocount)
            history.append(objective)
            slsqp["accepted"] = True
    newton={'attempted':False}
    if em_monotone and gap/normalizing_count>tol and newton_max_iter:
        weights,newton=_newton_polish(log_density,weights,pseudocount,tol,newton_max_iter)
        objective,counts,gap=_state(log_density,weights,pseudocount)
        history.extend(newton['objective_history'][1:])
        newton['objective_history']=[value+objective_offset for value in newton['objective_history']]
    converged = bool(em_monotone and gap / normalizing_count <= tol)
    return {
        "weights": weights.copy(), "patterns": patterns, "amplitudes": amplitudes,
        "replicated": {
            "positive": _replicated_summary(weights, patterns, amplitudes, 1),
            "negative": _replicated_summary(weights, patterns, amplitudes, -1),
        },
        "diagnostics": {
            "development_only": True, "training_rows": n, "components": components,
            "uses_truth_labels": False, "held_gene_access": False,
            "requires_caller_training_only_inputs": True,
            "pseudocount": float(pseudocount), "dirichlet_concentration": 1 + float(pseudocount),
            "df": df, "max_iter": int(max_iter), "tolerance": float(tol),
            "converged": converged, "em_iterations": em_iterations,
            "em_monotone": em_monotone,
            "em_relative_duality_gap": float(em_gap / normalizing_count),
            "duality_gap": gap, "relative_duality_gap": float(gap / normalizing_count),
            "objective": objective + objective_offset,
            "objective_history": [value + objective_offset for value in history],
            "objective_history_kind": "all EM iterates, best feasible SLSQP iterate, then accepted Newton recovery steps",
            "slsqp": slsqp,
            'newton':newton,
        },
    }
