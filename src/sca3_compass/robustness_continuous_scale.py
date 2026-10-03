"""Training-only continuous Gaussian effects convolved with Student noise.

Development nuisance learner, related to classical mash/empirical Bayes. This
is NOT a Student distribution with an added effect covariance. Non-null
convolutions use positive generalized Gauss--Laguerre integration; null and
Gaussian components are analytic. Numerical and inferential limits are in
docs/robustness_continuous_scale.md. This module does not choose testing rules.
"""

from __future__ import annotations

import math
import time
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import cho_factor, cho_solve, eigh_tridiagonal, solve_triangular
from scipy.optimize import least_squares, minimize
from scipy.special import expit, gammaln, logsumexp

DEFAULT_AMPLITUDES = (1., 3., 8.)
_NULL_PSEUDOCOUNT = 1.
_ALTERNATIVE_PSEUDOCOUNT = .02
_MAX_DF = 1e8
_TAIL_PROPOSAL_SWITCH = 1e6


def _positive(value, name, *, infinity=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive scalar")
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive scalar") from error
    if value <= 0 or np.isnan(value) or (not infinity and not np.isfinite(value)):
        raise ValueError(f"{name} must be positive and {'not NaN' if infinity else 'finite'}")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _inputs(means, base_variance, shape, df):
    x, v, shape = (np.asarray(a, dtype=float) for a in (means, base_variance, shape))
    if x.ndim != 2 or x.shape[1] != 4 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("means must be a nonempty finite N x 4 training array")
    if v.shape != (len(x),) or not np.isfinite(v).all() or np.any(v <= 0):
        raise ValueError("base_variance must be a finite positive length-N vector")
    if (shape.shape != (4, 4) or not np.isfinite(shape).all()
            or not np.allclose(shape, shape.T, atol=1e-12, rtol=1e-12)):
        raise ValueError("shape must be a finite symmetric positive definite 4 x 4 matrix")
    try:
        root = np.linalg.cholesky(shape)
    except np.linalg.LinAlgError as error:
        raise ValueError("shape must be positive definite") from error
    df = _positive(df, "df", infinity=True)
    if np.isfinite(df) and (df < .1 or df > _MAX_DF):
        raise ValueError("finite df must be in [.1, 1e8]; use infinity only for an actual Gaussian model")
    return x, v, root, df


def covariance_library(amplitudes=DEFAULT_AMPLITUDES, *, diagonal=True):
    """Return (C x 4 x 4 PSD covariances, JSON-safe component metadata).

    Forty nonzero signed supports modulo global sign, plus eleven diagonal
    supports of size >=2. Singletons are already rank-one. Default C=154.
    Entries are physical effect variances, never multiplied by tau or v_i.
    """
    try:
        values = tuple(_positive(a, "amplitude") for a in amplitudes)
    except TypeError as error:
        raise ValueError("amplitudes must be a nonempty iterable") from error
    if not values or len(set(values)) != len(values) or any(a > 1e100 for a in values):
        raise ValueError("amplitudes must be distinct, nonempty, positive and <= 1e100")
    if not isinstance(diagonal, (bool, np.bool_)):
        raise ValueError("diagonal must be boolean")  # noqa: TRY004 - uniform numerical API validation
    values = tuple(sorted(values))
    covariances, metadata = [np.zeros((4, 4))], [{"kind": "null", "support": [0]*4, "amplitude": 0.}]
    for b in product((-1, 0, 1), repeat=4):
        nonzero = [k for k in b if k]
        if not nonzero or nonzero[0] != 1:
            continue
        vector = np.asarray(b, dtype=float)
        for amplitude in values:
            covariances.append(amplitude**2 * np.outer(vector, vector))
            metadata.append({"kind": "rank_one", "support": list(b), "amplitude": amplitude})
    if diagonal:
        for b in product((0, 1), repeat=4):
            if sum(b) < 2:
                continue
            for amplitude in values:
                covariances.append(amplitude**2 * np.diag(b))
                metadata.append({"kind": "diagonal", "support": list(b), "amplitude": amplitude})
    return np.asarray(covariances), metadata


def _covariances(value):
    u = np.asarray(value, dtype=float)
    if (u.ndim != 3 or u.shape[1:] != (4, 4) or len(u) == 0 or not np.isfinite(u).all()
            or not np.allclose(u, u.transpose(0, 2, 1), atol=1e-12, rtol=1e-12)):
        raise ValueError("covariances must be nonempty finite symmetric C x 4 x 4")
    if np.any(u[0] != 0):
        raise ValueError("the first covariance must be exactly zero")
    return u


@lru_cache(maxsize=32)
def _gamma_rule(order, shape):
    """Normalized Gauss-Laguerre rule for Gamma(shape, rate=1).

    Golub-Welsch weights avoid overflow of Gamma(shape) in an unnormalized
    roots_genlaguerre call. Zeros due to eigenvector underflow remain explicit
    zero-weight nodes. The symmetric tridiagonal Jacobi matrix is standard.
    """
    j = np.arange(order, dtype=float)
    diagonal = 2*j + shape
    k = np.arange(1, order, dtype=float)
    off = np.sqrt(k) * np.sqrt(k + shape - 1)
    nodes, vectors = eigh_tridiagonal(diagonal, off, lapack_driver="stev")
    weights = vectors[0]**2
    if (not np.isfinite(nodes).all() or np.any(nodes <= 0)
            or not np.isfinite(weights).all() or weights.sum() <= 0):
        raise FloatingPointError("Invalid normalized Laguerre rule")
    weights /= weights.sum()
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def _log_square(x):
    with np.errstate(divide="ignore"):
        return 2*np.log(np.abs(x))


class _Convolution:
    def __init__(self, x, variance, root, df, covariances, order, batch_size, component_batch):
        self.x = solve_triangular(root, x.T, lower=True, check_finite=False).T
        if not np.isfinite(self.x).all():
            raise FloatingPointError("Whitened data exceed floating-point range")
        self.log_variance, self.df = np.log(variance), df
        self.n, self.components = len(x), len(covariances)
        self.batch_size, self.component_batch, self.order = batch_size, component_batch, order
        self.constant = -2*np.log(2*np.pi) - np.log(np.diag(root)).sum()
        self.log_q = logsumexp(_log_square(self.x), axis=1)
        self.log_eigenvalues = np.empty((self.components, 4))
        self.rotations = np.empty((self.components, 4, 4))
        self.null = np.zeros(self.components, dtype=bool)
        self.clipped_eigenvalues = 0
        for c, u in enumerate(covariances):
            left = solve_triangular(root, u, lower=True, check_finite=False)
            whitened = solve_triangular(root, left.T, lower=True, check_finite=False).T
            whitened = (whitened + whitened.T)/2
            eigenvalues, rotation = np.linalg.eigh(whitened)
            tolerance = 64*np.finfo(float).eps*max(1., np.max(np.abs(eigenvalues)))
            if np.min(eigenvalues) < -tolerance:
                raise ValueError("Effect covariances must be positive semidefinite")
            # Roundoff-level zero eigenvalues of exact rank-one matrices must
            # not turn a nominally singular covariance into a tiny full-rank one.
            self.clipped_eigenvalues += int(np.count_nonzero((eigenvalues != 0) & (np.abs(eigenvalues) <= tolerance)))
            eigenvalues[np.abs(eigenvalues) <= tolerance] = 0.
            with np.errstate(divide="ignore"):
                self.log_eigenvalues[c] = np.log(eigenvalues)
            self.rotations[c] = rotation
            self.null[c] = bool(np.all(u == 0))
        self.evaluations = 0

    def _nodes(self, log_noise, log_y2, log_u):
        """Component-adapted Gamma rule plus prior rule, positive partition.

        A smooth extreme-observation adjustment tends to the null-posterior
        rate when bounded Gaussian effects become negligible in Student tails.
        Node movement and the partition derivative are included explicitly.
        This is an integration proposal, not a fitted distribution parameter.
        """
        a = self.df/2
        log_a = np.log(a)
        log_reference = np.logaddexp(log_noise[:, None, None], log_u[None, :, :])
        log_terms = log_y2-log_reference
        log_noise_q = log_y2-log_noise[:, None, None]
        transition = 2*(log_terms-np.log(_TAIL_PROPOSAL_SWITCH))
        log_blend = -np.logaddexp(0., -transition)
        log_complement = -np.logaddexp(0., transition)
        log_effective = np.logaddexp(log_complement+log_terms, log_blend+log_noise_q)
        log_half_q = logsumexp(log_effective, axis=2)-np.log(2.)
        log_b = np.logaddexp(log_a, log_half_q)
        log_fraction = log_noise[:, None, None]-log_reference
        with np.errstate(divide="ignore", invalid="ignore"):
            log_effect_fraction = np.log(-np.expm1(log_fraction))
        derivative_terms = np.logaddexp(
            log_complement+log_terms+log_fraction,
            np.logaddexp(log_blend+log_noise_q,
                        np.log(2.)+log_blend+log_complement+log_fraction+log_noise_q+log_effect_fraction))
        b_gradient = -.5*np.sum(np.exp(derivative_terms-log_b[:, :, None]), axis=2)
        xp, wp = _gamma_rule(self.order, a)
        xt, wt = _gamma_rule(self.order, a+2)
        prior = np.broadcast_to(np.log(xp)-log_a, (*log_b.shape, self.order))
        tail = np.log(xt)[None, None, :]-log_b[:, :, None]
        log_nodes = np.concatenate((prior, tail), axis=2)
        node_gradient = np.concatenate((np.zeros_like(prior), np.broadcast_to(-b_gradient[:, :, None], tail.shape)), axis=2)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            delta = np.exp(log_half_q[:, :, None]+log_nodes)
            ratio = (a*np.logaddexp(0., log_half_q-log_a)[:, :, None]
                     +2*(log_b[:, :, None]+log_nodes)-log_a-np.log(a+1)-delta)
            log_weights = np.log(np.r_[wp, wt])[None, None, :]-np.logaddexp(0., ratio)
            ratio_gradient = ((a+2-np.exp(log_b[:, :, None]+log_nodes))*b_gradient[:, :, None]
                              +(2-delta)*node_gradient)
            partition_gradient = -expit(ratio)*ratio_gradient
        if np.isnan(log_weights).any() or not np.isfinite(log_nodes).all():
            raise FloatingPointError("Unrepresentable Gamma proposal partition")
        return log_nodes, log_weights, node_gradient, partition_gradient

    def densities(self, log_tau, *, analytic_null=True):
        """Component log densities and derivatives with respect to log(tau)."""
        self.evaluations += 1
        result = np.empty((self.n, self.components))
        scores = np.empty_like(result)
        for first in range(0, self.n, self.batch_size):
            last = min(self.n, first+self.batch_size)
            for c0 in range(0, self.components, self.component_batch):
                c1 = min(self.components, c0+self.component_batch)
                projected = np.einsum("nd,cdk->nck", self.x[first:last], self.rotations[c0:c1])
                if not np.isfinite(projected).all():
                    raise FloatingPointError("Rotated observations exceed floating-point range")
                log_y2 = _log_square(projected)
                if np.isinf(self.df):
                    nodes, weights, node_gradient, partition_gradient = [np.zeros((last-first, c1-c0, 1)) for _ in range(4)]
                else:
                    nodes, weights, node_gradient, partition_gradient = self._nodes(
                        log_tau+self.log_variance[first:last], log_y2, self.log_eigenvalues[c0:c1])
                log_noise = log_tau+self.log_variance[first:last, None, None]-nodes
                log_d = np.logaddexp(log_noise[:, :, :, None], self.log_eigenvalues[None, c0:c1, None, :])
                with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                    squared = np.exp(log_y2[:, :, None, :]-log_d)
                    noise_fraction = np.exp(log_noise[:, :, :, None]-log_d)
                    conditional_score = .5*np.sum(noise_fraction*(squared-1), axis=3)*(1-node_gradient)+partition_gradient
                    log_terms = self.constant-.5*np.sum(log_d+squared, axis=3)+weights
                log_density = logsumexp(log_terms, axis=2)
                with np.errstate(invalid="ignore", under="ignore"):
                    responsibilities = np.exp(log_terms-log_density[:, :, None])
                    contribution = np.zeros_like(conditional_score)
                    np.multiply(responsibilities, conditional_score, out=contribution,
                                where=responsibilities > 0)
                result[first:last, c0:c1] = log_density
                scores[first:last, c0:c1] = contribution.sum(axis=2)
            if analytic_null:
                z = self.log_q[first:last]-self.log_variance[first:last]-log_tau
                if np.isinf(self.df):
                    with np.errstate(over="ignore"):
                        radial = .5*np.exp(z)
                    null_log = self.constant-2*(log_tau+self.log_variance[first:last])-radial
                    null_score = radial-2
                else:
                    z -= np.log(self.df)
                    coefficient = self.df/2+2
                    null_log = (self.constant+np.log1p(2/self.df)
                                -2*(log_tau+self.log_variance[first:last])-coefficient*np.logaddexp(0., z))
                    null_score = coefficient*np.exp(-np.logaddexp(0., -z))-2
                result[first:last, self.null] = null_log[:, None]
                scores[first:last, self.null] = null_score[:, None]
        if not np.isfinite(result).all() or not np.isfinite(scores).all():
            raise FloatingPointError("A component density or score is not representable; no rows were discarded")
        return result, scores


def continuous_component_logpdf(means, base_variance, shape, df, tau=1., *,
                                amplitudes=DEFAULT_AMPLITUDES, diagonal=True,
                                covariances=None, quadrature_order=32,
                                batch_size=64, component_batch=16, return_score=False):
    """Diagnostic density API; finite Student convolution is numerical.

    Optional covariances are fixed, with the first matrix exactly zero. Fitter
    exposes the same option solely for prespecified ablations/tests, not for
    learning dictionaries from held observations.
    """
    x, variance, root, df = _inputs(means, base_variance, shape, df)
    tau = _positive(tau, "tau")
    order = _integer(quadrature_order, "quadrature_order", 4)
    if covariances is None:
        covariances, _ = covariance_library(amplitudes, diagonal=diagonal)
    kernel = _Convolution(x, variance, root, df, _covariances(covariances), order,
                          _integer(batch_size, "batch_size"), _integer(component_batch, "component_batch"))
    density, score = kernel.densities(np.log(tau))
    return (density, score) if return_score else density


def _penalty(components, null_pseudocount=1., alternative_pseudocount=.02):
    penalty = np.full(components, alternative_pseudocount/max(1, components-1))
    penalty[0] = null_pseudocount
    return penalty


def _weight_state(density, weights, pseudocount):
    offset = density.max(axis=1)
    likelihood = np.exp(density-offset[:, None])
    denominator = likelihood @ weights
    if np.any(denominator <= 0) or not np.isfinite(denominator).all():
        raise FloatingPointError("Invalid mixture denominator")
    ratio = likelihood/denominator[:, None]
    gradient = ratio.sum(axis=0)+pseudocount/weights
    normalizer = len(density)+np.sum(pseudocount)
    if not np.isfinite(gradient).all() or not np.isfinite(normalizer):
        raise FloatingPointError("Mixture gradient exceeds floating-point range")
    terms = np.r_[offset, np.log(denominator), pseudocount*np.log(weights)]
    objective = math.fsum(terms.tolist())
    return {"objective": objective, "objective_absolute_sum": float(np.abs(terms).sum()),
            "gradient": gradient, "ratio": ratio, "normalizer": normalizer,
            "weight_kkt_residual": float(np.max(np.abs(gradient/normalizer-1))),
            "conditional_weight_gap": max(0., float(gradient.max()-normalizer))}


def _fit_weights(density, initial=None, *, pseudocount=None, tol=1e-8, max_iter=100):
    """Damped equality-constrained Newton solve of a strictly concave MAP."""
    c = density.shape[1]
    if pseudocount is None:
        pseudocount = _penalty(c)
    weights = np.full(c, 1/c) if initial is None else np.array(initial, copy=True)
    weights /= weights.sum()
    state = _weight_state(density, weights, pseudocount)
    for iteration in range(max_iter+1):
        if state["weight_kkt_residual"] <= tol:
            return weights, state, iteration
        if iteration == max_iter:
            break
        hessian = state["ratio"].T @ state["ratio"]
        hessian.flat[::c+1] += pseudocount/weights**2
        factor = cho_factor(hessian, lower=True, check_finite=False)
        # Subtract the equality multiplier before solving to reduce cancellation.
        solved = cho_solve(factor, np.column_stack((state["gradient"]-state["normalizer"], np.ones(c))),
                          check_finite=False)
        direction = solved[:, 0] - solved[:, 1]*(solved[:, 0].sum()/solved[:, 1].sum())
        slope = float(np.dot(state["gradient"]-state["normalizer"], direction))
        negative = direction < 0
        step = min(1., .99*float(np.min(-weights[negative]/direction[negative]))) if negative.any() else 1.
        accepted = False
        for _ in range(45):
            trial = weights+step*direction
            trial /= trial.sum()
            if np.all(trial > 0):
                candidate = _weight_state(density, trial, pseudocount)
                rounding = 32*np.finfo(float).eps*max(1., state["objective_absolute_sum"], candidate["objective_absolute_sum"])
                if (candidate["objective"] >= state["objective"]+1e-4*step*slope-rounding
                        and (candidate["objective"] > state["objective"]+rounding
                             or candidate["weight_kkt_residual"] < state["weight_kkt_residual"])):
                    weights, state, accepted = trial, candidate, True
                    break
            step *= .5
        if not accepted:
            raise FloatingPointError("Weight Newton line search failed before KKT convergence")
    raise FloatingPointError(f"Weight iteration limit; KKT residual={state['weight_kkt_residual']:.6g}")


def _certificate(state, eta, bounds, n, tol):
    score = state["log_tau_gradient"]/n
    boundary = "lower" if eta <= bounds[0] else "upper" if eta >= bounds[1] else None
    residual = max(0., score) if boundary == "lower" else max(0., -score) if boundary == "upper" else abs(score)
    return {"log_tau_gradient_per_row": float(score), "projected_log_tau_residual": float(residual),
            "weight_kkt_residual": state["weight_kkt_residual"], "active_bound": boundary,
            "optimizer_converged": bool(max(residual, state["weight_kkt_residual"]) <= tol)}


def _fit_weights_floor(density, minimum_null_weight, initial=None, *, pseudocount=None,
                       tol=1e-8, max_iter=100):
    """Exact concave weight subproblem on w0>=b, NOT a null-fraction estimate.

    First solve the old unconstrained problem. If infeasible, strict
    concavity implies the constrained optimum has w0=b. Solve that face
    with damped equality-constrained Newton; certify its full constrained
    Frank-Wolfe gap, including the inward direction off the active face.
    """
    if (isinstance(minimum_null_weight,(bool,np.bool_)) or not np.isscalar(minimum_null_weight)
            or not np.isfinite(minimum_null_weight) or not 0<=minimum_null_weight<1):
        raise ValueError('minimum_null_weight must be a finite scalar in [0,1)')
    weights,state,iterations=_fit_weights(density,initial,pseudocount=pseudocount,tol=tol,max_iter=max_iter)
    if weights[0]>=minimum_null_weight:
        return weights,state,iterations
    c=len(weights)
    penalty=_penalty(c) if pseudocount is None else pseudocount
    remainder=1-minimum_null_weight
    weights[1:]*=remainder/weights[1:].sum()
    weights[0]=minimum_null_weight

    def constrained(w):
        result=_weight_state(density,w,penalty)
        gradient=result['gradient']
        # Feasible vertices are b*e0+(1-b)*ej, including j=0.
        gap=max(0.,float(minimum_null_weight*gradient[0]
            +remainder*np.max(gradient)-np.dot(w,gradient)))
        multiplier=float(np.dot(w[1:],gradient[1:])/remainder)
        residual=max(float(np.max(np.abs(gradient[1:]-multiplier))),
            max(0.,float(gradient[0]-multiplier)))/result['normalizer']
        result.update(conditional_weight_gap=gap,weight_kkt_residual=residual,
            null_weight_floor_active=True,null_weight_floor=minimum_null_weight)
        return result

    state=constrained(weights)
    for iteration in range(max_iter+1):
        if state['weight_kkt_residual']<=tol:
            return weights,state,iterations+iteration
        if iteration==max_iter:
            break
        ratio=state['ratio'][:,1:]
        hessian=ratio.T@ratio
        hessian.flat[::c]+=penalty[1:]/weights[1:]**2
        scaling=np.sqrt(np.diag(hessian))
        normalized=hessian/scaling[:,None]/scaling[None,:]
        factor=cho_factor(normalized,lower=True,check_finite=False)
        gradient=state['gradient'][1:]
        multiplier=float(np.dot(weights[1:],gradient)/remainder)
        rhs=np.column_stack((gradient-multiplier,np.ones(c-1)))
        solved=cho_solve(factor,rhs/scaling[:,None],check_finite=False)/scaling[:,None]
        direction=solved[:,0]-solved[:,1]*(solved[:,0].sum()/solved[:,1].sum())
        slope=float(np.dot(gradient-multiplier,direction))
        negative=direction<0
        step=min(1.,.99*float(np.min(-weights[1:][negative]/direction[negative]))) if negative.any() else 1.
        accepted=False
        for _ in range(45):
            trial=weights.copy();trial[1:]+=step*direction
            trial[1:]*=remainder/trial[1:].sum()
            if np.all(trial>0):
                candidate=constrained(trial)
                rounding=32*np.finfo(float).eps*max(1.,state['objective_absolute_sum'],candidate['objective_absolute_sum'])
                if (candidate['objective']>=state['objective']+1e-4*step*slope-rounding
                        and (candidate['objective']>state['objective']+rounding
                            or candidate['weight_kkt_residual']<state['weight_kkt_residual'])):
                    weights,state,accepted=trial,candidate,True
                    break
            step*=.5
        if not accepted:
            raise FloatingPointError('Null-floor Newton line search failed before constrained KKT convergence')
    raise FloatingPointError(f"Null-floor weight limit; KKT residual={state['weight_kkt_residual']:.6g}")


def _fit_start(kernel, initial_tau, bounds, max_iter, tol, weight_max_iter, pseudocount, minimum_null_weight=0.):
    cache = []
    events = []
    history = []
    weight_iterations = 0

    def evaluate(eta):
        nonlocal weight_iterations
        eta = float(np.clip(np.asarray(eta).ravel()[0], *bounds))
        for saved_eta, state in cache:
            if eta == saved_eta:
                return state
        density, component_scores = kernel.densities(eta)
        initial = min(cache, key=lambda item: abs(eta-item[0]))[1]["weights"] if cache else None
        if minimum_null_weight:
            weights,state,iterations=_fit_weights_floor(density,minimum_null_weight,initial,
                pseudocount=pseudocount,tol=min(tol*.1,1e-8),max_iter=weight_max_iter)
        else:
            weights, state, iterations = _fit_weights(density, initial, pseudocount=pseudocount,
                                                     tol=min(tol*.1, 1e-8), max_iter=weight_max_iter)
        weight_iterations += iterations
        responsibilities = state["ratio"]*weights
        state["log_tau_gradient"] = float(np.sum(responsibilities*component_scores))
        state["weights"] = weights
        state["eta"] = eta
        state.pop("ratio")
        state.pop("gradient")
        cache.append((eta, state))
        if len(cache) > 8:
            cache.pop(0)
        history.append({"tau": float(np.exp(eta)), "objective": state["objective"],
                        "log_tau_gradient_per_row": state["log_tau_gradient"]/kernel.n})
        return state

    def objective(eta):
        state = evaluate(eta)
        return -state["objective"]/kernel.n, np.array([-state["log_tau_gradient"]/kernel.n])

    eta = float(np.clip(np.log(initial_tau), *bounds))
    selected, solver, retry = None, None, None
    initial_evaluations = kernel.evaluations
    try:
        selected = evaluate(eta)
        if max_iter:
            result = minimize(objective, [eta], method="L-BFGS-B", jac=True, bounds=[bounds],
                              options={"maxiter": max_iter, "gtol": tol*.1, "ftol": 1e-14, "maxls": 30})
            eta = float(result.x[0])
            selected = evaluate(eta)
            solver = {"success": bool(result.success), "message": str(result.message),
                      "iterations": int(result.nit), "evaluations": int(result.nfev)}
            if not result.success:
                events.append({"stage": "profile_optimizer", "message": str(result.message)})
            if not _certificate(selected, eta, bounds, kernel.n, tol)["optimizer_converged"]:
                # Solve the projected score only as a numerical polish, never
                # accept an inferior objective as a successful replacement.
                before = selected

                def projected(value):
                    e = float(value[0])
                    score = evaluate(e)["log_tau_gradient"]/kernel.n
                    return np.array([e-np.clip(e+score, *bounds)])

                polished = least_squares(projected, [eta], bounds=([bounds[0]], [bounds[1]]),
                                        jac="3-point", max_nfev=30, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                after = evaluate(polished.x[0])
                rounding = 32*np.finfo(float).eps*max(1., before["objective_absolute_sum"], after["objective_absolute_sum"])
                accepted = (after["objective"] >= before["objective"]-rounding
                            and _certificate(after, after["eta"], bounds, kernel.n, tol)["optimizer_converged"])
                retry = {"success": bool(polished.success), "message": str(polished.message),
                         "evaluations": int(polished.nfev), "accepted": bool(accepted),
                         "before_objective": before["objective"], "after_objective": after["objective"]}
                if accepted:
                    selected, eta = after, after["eta"]
                else:
                    events.append({"stage": "score_polish", "message": "Polish did not meet score and objective safeguards"})
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
        events.append({"stage": "numeric_exception", "message": str(error)})
        # Retain a finite attempted point, but never certify this failed start.
        if cache:
            _, selected = max(cache, key=lambda item: item[1]["objective"])
            eta = selected["eta"]
    result = {"initial_tau": float(initial_tau), "solver": solver, "score_polish": retry,
              "numerical_failure_events": events, "numerical_failure_count": len(events),
              "evaluations": kernel.evaluations-initial_evaluations, "weight_iterations": weight_iterations,
              "objective_history": history, "weights": None, "tau": None, "objective": None,
              "optimizer_converged": False}
    if selected is not None:
        result.update(_certificate(selected, eta, bounds, kernel.n, tol))
        if any(event["stage"] == "numeric_exception" for event in events):
            result["optimizer_converged"] = False
        result.update(tau=float(np.exp(eta)), objective=selected["objective"], weights=selected["weights"].tolist(),
                      conditional_weight_gap=selected["conditional_weight_gap"])
    result["status"] = "converged" if result["optimizer_converged"] else "not_converged"
    return result


def _quadrature_check(kernel, refined, tau, weights, tolerance):
    old_log, old_score = kernel.densities(np.log(tau))
    new_log, new_score = refined.densities(np.log(tau))
    logw = np.log(weights)
    old_mix, new_mix = logsumexp(old_log+logw, axis=1), logsumexp(new_log+logw, axis=1)
    old_gradient = np.sum(np.exp(old_log+logw-old_mix[:, None])*old_score, axis=1)
    new_gradient = np.sum(np.exp(new_log+logw-new_mix[:, None])*new_score, axis=1)
    log_difference = float(np.max(np.abs(old_mix-new_mix)))
    score_difference = float(np.max(np.abs(old_gradient-new_gradient)))
    # The same mixture value can hide compensating component errors. Audit
    # the likelihood's weight gradient as well (the MAP penalty cancels).
    old_weight_gradient = np.exp(old_log-old_mix[:, None]).sum(axis=0)
    new_weight_gradient = np.exp(new_log-new_mix[:, None]).sum(axis=0)
    weight_difference = float(np.max(np.abs(new_weight_gradient-old_weight_gradient))/kernel.n)
    return {"order": kernel.order, "comparison_order": refined.order,
            "max_component_log_difference": float(np.max(np.abs(old_log-new_log))),
            "max_mixture_log_difference": log_difference, "max_mixture_score_difference": score_difference,
            "max_weight_gradient_difference_per_row": weight_difference,
            "objective_difference": float(np.sum(new_mix-old_mix)), "tolerance": tolerance,
            "passed": bool(max(log_difference, score_difference, weight_difference) <= tolerance)}


def _reference_check(kernel, tau, weights, tolerance):
    """Independent adaptive log-precision integration of the fitted mixture.

    Only an audit, not an optimization shortcut. Uses the true fixed-lambda
    score, not the moving-Laguerre-node derivative. Every supplied training row
    is checked. Gamma Chernoff bounds control the truncated density and score
    integrals; adaptive integration's internal error remains an estimate.
    """
    density, component_score = kernel.densities(np.log(tau))
    mixture = logsumexp(density+np.log(weights), axis=1)
    mixture_score = np.sum(np.exp(density+np.log(weights)-mixture[:, None])*component_score, axis=1)
    if np.isinf(kernel.df) or np.all(kernel.null):
        return {"passed": True, "reason": "Analytic component densities"}
    a, eta = kernel.df/2, np.log(tau)
    # Gamma(a,a) log-density of log(lambda), evaluated by a deviance form.
    # Stirling's correction avoids subtracting O(a log a) for large a.
    if a >= 1e4:
        gamma_constant = .5*np.log(a/(2*np.pi))-1/(12*a)+1/(360*a**3)
    else:
        gamma_constant = a*np.log(a)-gammaln(a)-a
    selected = ~kernel.null
    log_u = kernel.log_eigenvalues[selected]
    records, failures = [], []
    reference_weight_gradient = np.zeros(kernel.components)
    expected_weight_gradient = np.exp(density-mixture[:, None]).sum(axis=0)
    for i in range(kernel.n):
        projection = np.einsum("d,cdk->ck", kernel.x[i], kernel.rotations[selected])
        log_y2 = _log_square(projection)
        log_r = eta+kernel.log_variance[i]
        log_b = np.logaddexp(np.log(a), kernel.log_q[i]-log_r-np.log(2.))
        tail_center = np.log(a+2)-log_b
        lower = min(-40., tail_center-40.)
        upper = max(10., np.log((a+2)/a)+5.)
        # Component-relative integration gives each component comparable error
        # scale, including tiny-weight components needed for the MAP gradient.
        log_coefficient = kernel.constant-2*log_r-np.min(density[i, selected])

        def tail_log_bound(moment, lower=lower, upper=upper, log_coefficient=log_coefficient):
            k = a+moment
            log_low, log_high = np.log(a)+lower, np.log(a)+upper
            low = min(0., k*(log_low-np.log(k))+k-np.exp(log_low)) if log_low < np.log(k) else 0.
            high = min(0., k*(log_high-np.log(k))+k-np.exp(log_high)) if log_high > np.log(k) else 0.
            log_moment = sum(np.log1p(j/a) for j in range(moment))
            return log_coefficient+log_moment+np.logaddexp(low, high)

        for _ in range(12):
            density_tail = tail_log_bound(2, lower, upper)
            score_tail = np.logaddexp(np.log(2.)+density_tail,
                                     -np.log(2.)+kernel.log_q[i]-log_r+tail_log_bound(3, lower, upper))
            if max(density_tail, score_tail) <= np.log(1e-10):
                break
            lower -= 20.
            upper += 2.
        else:
            failures.append({"row": i, "message": "Could not bound the omitted integration tails"})
            continue
        centers = [0., tail_center]
        widths = [1/np.sqrt(a), 1/np.sqrt(a+2)]
        points = sorted({float(np.clip(center+multiple*width, lower, upper))
                         for center, width in zip(centers, widths) for multiple in [-8., -2., 0., 2., 8.]})
        normalizer = density[i, selected]

        def integrand(log_lambda, log_r=log_r, log_y2=log_y2, normalizer=normalizer):
            log_d = np.logaddexp(log_r-log_lambda, log_u)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                squared = np.exp(log_y2-log_d)
                log_normal = kernel.constant-.5*np.sum(log_d+squared, axis=1)
                radial_score = .5*np.sum(np.exp(log_r-log_lambda-log_d)*(squared-1), axis=1)
                log_gamma = gamma_constant+a*(log_lambda-np.expm1(log_lambda))
                relative_density = np.exp(log_normal+log_gamma-normalizer)
                score_parts = np.zeros_like(relative_density)
                np.multiply(relative_density, radial_score, out=score_parts, where=relative_density > 0)
            return np.r_[relative_density, score_parts]

        try:
            integral, error, info = quad_vec(integrand, lower, upper, points=points, epsabs=1e-8,
                                             epsrel=1e-8, limit=2000, full_output=True)
            count = int(selected.sum())
            if not info.success or not np.isfinite(integral).all() or np.any(integral[:count] <= 0):
                raise FloatingPointError(f"Adaptive integration failed: {info.message}")
            reference_log = density[i].copy()
            reference_log[selected] += np.log(integral[:count])
            reference_score = component_score[i].copy()
            reference_score[selected] = integral[count:]/integral[:count]
            reference_mix = logsumexp(reference_log+np.log(weights))
            reference_mix_score = np.sum(np.exp(reference_log+np.log(weights)-reference_mix)*reference_score)
            reference_weight_gradient += np.exp(reference_log-reference_mix)
            log_difference = float(reference_mix-mixture[i])
            score_difference = float(reference_mix_score-mixture_score[i])
            records.append({"row": i, "log_difference": log_difference, "score_difference": score_difference,
                            "estimated_absolute_error": float(error), "evaluations": int(info.neval),
                            "log_precision_bounds": [float(lower), float(upper)],
                            "relative_density_tail_bound": float(np.exp(density_tail)),
                            "relative_score_tail_bound": float(np.exp(score_tail)),
                            "log_relative_density_tail_bound": float(density_tail),
                            "log_relative_score_tail_bound": float(score_tail),
                            "max_component_log_difference": float(np.max(np.abs(reference_log-density[i]))),
                            "max_component_score_difference": float(np.max(np.abs(reference_score-component_score[i])))})
        except (FloatingPointError, OverflowError, ValueError) as error:
            failures.append({"row": i, "message": str(error)})
    maximum_log = max((abs(row['log_difference']) for row in records), default=None)
    maximum_score = max((abs(row['score_difference']) for row in records), default=None)
    weight_difference = float(np.max(np.abs(reference_weight_gradient-expected_weight_gradient))/kernel.n) if not failures else None
    return {"passed": bool(not failures and maximum_log is not None and max(maximum_log, maximum_score, weight_difference) <= tolerance),
            "max_mixture_log_difference": maximum_log, "max_mixture_score_difference": maximum_score,
            "max_weight_gradient_difference_per_row": weight_difference,
            "tolerance": tolerance, "rows": records, "failures": failures,
            "method": "Independent adaptive Gauss-Kronrod on log precision; Gamma-Chernoff density/score tail bounds"}


def fit_continuous_scale(means, base_variance, shape, df, *, amplitudes=DEFAULT_AMPLITUDES,
                         diagonal=True, covariances=None, tau_bounds=(1e-3, 1e3),
                         initial_taus=(.1, 1., 10.), max_iter=80, tol=1e-6,
                         weight_max_iter=100, quadrature_order=32,
                         max_quadrature_order=512, quadrature_tol=2e-4,
                         batch_size=64, component_batch=16,
                         null_pseudocount=_NULL_PSEUDOCOUNT,
                         alternative_pseudocount=_ALTERNATIVE_PSEUDOCOUNT,
                         minimum_null_weight=0.):
    """Fit outer-TRAINING rows only; return JSON-safe results and all attempts.

    ``variance_multiplier`` is a multiplier of conditional residual scale,
    not standard deviation and (for Student noise) not marginal covariance.
    ``converged`` requires numerical stationarity AND a successful doubled
    quadrature diagnostic. It is NOT consistency, identifiability or FDR.
    A finite but unaccepted estimate is retained; there is no silent fallback.
    """
    started = time.perf_counter()
    x, variance, root, df = _inputs(means, base_variance, shape, df)
    max_iter, weight_max_iter = _integer(max_iter, "max_iter", 0), _integer(weight_max_iter, "weight_max_iter")
    order, maximum = _integer(quadrature_order, "quadrature_order", 4), _integer(max_quadrature_order, "max_quadrature_order", 8)
    if maximum < 2*order:
        raise ValueError("max_quadrature_order must allow at least one doubled-order check")
    batch_size, component_batch = _integer(batch_size, "batch_size"), _integer(component_batch, "component_batch")
    tol, quadrature_tol = _positive(tol, "tol"), _positive(quadrature_tol, "quadrature_tol")
    null_pseudocount = _positive(null_pseudocount, "null_pseudocount")
    alternative_pseudocount = _positive(alternative_pseudocount, "alternative_pseudocount")
    if (isinstance(minimum_null_weight,(bool,np.bool_)) or not np.isscalar(minimum_null_weight)
            or not np.isfinite(minimum_null_weight) or not 0<=minimum_null_weight<1):
        raise ValueError('minimum_null_weight must be a finite scalar in [0,1)')
    try:
        taus = tuple(_positive(t, "initial_tau") for t in initial_taus)
        limits = tuple(_positive(t, "tau_bound") for t in tau_bounds)
    except TypeError as error:
        raise ValueError("initial_taus and tau_bounds must be iterables") from error
    if not taus or len(limits) != 2 or limits[0] >= limits[1] or np.log(limits[0]) >= np.log(limits[1]):
        raise ValueError("initial_taus must be nonempty and tau_bounds strictly increasing")
    bounds = (float(np.log(limits[0])), float(np.log(limits[1])))
    if covariances is None:
        u, metadata = covariance_library(amplitudes, diagonal=diagonal)
    else:
        u = _covariances(covariances)
        metadata = [{"kind": "prespecified", "index": c} for c in range(len(u))]
    u = _covariances(u)
    penalty = _penalty(len(u), null_pseudocount, alternative_pseudocount)
    if not np.isfinite(penalty.sum()):
        raise ValueError("Sum of weight penalties must be finite")
    # Canonical ordering makes all reduction/optimizer decisions row-order
    # invariant on the same runtime, without changing the statistical sample.
    permutation = np.lexsort(tuple(np.column_stack((x, variance)).T[::-1]))
    x, variance = x[permutation], variance[permutation]
    attempts, selected = [], None
    diagnostics = {"training_rows": len(x), "components": len(u), "df": None if np.isinf(df) else df,
                   "gaussian": bool(np.isinf(df)), "weight_log_penalty": penalty.tolist(),
                   "null_pseudocount": null_pseudocount, "alternative_pseudocount_total": alternative_pseudocount,
                   "tau_bounds": list(limits), "tolerance": tol,
                   "quadrature_tolerance": quadrature_tol, "batch_size": batch_size,
                   "component_batch": component_batch, "max_iter": max_iter,
                   "weight_max_iter": weight_max_iter, "attempts": attempts,
                   "input_provenance": "All supplied rows AND nuisance inputs must be training-only or fixed"}
    while True:
        initialization_started = time.perf_counter()
        kernel = _Convolution(x, variance, root, df, u, order, batch_size, component_batch)
        initialization_seconds = time.perf_counter()-initialization_started
        fit_started = time.perf_counter()
        starts = [_fit_start(kernel, float(np.clip(t, *limits)), bounds, max_iter, tol, weight_max_iter, penalty, minimum_null_weight) for t in taus]
        candidates = [(i, s) for i, s in enumerate(starts) if s["objective"] is not None]
        attempt = {"quadrature_order": None if np.isinf(df) else order, "starts": starts,
                   "eigenvalues_clipped_at_roundoff": kernel.clipped_eigenvalues,
                   "selected_start": None, "quadrature_check": None, "reference_check": None,
                   "initialization_seconds": initialization_seconds,
                   "fit_seconds": time.perf_counter()-fit_started,
                   "quadrature_check_seconds": 0., "reference_check_seconds": 0.}
        attempts.append(attempt)
        if not candidates:
            selected = None
            break
        index, selected = max(candidates, key=lambda item: item[1]["objective"])
        attempt["selected_start"] = index
        if np.isinf(df) or np.all(kernel.null):
            attempt["quadrature_check"] = {"passed": True, "reason": "All component densities analytic"}
            attempt["reference_check"] = {"passed": True, "reason": "All component densities analytic"}
            break
        refined = _Convolution(x, variance, root, df, u, 2*order, batch_size, component_batch)
        try:
            check_started = time.perf_counter()
            attempt["quadrature_check"] = _quadrature_check(kernel, refined, selected["tau"], np.asarray(selected["weights"]), quadrature_tol)
            attempt["quadrature_check_seconds"] = time.perf_counter()-check_started
            if attempt["quadrature_check"]["passed"]:
                reference_started = time.perf_counter()
                attempt["reference_check"] = _reference_check(kernel, selected["tau"], np.asarray(selected["weights"]), quadrature_tol)
                attempt["reference_check_seconds"] = time.perf_counter()-reference_started
        except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as error:
            attempt["quadrature_check"] = {"passed": False, "failure": str(error)}
        if ((attempt["quadrature_check"]["passed"] and attempt["reference_check"] and attempt["reference_check"]["passed"])
                or 4*order > maximum):
            break
        order *= 2
        # Every refinement repeats the same independently prespecified starts.
        # No held observations, labels or a favourable previous result select it.
    diagnostics["numerical_failure_count"] = sum(s["numerical_failure_count"] for a in attempts for s in a["starts"])
    diagnostics["nonconverged_start_count"] = sum(not s["optimizer_converged"] for a in attempts for s in a["starts"])
    diagnostics["quadrature_rejection_count"] = sum(bool(a["quadrature_check"] and not a["quadrature_check"]["passed"]) for a in attempts)
    diagnostics["reference_rejection_count"] = sum(bool(a["reference_check"] and not a["reference_check"]["passed"]) for a in attempts)
    diagnostics["reference_integration_failure_count"] = sum(len(a["reference_check"].get("failures", [])) for a in attempts if a["reference_check"])
    diagnostics["elapsed_seconds"] = time.perf_counter()-started
    diagnostics["final_attempt"] = len(attempts)-1
    diagnostics["quadrature_converged"] = bool(attempts[-1]["quadrature_check"] and attempts[-1]["quadrature_check"]["passed"]
                                                and attempts[-1]["reference_check"] and attempts[-1]["reference_check"]["passed"])
    converged = bool(selected and selected["optimizer_converged"] and diagnostics["quadrature_converged"])
    diagnostics["optimizer_converged"] = bool(selected and selected["optimizer_converged"])
    diagnostics["converged"] = converged
    if minimum_null_weight:
        diagnostics['minimum_null_weight']=minimum_null_weight
        diagnostics['null_floor_active']=bool(selected and selected['weights'][0]==minimum_null_weight)
        diagnostics['null_floor_is_assumed_not_estimated']=True
    assumptions = [
        "Conditionally independent training rows; shared mixture weights and scale multiplier",
        "Correct conditional residual df, shape and relative base scales, or empirical model approximation only",
        "Zero-centred Gaussian random effects independent of residual precision and base scale, within the fixed covariance library",
        "Effect/noise scale separation is not guaranteed; sufficient null/structural information is required",
        "No uncertainty in fitted nuisance inputs is propagated; training-only fitting does not prove null calibration or FDR",
        "Doubled quadrature plus every-row adaptive audit is numerical evidence; Gamma-Chernoff bounds cover truncation, not interior integration error",
    ]
    if minimum_null_weight:
        assumptions.append(f'ADDITIONAL working sparsity assumption: common zero-effect mixture weight >= {minimum_null_weight}; NOT guaranteed by the data or required PC null')
    return {"variance_multiplier": selected["tau"] if selected else None,
            "converged": converged, "objective": selected["objective"] if selected else None,
            "weights": selected["weights"] if selected else None,
            "covariances": u.tolist(), "components": metadata, "diagnostics": diagnostics,
            "assumptions": assumptions, "statistical_validation": "not_validated",
            "status": "converged" if converged else "not_converged" if selected else "all_starts_failed"}
