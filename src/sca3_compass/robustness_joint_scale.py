"""Development: training-only joint MAP location-mixture and scale fitting.

Classical empirical Bayes likelihood, using the fixed signed locations from
``pattern_library``. The Student scale matrix is tau * base_variance_i * shape;
it is not its marginal covariance. No nuisance parameters or locations are
learned here except weights and tau. See docs/robustness_joint_scale.md for
assumptions, numerical stationarity diagnostics, and identifiability limits.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2, f

from .robustness_patterns import pattern_library

_PSEUDOCOUNT = .02
_CACHE_ELEMENTS = 1_048_576  # At most 8 MiB of cached component distances.


def _positive_scalar(value, name, *, infinity=False):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a positive scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive scalar") from error
    if value <= 0 or np.isnan(value) or (not infinity and not np.isfinite(value)):
        raise ValueError(f"{name} must be positive and {'not NaN' if infinity else 'finite'}")
    return value


def _inputs(means, base_variance, shape, df):
    means = np.asarray(means, dtype=float)
    variance = np.asarray(base_variance, dtype=float)
    shape = np.asarray(shape, dtype=float)
    if means.ndim != 2 or means.shape[1] != 4 or len(means) == 0:
        raise ValueError("means must be a nonempty N x 4 training array")
    if not np.isfinite(means).all():
        raise ValueError("means must be finite")
    if variance.shape != (len(means),) or not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("base_variance must be a finite positive vector with N entries")
    if (shape.shape != (4, 4) or not np.isfinite(shape).all()
            or not np.allclose(shape, shape.T, rtol=1e-12, atol=1e-12)):
        raise ValueError("shape must be a finite symmetric positive definite 4 x 4 matrix")
    try:
        root = np.linalg.cholesky(shape)
    except np.linalg.LinAlgError as error:
        raise ValueError("shape must be positive definite") from error
    return means, variance, root, _positive_scalar(df, "df", infinity=True)


def _log_distances(means, locations, log_variance):
    residual = means[:, None, :] - locations[None, :, :]
    if not np.isfinite(residual).all():
        raise FloatingPointError("Whitened residuals exceed floating-point range")
    magnitude = np.max(np.abs(residual), axis=2)
    residual /= np.where(magnitude > 0, magnitude, 1.)[:, :, None]
    with np.errstate(divide="ignore"):
        return (2 * np.log(magnitude)
                + np.log(np.einsum("ncd,ncd->nc", residual, residual))
                - log_variance[:, None])


def _kernel(log_q, log_tau, df, *, logarithmic=True):
    """Return radial negative log density and its negative log-tau derivative.

    The returned radial derivative excludes the -S/2 term: score = radial-2.
    Log distances preserve Student tails beyond representable squared distance.
    """
    if not logarithmic:
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            ratio = log_q / np.exp(log_tau)
            if np.isfinite(df):
                ratio /= df
        if np.isfinite(ratio).all():
            if np.isinf(df):
                return .5 * ratio, .5 * ratio
            coefficient = df / 2 + 2
            return coefficient * np.log1p(ratio), coefficient * (ratio / (1 + ratio))
        with np.errstate(divide="ignore"):
            log_q = np.log(log_q)
    if np.isinf(df):
        with np.errstate(over="ignore", under="ignore"):
            kernel = .5 * np.exp(log_q - log_tau)
        return kernel, kernel
    z = log_q - log_tau - np.log(df)
    coefficient = df / 2 + 2
    # Log form also preserves tiny latent precisions when df is near 1e308.
    radial = np.exp(np.log(coefficient) - np.logaddexp(0., -z))
    return coefficient * np.logaddexp(0., z), radial


class _Problem:
    def __init__(self, means, variance, root, df, batch_size):
        patterns, amplitudes = pattern_library()
        self.patterns, self.amplitudes = patterns, amplitudes
        self.means = solve_triangular(root, means.T, lower=True, check_finite=False).T
        self.locations = solve_triangular(
            root, (patterns * amplitudes[:, None]).T, lower=True, check_finite=False,
        ).T
        if not np.isfinite(self.means).all() or not np.isfinite(self.locations).all():
            raise FloatingPointError("Whitened inputs exceed floating-point range")
        self.log_variance = np.log(variance)
        self.df, self.batch_size = df, batch_size
        self.n, self.components = len(means), len(patterns)
        self.normalizing_count = self.n + self.components * _PSEUDOCOUNT
        self.constant = (-2 * np.log(2 * np.pi) - np.log(np.diag(root)).sum()
                         - 2 * self.log_variance)
        if np.isfinite(df):
            # Exact four-dimensional Gamma ratio, stable even for very large df.
            self.constant += np.logaddexp(0., np.log(2.) - np.log(df))
        self.cache = None
        self.logarithmic = True
        if self.n * self.components <= _CACHE_ELEMENTS:
            self.cache = np.empty((self.n, self.components))
            for first in range(0, self.n, batch_size):
                last = min(first + batch_size, self.n)
                self.cache[first:last] = self.distances(first, last)
            finite_logs = self.cache[np.isfinite(self.cache)]
            if finite_logs.size == 0 or (finite_logs.min() > -500 and finite_logs.max() < 500):
                np.exp(self.cache, out=self.cache)
                self.logarithmic = False
        self.evaluations = 0

    def distances(self, first, last):
        return _log_distances(self.means[first:last], self.locations, self.log_variance[first:last])

    def blocks(self):
        for first in range(0, self.n, self.batch_size):
            last = min(first + self.batch_size, self.n)
            yield first, last, (self.cache[first:last] if self.cache is not None
                                else self.distances(first, last))

    def state(self, weights, log_tau):
        self.evaluations += 1
        log_weights = np.log(weights)
        objective = _PSEUDOCOUNT * log_weights.sum()
        objective_absolute_sum = _PSEUDOCOUNT * np.abs(log_weights).sum()
        counts = np.zeros(self.components)
        radial_sum = 0.
        for first, last, log_q in self.blocks():
            kernel, radial = _kernel(log_q, log_tau, self.df, logarithmic=self.logarithmic)
            joint = log_weights - kernel
            offset = joint.max(axis=1)
            if not np.isfinite(offset).all():
                raise FloatingPointError("A training row has no representable component density")
            responsibilities = np.exp(joint - offset[:, None])
            denominator = responsibilities.sum(axis=1)
            responsibilities /= denominator[:, None]
            objective += (offset + np.log(denominator) + self.constant[first:last] - 2 * log_tau).sum()
            # Density constants can cancel log kernels almost completely.
            # Roundoff scales with the terms being summed, not their tiny sum.
            objective_absolute_sum += (np.abs(offset)+np.abs(np.log(denominator))
                +np.abs(self.constant[first:last])+2*abs(log_tau)).sum()
            counts += responsibilities.sum(axis=0)
            # An impossible Gaussian component has zero responsibility and may
            # have an infinite radial score. It contributes exactly zero.
            contribution = np.zeros_like(radial)
            np.multiply(responsibilities, radial, out=contribution, where=responsibilities > 0)
            radial_sum += contribution.sum()
        if not np.isfinite(objective) or not np.isfinite(radial_sum):
            raise FloatingPointError("Training objective or scale derivative exceeds floating-point range")
        gradient = (counts + _PSEUDOCOUNT) / weights
        return {
            "objective": float(objective), "counts": counts,
            "objective_absolute_sum":float(objective_absolute_sum),
            "radial_sum": float(radial_sum), "log_tau_gradient": float(radial_sum - 2 * self.n),
            "weight_kkt_residual": float(np.max(np.abs(gradient / self.normalizing_count - 1))),
            "conditional_weight_gap": max(0., float(gradient.max() - self.normalizing_count)),
        }


def joint_pattern_logpdf(means, base_variance, shape, df, tau=1.):
    """Normalized N x 241 component log densities; does not fit parameters."""
    means, variance, root, df = _inputs(means, base_variance, shape, df)
    tau = _positive_scalar(tau, "tau")
    problem = _Problem(means, variance, root, df, 256)
    result = np.empty((problem.n, problem.components))
    for first, last, log_q in problem.blocks():
        kernel, _ = _kernel(log_q, np.log(tau), df, logarithmic=problem.logarithmic)
        result[first:last] = problem.constant[first:last, None] - 2 * np.log(tau) - kernel
    return result


def _certificate(state, log_tau, bounds, n, tol):
    gradient = state["log_tau_gradient"] / n
    bound = "lower" if log_tau <= bounds[0] else "upper" if log_tau >= bounds[1] else None
    residual = max(gradient, 0.) if bound == "lower" else max(-gradient, 0.) if bound == "upper" else abs(gradient)
    return {
        "weight_kkt_residual": state["weight_kkt_residual"],
        "conditional_weight_gap": state["conditional_weight_gap"],
        "log_tau_gradient": state["log_tau_gradient"],
        "log_tau_gradient_per_row": float(gradient),
        "projected_log_tau_residual": float(residual), "active_bound": bound,
        "converged": bool(max(state["weight_kkt_residual"], residual) <= tol),
    }


def _em_step(problem, weights, log_tau, state, bounds):
    updated_weights = (state["counts"] + _PSEUDOCOUNT) / problem.normalizing_count
    updated_weights /= updated_weights.sum()
    if state["radial_sum"] == 0:
        updated_log_tau = bounds[0]
    else:
        updated_log_tau = np.clip(log_tau + np.log(state["radial_sum"] / (2 * problem.n)), *bounds)
    return updated_weights, float(updated_log_tau)


def _objective_roundoff(before,after,multiplier=32):
    """Cancellation-aware numerical allowance; NOT a statistical tolerance."""
    return multiplier*np.finfo(float).eps*max(1.,before['objective_absolute_sum'],after['objective_absolute_sum'])


def _summary(weights, patterns, amplitudes):
    dominant = int(np.argmax(weights))
    entropy = float(-np.dot(weights, np.log(weights)))
    return {
        "null_mass": float(weights[0]), "non_null_mass": float(weights[1:].sum()),
        "dominant_component": dominant, "dominant_pattern": patterns[dominant].tolist(),
        "dominant_amplitude": float(amplitudes[dominant]),
        "dominant_mass": float(weights[dominant]), "entropy": entropy,
        "effective_components": float(np.exp(entropy)), "minimum_weight": float(weights.min()),
        "positive_replicated_mass": float(weights[(patterns == 1).sum(axis=1) >= 2].sum()),
        "negative_replicated_mass": float(weights[(patterns == -1).sum(axis=1) >= 2].sum()),
    }


def _run_start(problem, initial_tau, tau_bounds, max_iter, tol, accelerate):
    bounds = tuple(float(np.log(value)) for value in tau_bounds)

    def as_tau(value):
        # exp(log(endpoint)) can round just outside the caller's bounds.
        return float(np.clip(np.exp(value), *tau_bounds))

    weights = np.full(problem.components, 1 / problem.components)
    log_tau = float(np.clip(np.log(initial_tau), *bounds))
    diagnostics = {
        "initial_tau": as_tau(log_tau), "iterations": 0, "em_updates": 0,
        "accelerated_steps": 0, "rejected_acceleration_trials": 0,
        "numerical_failure_events": [], "numerical_failure_count": 0,
        "rejected_acceleration_numeric_count": 0, "fatal_solver_failure_count": 0,
        "failure": None, "objective_history": [], "tau_history": [],
    }
    initial_evaluations = problem.evaluations
    state = None
    stage, iteration = "initial", -1
    try:
        state = problem.state(weights, log_tau)
        diagnostics["objective_history"].append(state["objective"])
        diagnostics["tau_history"].append(as_tau(log_tau))
        for iteration in range(max_iter):
            if _certificate(state, log_tau, bounds, problem.n, tol)["converged"]:
                break
            old_weights, old_log_tau = weights, log_tau
            stage = "em"
            # Two monotone EM steps supply the local squared extrapolation.
            intermediate = None
            for _ in range(2 if accelerate else 1):
                new_weights, new_log_tau = _em_step(problem, weights, log_tau, state, bounds)
                new_state = problem.state(new_weights, new_log_tau)
                rounding = _objective_roundoff(state,new_state)
                if new_state["objective"] < state["objective"] - rounding:
                    raise FloatingPointError(f"EM objective decreased beyond roundoff: decrease={state['objective']-new_state['objective']}; allowance={rounding}")
                weights, log_tau, state = new_weights, new_log_tau, new_state
                diagnostics["em_updates"] += 1
                if intermediate is None:
                    intermediate = np.r_[weights, log_tau / np.sqrt(problem.components)]
            if accelerate:
                origin = np.r_[old_weights, old_log_tau / np.sqrt(problem.components)]
                current = np.r_[weights, log_tau / np.sqrt(problem.components)]
                r = intermediate - origin
                v = current - intermediate - r
                norm_v = np.dot(v, v)
                alpha = -float(np.clip(np.sqrt(np.dot(r, r) / norm_v), 1., 100.)) if norm_v > 0 else -1.
                for _ in range(6):
                    if alpha >= -1.0001:
                        break
                    trial = origin - 2 * alpha * r + alpha * alpha * v
                    trial_weights = np.maximum(trial[:-1], _PSEUDOCOUNT / problem.normalizing_count)
                    trial_weights /= trial_weights.sum()
                    trial_log_tau = float(np.clip(trial[-1] * np.sqrt(problem.components), *bounds))
                    try:
                        trial_state = problem.state(trial_weights, trial_log_tau)
                    except FloatingPointError as error:
                        diagnostics["numerical_failure_events"].append({"iteration": iteration + 1, "stage": "acceleration", "message": str(error)})
                        diagnostics["rejected_acceleration_numeric_count"] += 1
                        trial_state = None
                    if trial_state is not None:
                        rounding = _objective_roundoff(state,trial_state,8)
                        old_cert = _certificate(state, log_tau, bounds, problem.n, tol)
                        new_cert = _certificate(trial_state, trial_log_tau, bounds, problem.n, tol)
                        improved_residual = max(new_cert["weight_kkt_residual"], new_cert["projected_log_tau_residual"]) < max(old_cert["weight_kkt_residual"], old_cert["projected_log_tau_residual"])
                        if (trial_state["objective"] > state["objective"] + rounding
                                or (trial_state["objective"] >= state["objective"] - rounding and improved_residual)):
                            weights, log_tau, state = trial_weights, trial_log_tau, trial_state
                            diagnostics["accelerated_steps"] += 1
                            break
                    diagnostics["rejected_acceleration_trials"] += 1
                    alpha = (alpha - 1) / 2
            diagnostics["iterations"] = iteration + 1
            diagnostics["objective_history"].append(state["objective"])
            diagnostics["tau_history"].append(as_tau(log_tau))
    except FloatingPointError as error:
        diagnostics["failure"] = str(error)
        diagnostics["fatal_solver_failure_count"] = 1
        diagnostics["numerical_failure_events"].append({"iteration": iteration + 1, "stage": stage, "message": str(error)})
        if state is not None and iteration >= 0:
            diagnostics["iterations"] = iteration + 1
            diagnostics["objective_history"].append(state["objective"])
            diagnostics["tau_history"].append(as_tau(log_tau))
    diagnostics["numerical_failure_count"] = len(diagnostics["numerical_failure_events"])
    diagnostics["evaluations"] = problem.evaluations - initial_evaluations
    diagnostics["tau_bounds"] = [float(value) for value in tau_bounds]
    if state is None:
        diagnostics.update(tau=None, objective=None, weights=None, weight_summary=None, converged=False, status="failed")
        return diagnostics
    diagnostics.update(_certificate(state, log_tau, bounds, problem.n, tol))
    if diagnostics["failure"] is not None:
        diagnostics["converged"] = False
    diagnostics.update(
        tau=as_tau(log_tau), objective=state["objective"], weights=weights.tolist(),
        weight_summary=_summary(weights, problem.patterns, problem.amplitudes),
        status="failed" if diagnostics["failure"] else "converged" if diagnostics["converged"] else "iteration_limit",
    )
    return diagnostics


def fit_joint_pattern_scale(
    means, base_variance, shape, df, *, tau_bounds=(1e-3, 1e3),
    max_iter=250, tol=1e-6, batch_size=256, accelerate=True,
):
    """Return ``{'tau', 'weights', 'patterns', 'amplitudes', 'diagnostics'}``.

    Supply only outer-training rows and training-derived or fixed nuisance
    inputs. Four deterministic starts use central radial scale times .1, 1,
    and 10, plus tau=1; clipping/duplicate starts remain explicit. The central
    scale is the training 40th percentile divided by its central-law quantile,
    used ONLY for initialization. All rows enter every likelihood evaluation.

    The best finite objective is returned, even if unconverged; callers must
    inspect diagnostics['converged']. If all starts fail before a finite state,
    tau and weights are None. Convergence checks both simplex weight KKT and
    the bound-projected log-tau gradient. It is local numerical stationarity,
    not a global optimum claim or inferential uncertainty assessment.
    """
    means, variance, root, df = _inputs(means, base_variance, shape, df)
    tol = _positive_scalar(tol, "tol")
    try:
        lower, upper = tau_bounds
    except (TypeError, ValueError) as error:
        raise ValueError("tau_bounds must contain two finite positive increasing values") from error
    lower, upper = _positive_scalar(lower, "tau lower bound"), _positive_scalar(upper, "tau upper bound")
    if lower >= upper or np.log(lower) >= np.log(upper):
        raise ValueError("tau_bounds must be strictly increasing in log scale")
    for name, value, minimum in (("max_iter", max_iter, 0), ("batch_size", batch_size, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if not isinstance(accelerate, (bool, np.bool_)):
        raise ValueError("accelerate must be boolean")  # noqa: TRY004 - unified validation API
    order = np.lexsort((variance, means[:, 3], means[:, 2], means[:, 1], means[:, 0]))
    problem = _Problem(means[order], variance[order], root, df, batch_size)
    bounds = (float(np.log(lower)), float(np.log(upper)))
    # A single N-vector suffices even when component distances are not cached.
    log_radial = _log_distances(problem.means, np.zeros((1, 4)), problem.log_variance)[:, 0]
    quantile = chi2.ppf(.4, 4) if np.isinf(df) else 4 * f.ppf(.4, 4, df)
    central_failure = None
    if np.isfinite(quantile) and quantile > 0:
        # inverted_cdf avoids interpolation involving -inf for exact zeros.
        log_central = float(np.quantile(log_radial, .4, method="inverted_cdf") - np.log(quantile))
        log_central = float(np.clip(log_central, *bounds))
    else:
        central_failure = "Central-law quantile is not representable; using clipped tau=1"
        log_central = float(np.clip(0., *bounds))
    starts = []
    for name, initial_log in (("central_x0.1", log_central + np.log(.1)),
                              ("central", log_central),
                              ("central_x10", log_central + np.log(10.)), ("unit", 0.)):
        initial_tau = float(np.exp(np.clip(initial_log, *bounds)))
        diagnostic = _run_start(problem, initial_tau, (lower, upper), max_iter, tol, accelerate)
        diagnostic["name"] = name
        starts.append(diagnostic)
    finite = [i for i, start in enumerate(starts) if start["objective"] is not None]
    selected = max(finite, key=lambda i: starts[i]["objective"]) if finite else None
    best = starts[selected] if selected is not None else None
    diagnostics = {
        "development_only": True, "training_rows": problem.n, "components": problem.components,
        "uses_truth_labels": False, "held_gene_access": False,
        "requires_caller_training_only_inputs": True,
        "df": df, "pseudocount": _PSEUDOCOUNT, "dirichlet_concentration": 1 + _PSEUDOCOUNT,
        "tau_bounds": [lower, upper], "tolerance": tol, "max_iter": int(max_iter),
        "batch_size": int(batch_size), "distance_cache_bytes": 0 if problem.cache is None else problem.cache.nbytes,
        "acceleration": bool(accelerate), "central_initial_tau": float(np.exp(log_central)),
        "central_initialization_failure": central_failure,
        "starts": starts, "selected_start": selected, "evaluations": problem.evaluations,
        "numerical_failure_count": sum(s["numerical_failure_count"] for s in starts),
        "rejected_acceleration_numeric_count": sum(s["rejected_acceleration_numeric_count"] for s in starts),
        "fatal_solver_failure_count": sum(s["fatal_solver_failure_count"] for s in starts),
        "converged": bool(best is not None and best["converged"]),
        "status": best["status"] if best is not None else "all_starts_failed",
        "objective": best["objective"] if best is not None else None,
        "stationarity_kind": "weight KKT and bound-projected log-tau gradient; joint objective is nonconcave",
        "objective_history_kind": "initial state and accepted outer iterates; acceleration trials omitted",
    }
    if best is not None:
        for key in ("tau", "weight_summary", "weight_kkt_residual", "conditional_weight_gap",
                    "log_tau_gradient", "log_tau_gradient_per_row", "projected_log_tau_residual", "active_bound"):
            diagnostics[key] = best[key]
    return {
        "tau": best["tau"] if best is not None else None,
        "weights": np.asarray(best["weights"]) if best is not None else None,
        "patterns": problem.patterns, "amplitudes": problem.amplitudes, "diagnostics": diagnostics,
    }
