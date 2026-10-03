"""Post-C2 DEVELOPMENT numerical repair, not an independently validated method.

Same centered compound-symmetry Student/Gaussian likelihood, parameter bounds,
two Student starts and BIC penalty as the frozen implementation. The interior
Gaussian MLE is analytic; unsuccessful Student starts cannot displace successful
ones by roundoff. This does not certify a global Student optimum or FDR control.
The historical implementation and frozen C1/C2 sources remain unchanged.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import OptimizeResult, minimize
from scipy.special import gammaln

from .robustness_methods import RadialFit, finite_matrix


def gaussian_value_gradient(params, orth_mean, common_mean, k):
    """Mean negative log likelihood and exact gradient in (rho, log scatter)."""
    rho, ls = np.asarray(params, float)
    a, b, scale = 1-rho, 1+(k-1)*rho, np.exp(ls)
    if min(a, b, scale) <= 0:
        raise ValueError('Positive compound eigenvalues and scale required')
    q = (orth_mean/a+common_mean/b)/scale
    value = .5*(q+(k-1)*np.log(a)+np.log(b)+k*ls+k*np.log(2*np.pi))
    gradient = np.array([.5*(-(k-1)/a+(k-1)/b+
        (orth_mean/a**2-(k-1)*common_mean/b**2)/scale), .5*(k-q)])
    return float(value), gradient


def gaussian_interior_solution(orth_mean, common_mean, k):
    """Return exact interior optimum; never clip an out-of-box solution.

    Eigenvariances have unconstrained optima O/(k-1) and C. Their bijective
    transform gives scatter=(O+C)/k and rho=(C-O/(k-1))/(O+C). The result is
    the box optimum only when strictly inside the original parameter box.
    """
    o, c = float(orth_mean), float(common_mean)
    if k < 2 or not np.isfinite([o, c]).all() or min(o, c) <= 0:
        return None
    scale = (o+c)/k
    rho, ls = (c-o/(k-1))/(o+c), float(np.log(scale))
    if not (-.95/(k-1) < rho < .995 and -7 < ls < 5):
        return None
    value, grad = gaussian_value_gradient([rho, ls], o, c, k)
    return OptimizeResult(x=np.array([rho, ls]), fun=value, jac=grad,
        success=True, status=0, nit=0, nfev=1, message='Exact interior Gaussian MLE')


def _finite_success(result):
    return bool(result.success and np.isfinite(result.fun) and np.isfinite(result.x).all())


def select_student_start(fits):
    """Best finite successful start; preserve explicit failure when none succeed."""
    successful = [fit for fit in fits if _finite_success(fit)]
    finite = [fit for fit in fits if np.isfinite(fit.fun) and np.isfinite(fit.x).all()]
    if not finite:
        raise ArithmeticError('All Student calibration starts nonfinite')
    return min(successful or finite, key=lambda fit: fit.fun), bool(successful)


def fit_calibration_with_diagnostics(calibration):
    x = finite_matrix(calibration)
    n, k = x.shape
    norm2 = np.sum(x*x, axis=1)
    common = k*x.mean(axis=1)**2
    orth = norm2-common  # preserve original likelihood arithmetic
    initial_scale = max(float(np.median(norm2))/k, .01)
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(x, rowvar=False)
    if not np.isfinite(corr).all():
        raise ArithmeticError('Degenerate calibration correlation initialization')
    initial_rho = float(np.clip((corr.sum()-k)/(k*(k-1)), -.5/(k-1), .98))
    bounds = [(-.95/(k-1), .995), (-7, 5), (np.log(1.05), np.log(200))]

    def objective(params, gaussian=False):
        rho, log_scale = params[:2]
        a, b, scale = 1-rho, 1+(k-1)*rho, np.exp(log_scale)
        q = (orth/a+common/b)/scale
        logdet = (k-1)*np.log(a)+np.log(b)+k*log_scale
        if gaussian:
            return float(.5*(np.mean(q)+logdet+k*np.log(2*np.pi)))
        df = np.exp(params[2])
        return float(gammaln(df/2)-gammaln((df+k)/2)+.5*(k*np.log(df*np.pi)+logdet)
                     +(df+k)/2*np.mean(np.log1p(q/df)))

    fits = [minimize(objective, [initial_rho, np.log(initial_scale), np.log(df)],
        method='L-BFGS-B', bounds=bounds, options={'maxiter':80, 'ftol':1e-10}) for df in [5.,30.]]
    best, student_ok = select_student_start(fits)
    gaussian = gaussian_interior_solution(float(orth.mean()), float(common.mean()), k)
    analytic = gaussian is not None
    if not analytic:
        gaussian = minimize(lambda p: objective(p, True), best.x[:2], method='L-BFGS-B',
            bounds=bounds[:2], options={'maxiter':80, 'ftol':1e-10})
    gaussian_ok = _finite_success(gaussian)
    if not np.isfinite(gaussian.fun) or not np.isfinite(gaussian.x).all():
        raise ArithmeticError('Nonfinite Gaussian calibration fit')
    choose_gaussian = 2*n*gaussian.fun+2*np.log(n) <= 2*n*best.fun+3*np.log(n)
    selected = gaussian if choose_gaussian else best
    # Both likelihood candidates must be usable for their BIC comparison.
    # A failed model is NOT silently discarded in favor of an easier model.
    converged = student_ok and gaussian_ok and _finite_success(selected)
    fit = RadialFit(float(selected.x[0]), float(np.exp(selected.x[1])),
        1e8 if choose_gaussian else float(np.exp(selected.x[2])), float(selected.fun),
        bool(converged), int(selected.nit), bool(choose_gaussian))
    details = {'version':'POST_C2_NUMERICAL_REPAIR_DEV1',
        'gaussian_analytic_interior':analytic, 'gaussian_success':gaussian_ok,
        'student_any_success':student_ok, 'student_selected_index':next(i for i,f in enumerate(fits) if f is best),
        'student_starts':[{'success':_finite_success(f), 'objective':float(f.fun),
            'iterations':int(f.nit), 'message':str(f.message)} for f in fits],
        'bic_gap_gaussian_minus_student':float(2*n*(gaussian.fun-best.fun)-np.log(n)),
        'both_models_required':True, 'independent_confirmation':False}
    return fit, details


def fit_calibration(calibration):
    return fit_calibration_with_diagnostics(calibration)[0]
