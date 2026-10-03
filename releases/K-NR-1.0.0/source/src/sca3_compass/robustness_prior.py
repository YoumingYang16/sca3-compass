"""Development energy-prior fits for conditional cone candidates.

Under the working model, radial variance is IG(nu/2, nu*scatter/2) and
Q/(d*scatter) is F(d, nu). The Gaussian branch fixes radial variance at
scatter. Target residuals annihilate a common pipeline location, permitting
prior estimation without effect labels. Pooling calibration and target
energies additionally assumes their radial laws agree. Fitted shapes and
priors have no claimed exact plug-in FDR guarantee or methodological novelty.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import betaln, digamma, expit, gammaln
from scipy.stats import norm, t

from .robustness_cone import cone_partial_conjunction
from .robustness_weighting import power_radial_weight
from .robustness_simes import simes_pc


def _checked_energies(q, dimensions):
    q = np.asarray(q, dtype=float)
    d = np.asarray(dimensions, dtype=float)
    if q.ndim != 1 or q.size == 0 or not np.isfinite(q).all() or np.any(q <= 0):
        raise ValueError("q must be a nonempty vector of finite positive raw energies")
    if d.ndim != 0 and d.shape != q.shape:
        raise ValueError("dimensions must be scalar or a vector matching q")
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("dimensions must be finite and positive")
    return q, np.broadcast_to(d, q.shape)


def _student_logpdf(log_q, half_d, log_scatter, log_df):
    half_df = np.exp(log_df) / 2
    log_ratio = log_q - log_df - log_scatter
    log_kernel = np.logaddexp(0, log_ratio)
    density = (
        (half_d - 1) * log_q - half_d * (log_df + log_scatter)
        - betaln(half_d, half_df) - (half_d + half_df) * log_kernel
    )
    return density, log_kernel, expit(log_ratio)


def energy_logpdf(q, dimensions, df, scatter):
    """Raw-Q log density, including the F or chi-square scale Jacobian.

    ``scatter`` is sigma squared, not the marginal Student variance;
    ``df=np.inf`` selects the Gaussian energy distribution.
    """
    q, d = _checked_energies(q, dimensions)
    if not np.isscalar(df) or np.isnan(df) or df <= 0:
        raise ValueError("df must be positive or positive infinity")
    if not np.isscalar(scatter) or not np.isfinite(scatter) or scatter <= 0:
        raise ValueError("scatter must be finite and positive")
    log_q, half_d = np.log(q), d / 2
    if np.isinf(df):
        return (
            (half_d - 1) * log_q - .5 * (q / scatter)
            - half_d * (np.log(2) + np.log(scatter)) - gammaln(half_d)
        )
    return _student_logpdf(log_q, half_d, np.log(scatter), np.log(df))[0]


def fit_energy_prior(q, dimensions):
    """MLE energy prior with Gaussian-versus-Student BIC selection.

    Student optimization uses starts nu=5,30, nu in [1.05,200], and log
    scatter in [-9,9]. The Gaussian MLE is sum(Q)/sum(d). BIC counts each
    energy as one observation, with two Student parameters and one Gaussian
    parameter. ``objective`` is the selected mean negative raw-Q log density;
    Gaussian fits return df=inf, converged=True and iterations=0.
    """
    q, d = _checked_energies(q, dimensions)
    log_q, half_d = np.log(q), d / 2

    def objective(params):
        log_scatter, log_df = params
        density, log_kernel, ratio = _student_logpdf(
            log_q, half_d, log_scatter, log_df
        )
        half_df = np.exp(log_df) / 2
        scale_score = -half_d + (half_d + half_df) * ratio
        df_score = (
            half_df * (digamma(half_d + half_df) - digamma(half_df))
            - half_d - half_df * log_kernel + (half_d + half_df) * ratio
        )
        return -float(density.mean()), -np.array([scale_score.mean(), df_score.mean()])

    initial = float(np.clip(np.median(log_q - np.log(d)), -9, 9))
    fits = [
        minimize(
            objective, [initial, np.log(start)], jac=True, method="L-BFGS-B",
            bounds=[(-9, 9), (np.log(1.05), np.log(200))],
            options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-7},
        )
        for start in (5., 30.)
    ]
    finite_fits = [fit for fit in fits if np.isfinite(fit.fun) and np.isfinite(fit.x).all()]
    if not finite_fits:
        raise FloatingPointError("Energy-prior optimization produced no finite fit")
    student = min(finite_fits, key=lambda fit: fit.fun)
    # Divide before summing to avoid overflow for large finite raw energies.
    n = q.size
    gaussian_scatter = float(np.sum(q / n) / np.sum(d / n))
    gaussian_objective = -float(energy_logpdf(q, d, np.inf, gaussian_scatter).mean())
    gaussian_bic = 2 * n * gaussian_objective + np.log(n)
    student_bic = 2 * n * student.fun + 2 * np.log(n)
    gaussian = bool(gaussian_bic <= student_bic)
    bounds=np.array([[-9.,np.log(1.05)],[9.,np.log(200.)]])
    def projected_score(params):
        gradient=objective(params)[1]
        return params-np.clip(params-gradient,bounds[0],bounds[1])
    score=float(np.max(np.abs(projected_score(student.x))))
    original_success=bool(student.success)
    recovery=None
    # Line-search status alone is neither a stationarity certificate nor a
    # failure of the statistical model. Refine ONLY an uncertified selected
    # Student solution, retain the original objective and every retry receipt.
    if not gaussian and (not original_success or score>1e-7):
        initial_objective=float(student.fun)
        retry=least_squares(projected_score,student.x,bounds=(bounds[0],bounds[1]),
            jac='3-point',max_nfev=100,ftol=1e-12,xtol=1e-12,gtol=1e-12)
        retry_objective=float(objective(retry.x)[0])
        retry_score=float(np.max(np.abs(projected_score(retry.x))))
        # The acceptance slack accounts only for last-bit objective evaluation;
        # stationarity itself retains the original 1e-7 projected-score target.
        accepted=bool(np.isfinite(retry_objective) and retry_score<=1e-7 and
            retry_objective<=initial_objective+1e-10)
        recovery={'solver':'bounded_projected_score_least_squares','optimizer_success':bool(retry.success),
            'message':str(retry.message),'evaluations':int(retry.nfev),'accepted':accepted,
            'original_objective':initial_objective,'retry_objective':retry_objective,
            'original_projected_score':score,'retry_projected_score':retry_score,
            'objective_roundoff_slack':1e-10,'score_tolerance':1e-7}
        if accepted:
            student.x=retry.x;student.fun=retry_objective
            score=retry_score
            student_bic=2*n*student.fun+2*np.log(n)
            gaussian=bool(gaussian_bic<=student_bic)
    return {
        "df": np.inf if gaussian else float(np.exp(student.x[1])),
        "scatter": gaussian_scatter if gaussian else float(np.exp(student.x[0])),
        "gaussian_bic_selected": gaussian,
        "converged": True if gaussian else score<=1e-7,
        "iterations": 0 if gaussian else int(student.nit),
        "objective": gaussian_objective if gaussian else float(student.fun),
        "optimizer_success":original_success,"selected_projected_score":None if gaussian else score,
        "score_tolerance":1e-7,"numerical_recovery":recovery,
    }


def energy_prior_candidates(z, calibration, diagnostics, scale_mode=None, *, geometries=None):
    """Return cone and fair simple G x 2 PC arrays; export fold diagnostics.

    Read only rho from diagnostics['fit'] and the two outer-training study
    matrices from diagnostics['shape'][fold]['matrix'] (matrix arrays are
    also accepted). Caller-provided shapes are frozen: no Tyler refitting,
    ridge, or normalization is performed. Each outer fold recomputes BOTH
    training and held energies using its own matrix. No held gene fits that
    fold's prior. Both prior sources are always returned, without selection.

    With four studies and six pipelines, target d=20 and calibration d=6;
    other pipeline counts use d=4*(K-1) and K. Calibration observations may
    be N x K or studies x N x K and are flattened across leading axes.
    Optional geometries[fold] is None (EXACT existing path) or a training-only
    frozen projected mean/Q/shape/dimension/variance record. Its nuisance
    provenance must be supplied by the caller; arrays alone cannot prove it.
    """
    # Local import permits integration from robustness_methods itself.
    from .robustness_methods import compound_quadratic, contrasts

    z = np.asarray(z, dtype=float)
    x = np.asarray(calibration, dtype=float)
    if (
        z.ndim != 3 or z.shape[0] < 4 or z.shape[1] != 4 or z.shape[2] < 2
        or not np.isfinite(z).all()
    ):
        raise ValueError("Finite G x 4 x K targets with G>=4 and K>=2 required")
    g, s, k = z.shape
    if x.ndim < 2 or x.shape[-1] != k or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Finite nonempty calibration with matching pipeline count required")
    x = x.reshape(-1, k)
    rho = float(diagnostics["fit"]["rho"])
    if not np.isfinite(rho) or not -1 / (k - 1) < rho < 1:
        raise ValueError("Calibration rho must define a positive definite pipeline shape")
    shape_info = diagnostics["shape"]
    if len(shape_info) != 2:
        raise ValueError("Exactly two outer-fold study shapes required")
    shapes = []
    for item in shape_info:
        shape = np.array(item["matrix"] if isinstance(item, dict) else item, dtype=float, copy=True)
        if (
            shape.shape != (s, s) or not np.isfinite(shape).all()
            or not np.allclose(shape, shape.T, rtol=1e-12, atol=1e-12)
            or np.linalg.eigvalsh(shape).min() <= 0
        ):
            raise ValueError("Finite symmetric positive definite study shapes required")
        shape.setflags(write=False)
        shapes.append(shape)

    dimension = s * (k - 1)
    projection_variance = (1 + (k - 1) * rho) / k
    calibration_q, calibration_d = _checked_energies(compound_quadratic(x, rho), k)
    means = z.mean(axis=-1)
    signed_means = np.stack((means, -means), axis=1)
    # Centering reduces cancellation in the signal-annihilating contrasts.
    residual = (z - means[..., None]) @ contrasts(k)
    original_means,original_signed_means=means,signed_means
    original_dimension,original_projection_variance=dimension,projection_variance
    if geometries is not None and len(geometries)!=2:
        raise ValueError('Exactly two optional fold geometries required')
    labels = {
        "target_only_cone_PC": "target residual prior; uniform weights",
        "target_only_weighted_cone_PC": "target residual prior; model own-Q power weights",
        "pooled_prior_cone_PC": "shared target/calibration prior; uniform weights",
        "pooled_prior_weighted_cone_PC": "shared target/calibration prior; model own-Q power weights",
    }
    for source in ('target_only', 'pooled_prior'):
        for baseline in ('bonf', 'simes', 'weighted_simes'):
            labels[f'{source}_{baseline}_PC'] = f'{source} prior; {baseline} simple comparator'
    output = {label: np.empty((g, 2)) for label in labels}
    folds = []
    for fold, shape in enumerate(shapes):
        held = np.arange(g) % 2 == fold
        geometry=None if geometries is None else geometries[fold]
        means,signed_means=original_means,original_signed_means
        dimension,projection_variance=original_dimension,original_projection_variance
        if geometry is None:
            q = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual) / (1 - rho)
        else:
            shape=np.asarray(geometry['study_shape'],float)
            means=np.asarray(geometry['means'],float)
            q=np.asarray(geometry['q'],float)
            dimension=geometry['dimension']
            projection_variance=geometry['projection_variance']
            if (means.shape!=(g,s) or not np.isfinite(means).all() or q.shape!=(g,)
                    or shape.shape!=(s,s) or not np.isfinite(shape).all()
                    or not np.allclose(shape,shape.T) or np.linalg.eigvalsh(shape).min()<=0
                    or not np.isfinite(projection_variance) or projection_variance<=0):
                raise ValueError('Invalid frozen projected geometry')
            signed_means=np.stack((means,-means),axis=1)
        q, target_d = _checked_energies(q, dimension)
        variance_factor=projection_variance
        scale_info=None
        joint_modes=('joint_pattern','anchored_joint_pattern','central_capped_joint_pattern','free_central_capped_joint_pattern')
        continuous_modes=('continuous_pattern','continuous_floor_pattern')
        if scale_mode is not None and scale_mode not in joint_modes+continuous_modes:
            from .robustness_domain import domain_variance
            variance_factor,scale_info=domain_variance(means[~held],q[~held],dimension,
                                                       projection_variance,scale_mode,shape=shape)
        priors = {
            "target_only": fit_energy_prior(q[~held], dimension),
            "pooled_prior": fit_energy_prior(
                np.concatenate((q[~held], calibration_q)),
                np.concatenate((target_d[~held], calibration_d)),
            ),
        }
        if scale_mode in continuous_modes:
            from .robustness_continuous_scale import fit_continuous_scale
            from .robustness_domain import domain_variance
            from .robustness_pattern_test import _json_value
            target_prior=priors['target_only']
            conditional_df=np.inf if target_prior['gaussian_bic_selected'] else target_prior['df']+dimension
            training_variance=(np.full((~held).sum(),projection_variance*target_prior['scatter'])
                if np.isinf(conditional_df) else projection_variance*
                (target_prior['df']*target_prior['scatter']+q[~held])/conditional_df)
            floor=.5 if scale_mode=='continuous_floor_pattern' else 0.
            continuous=fit_continuous_scale(means[~held],training_variance,shape,conditional_df,
                minimum_null_weight=floor)
            fallback=not continuous['converged']
            fallback_info=None
            if fallback:
                variance_factor,fallback_info=domain_variance(means[~held],q[~held],dimension,
                    projection_variance,'transport_fcentral',shape=shape)
            else:
                variance_factor=projection_variance*continuous['variance_multiplier']
            scale_info={'mode':scale_mode,'continuous_fit':_json_value(continuous),
                'converged':not fallback,'minimum_null_weight':floor,
                'floor_assumed_not_estimated':True,'selected_variance':variance_factor,
                'base_variance':projection_variance,'tau':continuous['variance_multiplier'],
                'fallback_to_transport_fcentral':fallback,'fallback_details':fallback_info,
                'fallback_rule':'selected continuous fit fails fixed stationarity or quadrature audits',
                'uses_held_gene':False,'uses_truth_labels':False,
                'interpretation':'training-only continuous-effect noise scale; no fitted FDR theorem'}
        if scale_mode in joint_modes:
            from .robustness_joint_scale import fit_joint_pattern_scale
            from .robustness_domain import domain_variance
            from .robustness_pattern_test import _json_value
            target_prior=priors['target_only']
            conditional_df=np.inf if target_prior['gaussian_bic_selected'] else target_prior['df']+dimension
            training_variance=(np.full((~held).sum(),projection_variance*target_prior['scatter'])
                if np.isinf(conditional_df) else projection_variance*
                (target_prior['df']*target_prior['scatter']+q[~held])/conditional_df)
            joint=fit_joint_pattern_scale(means[~held],training_variance,shape,conditional_df)
            attempts=[_json_value(joint['diagnostics'])]
            if not joint['diagnostics']['converged']:
                joint=fit_joint_pattern_scale(means[~held],training_variance,shape,conditional_df,max_iter=1000)
                attempts.append(_json_value(joint['diagnostics']))
            fallback=not joint['diagnostics']['converged']
            fallback_info=None
            anchor_info=None
            cap_info=None
            if fallback:
                variance_factor,fallback_info=domain_variance(means[~held],q[~held],dimension,
                    projection_variance,'transport_fcentral',shape=shape)
            else:
                selected_tau=joint['tau']
                if scale_mode=='anchored_joint_pattern':
                    from .robustness_scale_anchor import anchor_joint_scale
                    selected_tau,anchor_info=anchor_joint_scale(means[~held],training_variance,shape,conditional_df,joint)
                variance_factor=projection_variance*selected_tau
                if scale_mode in ('central_capped_joint_pattern','free_central_capped_joint_pattern'):
                    cap_mode=('transport_fcentral' if scale_mode=='free_central_capped_joint_pattern'
                        else 'anchored_fcentral')
                    central_cap,cap_receipt=domain_variance(means[~held],q[~held],dimension,
                        projection_variance,cap_mode,shape=shape)
                    cap_info={'joint_variance':variance_factor,'central_cap':central_cap,
                        'cap_applied':variance_factor>central_cap,'central_fit':cap_receipt,
                        'interpretation':'heuristic training-only cap to limit effect/noise confounding; NOT a confidence upper bound',
                        'rule':'minimum of converged point-joint variance and '+cap_mode+' variance'}
                    variance_factor=min(variance_factor,central_cap)
            scale_info={'mode':scale_mode,'joint_attempts':attempts,'converged':not fallback and (anchor_info is None or anchor_info['converged']),
                'selected_variance':variance_factor,'base_variance':projection_variance,
                'tau':joint['tau'],'fallback_to_transport_fcentral':fallback,'fallback_details':fallback_info,
                'fallback_rule':'after250and1000iterationfourstartjointfitsfailstationarity',
                'uses_held_gene':False,'uses_truth_labels':False,
                'interpretation':'training-only joint empirical scale; no fitted FDR theorem'}
            if anchor_info is not None:
                scale_info['anchor']=_json_value(anchor_info)
            if cap_info is not None:
                scale_info['central_cap']=_json_value(cap_info)
        info = {
            "fold": fold, "training_genes": int((~held).sum()),
            "held_genes": int(held.sum()), "calibration_observations": len(x),
            "residual_dimension": dimension, "calibration_dimension": k,
            "projection_variance": variance_factor, "study_shape": shape.tolist(),
            "custom_geometry":geometry is not None,
            "domain_scale":scale_info,
            **{name: {**prior, 'df': None if np.isinf(prior['df']) else prior['df']}
               for name, prior in priors.items()}, "power_weighting": {},
        }
        for source, prior in priors.items():
            if prior["gaussian_bic_selected"]:
                df = np.inf
                variance = np.full(held.sum(), variance_factor * prior["scatter"])
                weight = np.ones(held.sum())
                weight_info = {"gaussian_uniform": True}
            else:
                df = prior["df"] + dimension
                variance = variance_factor * (prior["df"] * prior["scatter"] + q[held]) / df
                weight, weight_info = power_radial_weight(
                    q[held], dimension, prior["df"], prior["scatter"], variance_factor
                )
            # This helper internally partitions the held array by parity.
            # Both internal partitions MUST use the current outer shape.
            pc = cone_partial_conjunction(signed_means[held], variance, [shape, shape], df)
            output[f"{source}_cone_PC"][held] = pc
            output[f"{source}_weighted_cone_PC"][held] = np.minimum(1, pc / weight[:, None])
            standardized = signed_means[held] / np.sqrt(variance[:, None, None] * np.diag(shape))
            marginal = norm.sf(standardized) if np.isinf(df) else t.sf(standardized, df)
            marginal = np.where(standardized > 0, marginal, 1.)
            ordered = np.sort(marginal, axis=-1)[..., 1:]
            output[f'{source}_bonf_PC'][held] = np.minimum(1, 3 * ordered[..., 0])
            simes_info = {}
            simes = simes_pc(marginal, shape, diagnostics=simes_info)
            info['simes'] = simes_info  # Same shape-only policy for both priors.
            output[f'{source}_simes_PC'][held] = simes
            output[f'{source}_weighted_simes_PC'][held] = np.minimum(1, simes / weight[:, None])
            info["power_weighting"][source] = weight_info
        folds.append(info)
    for label, values in output.items():
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise FloatingPointError(f"Invalid PC probabilities: {label}")
    diagnostics["energy_prior"] = {
        "folds": folds, "model_labels": labels,
        "uses_truth_labels": False, "uses_held_genes_to_fit_prior": False,
        "refits_study_shape": False, "automatic_prior_source_selection": False,
        "assumptions": (
            "Common pipeline location and separable shape; inverse-gamma radial law "
            "or Gaussian branch. Pooled prior additionally assumes shared target and "
            "calibration radial law. Fitted inference is empirical."
        ),
    }
    return output
