"""Development cross-fitted positive loading / contrast geometry adaptation.

Failure addressed: unequal pipeline sensitivities leak genuine signal into
the old fixed 1-perpendicular residuals, causing nearly zero replication power.
The working mean is beta_gs * loading_k, loading_k>0. Other genes estimate
the loading. Nested gene-fold prediction compares it with constant loadings;
only this training evidence determines whether to retain the fixed geometry.

This uses robust PCA (Tyler shape), NNLS and noise-orthogonal projections.
It is not yet a novel method/theorem. Fitted loading, shape, and prior mean
plug-in validity are empirical; common loading structure is an assumption.

The simple comparators share the candidate's gate, positive projection,
residual energy and training-only study shape. Their additional variance
moderation is the existing classical limma comparator, not a new method.
"""
from types import SimpleNamespace

import numpy as np
from scipy.linalg import null_space
from scipy.optimize import nnls
from scipy.special import ndtr
from scipy.stats import t

from .molecular_methods import partial_conjunction
from .robustness_cone import cone_partial_conjunction
from .robustness_limma import squeeze_var
from .robustness_methods import (
    contrasts,
    crossfit_residual_energy,
    student_tail,
    tyler_shape,
)
from .robustness_weighting import power_radial_weight
from .robustness_simes import simes_pc


def pipeline_roots(rho,k):
    covariance=(1-rho)*np.eye(k)+rho*np.ones((k,k))
    eigen,vectors=np.linalg.eigh(covariance)
    if eigen.min()<=0:
        raise ValueError('Positive definite pipeline covariance required')
    return covariance,(vectors*np.sqrt(eigen))@vectors.T,(vectors/np.sqrt(eigen))@vectors.T


def learned_loading(whitened,root):
    # Tyler helper returns unit-diagonal correlation, which would REMOVE the
    # loading anisotropy. Fit the trace-normalized scatter directly instead.
    x=whitened.reshape(-1,whitened.shape[-1])
    x=x/np.linalg.norm(x,axis=1)[:,None]
    d=x.shape[1]
    scatter=np.eye(d)
    error=np.inf
    for iteration in range(100):
        q=np.einsum('ni,ij,nj->n',x,np.linalg.inv(scatter),x)
        updated=(x.T/q)@x
        updated*=d/np.trace(updated)
        error=float(np.linalg.norm(updated-scatter)/np.linalg.norm(scatter))
        scatter=updated
        if error<1e-7:
            break
    eig,vec=np.linalg.eigh(scatter)
    loading=root@vec[:,-1]
    if loading.sum()<0:
        loading=-loading
    loading=np.maximum(loading,.05*np.linalg.norm(loading))
    loading/=loading.mean()
    return loading,{'converged':error<1e-7,'iterations':iteration+1,
        'relative_change':error,'leading_eigenvalue':float(eig[-1]),
        'eigen_gap':float(eig[-1]-eig[-2]),'positive_floor_relative_norm':.05}


def residual_profile(whitened,loading,inverse_root):
    direction=inverse_root@loading
    direction/=np.linalg.norm(direction)
    projection=whitened@direction
    return np.maximum(np.sum(whitened**2,axis=(1,2))-np.sum(projection**2,axis=1),np.finfo(float).tiny)


def loading_gate(training,root,inverse_root):
    whitened=training@inverse_root
    gain=np.empty(len(training))
    fits=[]
    for fold in range(3):
        held=np.arange(len(training))%3==fold
        loading,diagnostic=learned_loading(whitened[~held],root)
        old=residual_profile(whitened[held],np.ones(training.shape[-1]),inverse_root)
        new=residual_profile(whitened[held],loading,inverse_root)
        gain[held]=np.log(old/new)
        fits.append(diagnostic)
    mean=float(gain.mean())
    se=float(gain.std(ddof=1)/np.sqrt(len(gain)))
    adopt=mean>max(.05,2.58*se) and all(d['converged'] for d in fits)
    return adopt,{'adopt':bool(adopt),'mean_log_residual_gain':mean,'gene_level_standard_error':se,
        'min_gain':.05,'standard_error_multiplier':2.58,'inner_fits':fits,
        'interpretation':'training-only prediction gate, not a significance claim or null proof'}


def positive_geometry(loading,covariance,root,inverse_root):
    loading=np.asarray(loading,float)
    if np.any(loading<=0) or not np.isfinite(loading).all():
        raise ValueError('Finite positive loadings required')
    weights,_=nnls(root,inverse_root@loading)
    weights/=weights@loading
    signal=inverse_root@loading
    noise=root@weights
    basis=null_space(np.stack((signal/np.linalg.norm(signal),noise/np.linalg.norm(noise))),rcond=1e-9)
    if basis.shape[1]<len(loading)-2:
        raise ArithmeticError('Unexpected residual dimension')
    variance=float(weights@covariance@weights)
    return weights,basis,variance


_BASELINE_SOURCES = {
    "loading_bonf_PC": "conditional_t_crossfit",
    "loading_simes_assumption_reference_PC": "conditional_simes_assumption_reference_PC",
    "loading_weighted_simes_assumption_reference_PC": "power_weighted_simes_assumption_reference_PC",
    "loading_limma_standard_PC": "limma_standard",
    "loading_limma_robust_PC": "limma_robust",
}


def _simes_pc(study_p, shape, *, diagnostics=None):
    return simes_pc(study_p, shape, diagnostics=diagnostics)


def _limma_pc(means, q_all, dimension, variance, held=None):
    """Fit each limma variant on ALL genes under one frozen geometry.

    The parity-checked squeeze_var API accepts scalar residual df only. Do
    not pretend it supports a mixed-df vector, or weaken the comparator by
    fitting each half-family alone. For an adopted fold, q_all is recomputed
    for every gene with that fold's loading, basis and study shape; only the
    held genes' p-values are returned. Its hyperprior includes own Q, exactly
    as ordinary all-gene empirical Bayes does. This is NOT a cross-fitted
    hyperprior or an independently validated unequal-df limma extension.
    """
    selector = slice(None) if held is None else held
    result, infos = {}, []
    for robust in [False, True]:
        post, prior, info = squeeze_var(q_all / dimension, df=dimension, robust=robust)
        total_df = np.minimum(dimension + np.broadcast_to(prior, q_all.shape), len(q_all) * dimension)
        study_p = t.sf(means / np.sqrt(variance * post[selector, None, None]),
                       total_df[selector, None, None])
        name = "loading_limma_robust_PC" if robust else "loading_limma_standard_PC"
        result[name] = partial_conjunction(study_p, 2)
        infos.append({key: (None if isinstance(value, float) and not np.isfinite(value) else value)
                      for key, value in info.items()})
    return result, infos


def _fixed_baselines(z, diagnostics, base_baselines):
    """Exact fixed-geometry fallback; supplied mapping is evaluate_candidates' p.

    A supplied mapping must contain all five documented source keys, with
    study arrays for Bonf/limma and PC arrays for the Simes references. It is
    never used to select a winner. Without it, reproduce the original fixed
    mean/contrast calculation, reusing recorded fold shapes when available.
    """
    g, s, k = z.shape
    if base_baselines is not None:
        result = {}
        for name, source in _BASELINE_SOURCES.items():
            if source not in base_baselines:
                raise ValueError(f"Missing fixed baseline mapping key: {source}")
            values = np.asarray(base_baselines[source], dtype=float)
            expected = (g, 2) if source.endswith("_PC") else (g, 2, s)
            if (values.shape != expected or not np.isfinite(values).all()
                    or np.any((values < 0) | (values > 1))):
                raise ValueError(f"Invalid fixed baseline mapping value: {source}")
            result[name] = np.array(values, copy=True) if source.endswith("_PC") else partial_conjunction(values, 2)
        # Old supplied Simes arrays must not silently bypass the repair.
        # Validate against the same supplied marginal noise and fold shapes;
        # no geometry or all-gene limma hyperprior is refitted here.
        if len(diagnostics.get("shape", [])) != 2:
            raise ValueError("Two fixed study shapes required to validate supplied Simes baselines")
        simes_infos = []
        for fold, entry in enumerate(diagnostics["shape"]):
            held = np.arange(g) % 2 == fold
            decision = {"fold": fold, "held_genes": int(held.sum())}
            expected = _simes_pc(np.asarray(base_baselines["conditional_t_crossfit"])[held],
                                 np.asarray(entry["matrix"]), diagnostics=decision)
            if not np.array_equal(result["loading_simes_assumption_reference_PC"][held], expected):
                raise ValueError("Supplied fixed Simes baseline does not match repaired shape policy; rerun evaluator")
            simes_infos.append(decision)
        return result, {"source": "supplied_evaluate_candidates", "limma": [], "simes": simes_infos}

    fit = SimpleNamespace(**diagnostics["fit"])
    shapes = []
    if "shape" in diagnostics:
        if len(diagnostics["shape"]) != 2:
            raise ValueError("Two fixed-geometry fold shapes required")
        y = z @ contrasts(k)
        q = np.empty(g)
        for fold, info in enumerate(diagnostics["shape"]):
            held = np.arange(g) % 2 == fold
            shape = np.asarray(info["matrix"])
            if (shape.shape != (s, s) or not np.isfinite(shape).all()
                    or not np.allclose(shape, shape.T) or np.linalg.eigvalsh(shape).min() <= 0):
                raise ValueError("Finite symmetric positive definite fixed study shape required")
            q[held] = np.einsum("gsk,st,gtk->g", y[held], np.linalg.inv(shape), y[held]) / (1-fit.rho)
            shapes.append(shape)
    else:
        q, shape_infos = crossfit_residual_energy(z, fit.rho)
        shapes = [np.asarray(info["matrix"]) for info in shape_infos]
    dimension = s * (k-1)
    variance = (1+(k-1)*fit.rho)/k
    means = np.stack((z, -z), axis=1).mean(axis=-1)
    study_p = student_tail(means, variance, fit, q[:, None, None], dimension)
    weights = (np.ones(g) if fit.gaussian_bic_selected else
               power_radial_weight(q, dimension, fit.df, fit.scatter, variance)[0])
    simes = np.empty((g, 2))
    simes_infos = []
    for fold, shape in enumerate(shapes):
        held = np.arange(g) % 2 == fold
        decision = {"fold": fold, "held_genes": int(held.sum())}
        simes[held] = _simes_pc(study_p[held], shape, diagnostics=decision)
        simes_infos.append(decision)
    result, limma_info = _limma_pc(means, q, dimension, variance)
    result.update(loading_bonf_PC=partial_conjunction(study_p, 2),
                  loading_simes_assumption_reference_PC=simes,
                  loading_weighted_simes_assumption_reference_PC=np.minimum(1, simes/weights[:, None]))
    return result, {"source": "recomputed_fixed_geometry", "limma": limma_info, "simes": simes_infos}


def _loading_baselines(z, diagnostics, adapted, base_baselines):
    g = len(z)
    if len(adapted) < 2:
        result, fallback = _fixed_baselines(z, diagnostics, base_baselines)
    else:
        result = {name: np.empty((g, 2)) for name in _BASELINE_SOURCES}
        fallback = {"source": "not_used_all_folds_adopted", "limma": []}
    fold_infos = []
    for geometry in adapted:
        held = geometry["held"]
        means, variance, df = geometry["means"], geometry["variance"], geometry["df"]
        statistic = means / np.sqrt(variance * geometry["posterior"][:, None, None])
        study_p = np.where(means > 0, ndtr(-statistic) if np.isinf(df) else t.sf(statistic, df), 1.)
        simes_info = {}
        simes = _simes_pc(study_p, geometry["shape"], diagnostics=simes_info)
        projected, limma_info = _limma_pc(means, geometry["q_all"], geometry["dimension"], variance, held)
        projected.update(loading_bonf_PC=partial_conjunction(study_p, 2),
                         loading_simes_assumption_reference_PC=simes,
                         loading_weighted_simes_assumption_reference_PC=np.minimum(1, simes/geometry["weight"][:, None]))
        for name, values in projected.items():
            result[name][held] = values
        fold_infos.append({"fold": geometry["fold"], "n_prior_genes": g,
                           "n_returned_genes": int(held.sum()), "residual_dimension": geometry["dimension"],
                           "projection_variance": variance,
                           "minimum_study_off_diagonal": float(geometry["shape"][np.triu_indices(4, 1)].min()),
                           "limma": limma_info, "simes": simes_info})
    for name, values in result.items():
        if values.shape != (g, 2) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise FloatingPointError(f"Invalid loading baseline probabilities: {name}")
    return result, {
        "fallback": fallback, "folds": fold_infos,
        "limma_fit_scope": "all genes reprojected under each adopted fold's fixed geometry; return held genes only",
        "limma_residual_df_policy": "one scalar dimension per fitted full-family geometry; no unequal-df extension",
        "limma_prior_is_crossfit": False,
        "limma_uses_own_gene_residual_in_prior": True,
        "geometry_and_gate_shared_with_candidate": True,
        "assumptions": "Simes uses the predeclared retained-shape check and Bonferroni fallback; fitted inference and learned geometry remain empirical",
    }


def loading_candidates(z,diagnostics,base_pc,base_weighted_pc,force=False,*,base_baselines=None):
    """Return the original two candidates plus five fair simple PC baselines.

    ``base_baselines`` optionally takes the unmodified p mapping returned by
    evaluate_candidates: conditional_t_crossfit, limma_standard, limma_robust,
    conditional_simes_assumption_reference_PC, and
    power_weighted_simes_assumption_reference_PC. On fallback, copy/reduce
    those corresponding baselines exactly; otherwise reconstruct the same
    fixed-geometry baselines. No outcome-dependent comparator selection.
    Supplied Simes must match the repaired rule using the supplied marginal
    probabilities and recorded fold shapes; stale arrays raise for rerunning.

    Adopted folds share exactly the candidate's geometry and residuals.
    Limma hyperpriors use all genes reprojected into each fold's geometry,
    because the existing parity-checked limma implementation requires scalar
    df. This choice and its non-cross-fitted hyperprior are diagnosed.
    """
    g,s,k=z.shape
    if s!=4 or k<3 or g<24:
        raise ValueError('At least 24 genes, four studies and >=3 pipelines required')
    fit=diagnostics['fit']
    covariance,root,inverse_root=pipeline_roots(fit['rho'],k)
    output=np.array(base_pc,copy=True)
    weighted=np.array(base_weighted_pc,copy=True)
    infos=[]
    adapted=[]
    for fold in range(2):
        held=np.arange(g)%2==fold
        training=z[~held]
        adopt,gate=loading_gate(training,root,inverse_root)
        adopt=bool(adopt or force)
        info={'fold':fold,'training_genes':int((~held).sum()),'held_genes':int(held.sum()),'gate':gate,
            'fallback_to_fixed_geometry':not bool(adopt),'forced_ablation':bool(force)}
        if adopt:
            loading,load_info=learned_loading(training@inverse_root,root)
            weights,basis,v=positive_geometry(loading,covariance,root,inverse_root)
            train_residual=(training@inverse_root)@basis
            shape,shape_info=tyler_shape(train_residual.transpose(0,2,1).reshape(-1,s))
            y=(z[held]@inverse_root)@basis
            q=np.einsum('gsk,st,gtk->g',y,np.linalg.inv(shape),y)
            d=s*basis.shape[1]
            df=np.inf if fit['gaussian_bic_selected'] else fit['df']+d
            posterior=np.full(len(q),fit['scatter']) if np.isinf(df) else (fit['df']*fit['scatter']+q)/df
            means=z[held]@weights
            means=np.stack((means,-means),1)
            value=cone_partial_conjunction(means,v*posterior,[shape,shape],df)
            output[held]=value
            weight=np.ones(len(q)) if np.isinf(df) else power_radial_weight(q,d,fit['df'],fit['scatter'],v)[0]
            weighted[held]=np.minimum(1,value/weight[:,None])
            # Preserve the candidate's held Q verbatim. The training genes
            # are reprojected with THIS same basis and shape for all-gene
            # limma moderation; never pool incompatible residual dimensions.
            q_all=np.empty(g)
            q_all[held]=q
            q_all[~held]=np.einsum('gsk,st,gtk->g',train_residual,np.linalg.inv(shape),train_residual)
            adapted.append({'fold':fold,'held':held,'means':means,'variance':v,
                'q_all':q_all,'dimension':d,'df':df,'posterior':posterior,'weight':weight,'shape':shape})
            info.update(loading=loading.tolist(),projection_weights=weights.tolist(),residual_dimension=d,
                projection_variance=v,loading_fit=load_info,study_shape=shape_info)
        infos.append(info)
    baselines,baseline_info=_loading_baselines(z,diagnostics,adapted,base_baselines)
    return {'loading_adaptive_PC':output,'loading_weighted_PC':weighted,**baselines},{'folds':infos,
        'fallback_gene_fraction':float(sum(i['held_genes'] for i in infos if i['fallback_to_fixed_geometry'])/g),
        'uses_truth_labels':False,'uses_own_gene_to_learn_loading':False,
        'simple_baselines':baseline_info,
        'assumptions':'shared positive loading, separable pipeline covariance, shared radial law; fitted inference empirical'}
