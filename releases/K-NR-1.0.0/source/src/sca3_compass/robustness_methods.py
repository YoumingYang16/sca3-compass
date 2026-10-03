"""Development-only conditional radial calibration. No plug-in FDR theorem.

Shared-location, separable Gaussian scale mixtures; see RESEARCH_SPEC.md.
All parameter fitting excludes effect labels. Classical Student conditioning
and Tyler shape estimation are attributed methods, not invented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import helmert
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp, ndtr
from scipy.stats import beta, f, t

from .robustness_conjunction import hunter_partial_conjunction
from .robustness_limma import squeeze_var
from .robustness_evidence import directional_likelihood_e, partial_conjunction_e
from .molecular_methods import p_to_e_mixture
from .robustness_universal import universal_pc_e
from .robustness_cone import cone_partial_conjunction
from .molecular_envelope import efilter
from .robustness_weighting import radial_weight, power_radial_weight
from .robustness_calibrators import focused_calibrator
from .robustness_adaptive_cone import adaptive_cone_pc
from .robustness_simes import simes_pc


def finite_matrix(x):
    x = np.asarray(x, float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Finite matrix with both dimensions >=2 required")
    return x


@lru_cache(maxsize=16)
def contrasts(k: int) -> np.ndarray:
    if k < 2:
        raise ValueError("Two or more pipelines required")
    return helmert(k, full=False).T


def tyler_shape(x: np.ndarray, tolerance=1e-7, max_iter=100) -> tuple[np.ndarray, dict]:
    x = finite_matrix(x)
    n,d = x.shape
    squared = np.sum(x*x,axis=1)
    if np.any(squared <= np.finfo(float).tiny) or n <= d:
        raise ValueError("Nonzero directions and n>d required for Tyler shape")
    x = x / np.sqrt(squared[:,None])
    shape = np.eye(d)
    error = np.inf
    for iteration in range(max_iter):
        inverse = np.linalg.inv(shape)
        q = np.einsum("ni,ij,nj->n",x,inverse,x)
        updated = d/n * (x.T / q) @ x
        updated *= d/np.trace(updated)
        error = float(np.linalg.norm(updated-shape,ord="fro")/np.linalg.norm(shape,ord="fro"))
        shape = updated
        if error < tolerance:
            break
    # A tiny fixed ridge protects numerical inversion, disclosed in diagnostics.
    shape = .999*shape + .001*np.eye(d)
    diagonal = np.sqrt(np.diag(shape))
    shape /= diagonal[:,None]*diagonal[None,:]
    return shape, {"converged": error < tolerance, "iterations": iteration+1, "relative_change": error,
                   "min_eigenvalue": float(np.linalg.eigvalsh(shape).min()), "ridge": .001}


@dataclass(frozen=True)
class RadialFit:
    rho: float
    scatter: float
    df: float
    objective: float
    converged: bool
    iterations: int
    gaussian_bic_selected: bool


def compound_quadratic(x: np.ndarray, rho: float) -> np.ndarray:
    k = x.shape[-1]
    mean = x.mean(axis=-1)
    return ((x-mean[...,None])**2).sum(axis=-1)/(1-rho) + k*mean**2/(1+(k-1)*rho)


def fit_calibration(calibration: np.ndarray) -> RadialFit:
    """Centered null calibration pooled only under shared pipeline/tail law.

    Student df, scale and compound shape fit jointly; Gaussian BIC comparison
    is calibration-only and reported. No oracle distribution parameters used.
    """
    x=finite_matrix(calibration)
    n,k=x.shape
    norm2=np.sum(x*x,axis=1)
    common=k*x.mean(axis=1)**2
    orth=norm2-common
    initial_scale=max(float(np.median(norm2))/k,.01)
    corr=np.corrcoef(x,rowvar=False)
    initial_rho=float(np.clip((corr.sum()-k)/(k*(k-1)), -.5/(k-1), .98))
    lower=-.95/(k-1)
    def objective(params, gaussian=False):
        rho,log_scale=params[:2]
        scale=np.exp(log_scale)
        a,b=1-rho,1+(k-1)*rho
        q=(orth/a+common/b)/scale
        logdet=(k-1)*np.log(a)+np.log(b)+k*log_scale
        if gaussian:
            return float(.5*(np.mean(q)+logdet+k*np.log(2*np.pi)))
        df=np.exp(params[2])
        return float(gammaln(df/2)-gammaln((df+k)/2)+.5*(k*np.log(df*np.pi)+logdet)
                     +(df+k)/2*np.mean(np.log1p(q/df)))
    # Finite variance is not needed for the conditional pivot. Retain a
    # finite-mean location target, but allow infinite-variance Student laws.
    bounds=[(lower,.995),(-7,5),(np.log(1.05),np.log(200))]
    fits=[]
    for initial_df in [5.,30.]:
        fit=minimize(objective,[initial_rho,np.log(initial_scale),np.log(initial_df)],
                     method="L-BFGS-B",bounds=bounds,options={"maxiter":80,"ftol":1e-10})
        fits.append(fit)
    best=min(fits,key=lambda z:z.fun)
    gaussian=minimize(lambda z:objective(z,True),best.x[:2],method="L-BFGS-B",bounds=bounds[:2],
                      options={"maxiter":80,"ftol":1e-10})
    choose_gaussian = 2*n*gaussian.fun+2*np.log(n) <= 2*n*best.fun+3*np.log(n)
    selected=gaussian if choose_gaussian else best
    return RadialFit(float(selected.x[0]),float(np.exp(selected.x[1])),
        1e8 if choose_gaussian else float(np.exp(selected.x[2])),float(selected.fun),
        bool(selected.success),int(selected.nit),bool(choose_gaussian))


def student_tail(mean: np.ndarray, variance_factor: float, fit: RadialFit, q=None, dimension=0):
    if variance_factor <= 0:
        raise ValueError("Positive projection variance needed")
    scale=fit.scatter*variance_factor
    df=fit.df
    if fit.gaussian_bic_selected:
        return np.where(mean>0,ndtr(-mean/np.sqrt(scale)),1.)
    if q is not None:
        scale=variance_factor*(fit.df*fit.scatter+np.asarray(q))/(fit.df+dimension)
        df += dimension
    # Extending the nonpositive half to p=1 preserves all small-p tests and
    # ensures monotonic conservativeness under noncentral contrast inflation.
    return np.where(mean>0,t.sf(mean/np.sqrt(scale),df),1.)


def crossfit_residual_energy(z: np.ndarray, rho: float) -> tuple[np.ndarray,list[dict]]:
    g,s,k=z.shape
    y=z@contrasts(k)
    output=np.empty(g)
    diagnostics=[]
    # Partition is fixed without outcome labels; every gene's own contrasts are
    # excluded from learning its study shape. Cross-gene independence is NOT
    # created by this split when genes are dependent.
    for fold in range(2):
        held=np.arange(g)%2==fold
        training=y[~held].transpose(0,2,1).reshape(-1,s)
        shape,diagnostic=tyler_shape(training)
        output[held]=np.einsum("gsk,st,gtk->g",y[held],np.linalg.inv(shape),y[held])/(1-rho)
        diagnostics.append({**diagnostic,"training_genes":int((~held).sum()),"held_genes":int(held.sum()),
                            "mean_off_diagonal":float((shape.sum()-s)/(s*(s-1))),"matrix":shape.tolist()})
    return output,diagnostics


@lru_cache(maxsize=32)
def angular_order(n: int, k: int, delta: float) -> tuple[int,float]:
    candidates=range(max(1,n//4),n)
    def expected_inflation(j):
        return f.ppf(j/(n+1),1,k-1)/f.ppf(beta.ppf(delta,j,n-j+1),1,k-1)
    j=min(candidates,key=expected_inflation)
    return j,float(f.ppf(beta.ppf(delta,j,n-j+1),1,k-1))


def angular_pvalues(z: np.ndarray, calibration: np.ndarray, delta=.005):
    """Exact common-location angular upper envelope, pooled centered calibration."""
    x=finite_matrix(calibration)
    n,k=x.shape
    q=k*x.mean(axis=1)**2/x.var(axis=1,ddof=1)
    j,denominator=angular_order(n,k,delta)
    upper=np.partition(q,j-1)[j-1]/denominator
    signed=np.stack((z,-z),axis=1)
    statistic=np.sqrt(k)*signed.mean(axis=-1)/z.std(axis=-1,ddof=1)[:,None,:]
    return np.where(statistic>0,t.sf(statistic/np.sqrt(upper),k-1),1),float(upper)


def radial_grid_fit(calibration: np.ndarray, rho: float, scatter: float, max_iter=400):
    """Nonparametric Gaussian radial-mixture likelihood on a prespecified grid.

    Finite grid is an approximation, not a theorem for arbitrary mixing laws.
    A tiny weight floor is explicit regularization; no observed genes are fit.
    """
    x=finite_matrix(calibration)
    k=x.shape[1]
    grid=scatter*np.exp(np.linspace(np.log(.015),np.log(300),48))
    q=compound_quadratic(x,rho)
    loglik=-.5*(q[:,None]/grid[None,:]+k*np.log(grid)[None,:])
    # The previous 150-step EM did NOT converge in the development grid.
    # Solve the convex simplex problem directly and report its KKT gap.
    likelihood=np.exp(loglik-loglik.max(axis=1,keepdims=True))
    def objective(weights):
        density=np.maximum(likelihood@weights,np.finfo(float).tiny)
        return -float(np.mean(np.log(density))),-np.mean(likelihood/density[:,None],axis=0)
    solution=minimize(objective,np.full(len(grid),1/len(grid)),jac=True,method="SLSQP",
        bounds=[(1e-12,1)]*len(grid),constraints={"type":"eq","fun":lambda w:w.sum()-1,"jac":lambda w:np.ones_like(w)},
        options={"maxiter":max_iter,"ftol":1e-11})
    weights=np.maximum(solution.x,1e-12)
    weights/=weights.sum()
    gradient=objective(weights)[1]
    gap=float(max(0,-gradient.min()-1))
    return grid,weights,{"iterations":int(solution.nit),"kkt_gap":gap,"converged":bool(solution.success and gap<2e-5),
        "optimizer_success":bool(solution.success),"message":str(solution.message),
        "grid_edge_mass":float(weights[0]+weights[-1]),"optimizer":"SLSQP convex simplex; fixed grid; no global distribution guarantee"}


def mixture_tail(mean: np.ndarray, q: np.ndarray, dimension: int, variance_factor: float, grid,weights):
    posterior=np.log(weights)[None,:]-.5*(q[:,None]/grid[None,:]+dimension*np.log(grid)[None,:])
    posterior=np.exp(posterior-logsumexp(posterior,axis=1,keepdims=True))
    posterior/=posterior.sum(axis=1,keepdims=True)
    p=ndtr(-mean[...,None]/np.sqrt(variance_factor*grid)[None,None,None,:])
    result=np.sum(p*posterior[:,None,None,:],axis=-1)
    # Convex averages can exceed 1 by a few floating-point ulps. Reject real
    # invalid values; project only the provable [0,1] range's roundoff error.
    tolerance=16*np.finfo(float).eps
    if not np.isfinite(result).all() or ((result < -tolerance)|(result>1+tolerance)).any():
        raise FloatingPointError("Mixture tail outside probability range beyond roundoff")
    return np.clip(result,0,1)


def evaluate_candidates(z: np.ndarray, calibration: np.ndarray, profile="full"):
    if profile not in ("full","core","frontier"):
        raise ValueError("Unknown evaluation profile")
    if z.ndim!=3 or not np.isfinite(z).all():
        raise ValueError("Finite genes x studies x pipelines required")
    g,s,k=z.shape
    x=finite_matrix(calibration.reshape(-1,k))
    fit=fit_calibration(x)
    signed=np.stack((z,-z),axis=1)
    mean=signed.mean(axis=-1)
    v=(1+(k-1)*fit.rho)/k
    raw=student_tail(signed,1,fit)
    p={"gaussian_fixed_old":ndtr(-signed[...,0]),
       "t_fixed_repaired":raw[...,0],
       "t_bonf_repaired":np.minimum(1,k*raw.min(axis=-1)),
       "t_mean_simple":student_tail(mean,v,fit),
       "t_pmerge_simple":np.minimum(1,2*raw.mean(axis=-1))}
    angular,upper=angular_pvalues(z,x)
    p["angular_envelope"]=angular
    local=np.sum((z-z.mean(axis=-1,keepdims=True))**2,axis=-1)/(1-fit.rho)
    p["conditional_t_local"]=student_tail(mean,v,fit,local[:,None,:],k-1)
    full,shape_info=crossfit_residual_energy(z,fit.rho)
    # Strong radial-distribution-free studentization: with KNOWN separable
    # shapes and common pipeline location, the shared positive radial factor
    # cancels exactly, leaving t_{S(K-1)}. Fitted shapes remain plug-in here.
    p["radial_studentized_simple"]=t.sf(mean/np.sqrt(v*full[:,None,None]/(s*(k-1))),s*(k-1))
    p["conditional_t_crossfit"]=student_tail(mean,v,fit,full[:,None,None],s*(k-1))
    limma_info=[]
    for robust in [False,True]:
        post,prior,info=squeeze_var(full/(s*(k-1)),df=s*(k-1),robust=robust)
        total_df=np.minimum(s*(k-1)+np.broadcast_to(prior,(g,)),g*s*(k-1))
        p["limma_robust" if robust else "limma_standard"]=t.sf(mean/np.sqrt(v*post[:,None,None]),total_df[:,None,None])
        if profile=="full":
            p[("limma_robust" if robust else "limma_standard")+"_likelihood_eBH"]=partial_conjunction_e(
                directional_likelihood_e(mean,np.sqrt(v*post[:,None,None]),total_df[:,None,None]))
        limma_info.append({key:(None if isinstance(value,float) and not np.isfinite(value) else value)
                           for key,value in info.items()})
    df=np.inf if fit.gaussian_bic_selected else fit.df+s*(k-1)
    posterior_scale=(fit.scatter if fit.gaussian_bic_selected else (fit.df*fit.scatter+full)/(fit.df+s*(k-1)))
    statistic=mean/np.sqrt(v*np.asarray(posterior_scale).reshape(-1,1,1))
    scale=np.sqrt(v*np.asarray(posterior_scale).reshape(-1,1,1))
    if profile=="full":
        p["conditional_likelihood_eBH"]=partial_conjunction_e(directional_likelihood_e(mean,scale,df))
        p["conditional_pmerge_eBH"]=partial_conjunction_e(p_to_e_mixture(p["conditional_t_crossfit"]))
    shapes=[np.asarray(info["matrix"]) for info in shape_info]
    if profile=="full":
        p["conditional_universal_eBH"]=universal_pc_e(mean,np.broadcast_to(v*posterior_scale,(g,)),shapes,df)
    p["conditional_cone_PC"]=cone_partial_conjunction(mean,np.broadcast_to(v*posterior_scale,(g,)),shapes,df)
    adaptive=adaptive_cone_pc(mean,np.broadcast_to(v*posterior_scale,(g,)),shapes,df)
    p["active_face_experimental_PC"]=adaptive
    p["gaussian_active_face_hybrid_PC"]=adaptive if fit.gaussian_bic_selected else p["conditional_cone_PC"]
    if fit.gaussian_bic_selected:
        power_weights=np.ones(g)
        weight_info={"gaussian_uniform":True}
    else:
        power_weights,weight_info=power_radial_weight(full,s*(k-1),fit.df,fit.scatter,v)
    p["power_weighted_cone_PC"]=np.minimum(1,p["conditional_cone_PC"]/power_weights[:,None])
    p["power_weighted_active_face_experimental_PC"]=np.minimum(1,adaptive/power_weights[:,None])
    from .molecular_methods import partial_conjunction
    p["power_weighted_bonf_PC"]=np.minimum(1,partial_conjunction(p["conditional_t_crossfit"],2)/power_weights[:,None])
    for a,b in [(1,2),(1,3),(2,2),(2,3),(2,4),(3,3),(3,5),(5,3)]:
        weight=(np.ones(g) if fit.gaussian_bic_selected else radial_weight(full,s*(k-1),fit.df,fit.scatter,a,b))
        p[f"weighted_cone_a{a}b{b}_PC"]=np.minimum(1,p["conditional_cone_PC"]/weight[:,None])
    for kappa in [.01,.03,.05,.1,.2,.3,.5,.75]:
        p[f"efilter_k{kappa}_assumption_reference_decision"]=np.stack([
            efilter(p["conditional_t_crossfit"][:,sign,:],2,.025,kappa) for sign in range(2)],axis=1)
    # Assumption-sensitive literature comparators, NOT automatically certified
    # by positive fitted correlations. Retain results and audit requirements.
    simes=np.empty((g,2))
    simes_info=[]
    for fold,shape in enumerate(shapes):
        held=np.arange(g)%2==fold
        decision={'fold':fold,'held_genes':int(held.sum())}
        simes[held]=simes_pc(p['conditional_t_crossfit'][held],shape,diagnostics=decision)
        simes_info.append(decision)
    p["conditional_simes_assumption_reference_PC"]=simes
    p["power_weighted_simes_assumption_reference_PC"]=np.minimum(1,simes/power_weights[:,None])
    calibration_inputs={"cone":p["conditional_cone_PC"],"weighted_cone":p["power_weighted_cone_PC"],
        "bonf":partial_conjunction(p["conditional_t_crossfit"],2),
        "limma_robust":partial_conjunction(p["limma_robust"],2),
        "mean":partial_conjunction(p["t_mean_simple"],2),"simes":simes,
        "weighted_simes":p["power_weighted_simes_assumption_reference_PC"],
        "active_face_experimental":adaptive,
        "weighted_active_face_experimental":p["power_weighted_active_face_experimental_PC"],
        "gaussian_active_hybrid":p["gaussian_active_face_hybrid_PC"]}
    for label,pc in calibration_inputs.items():
        for threshold,fraction in [(.0003,.5),(.001,.5),(.001,.8),(.003,.5)]:
            p[f"focused_{label}_t{threshold}_f{fraction}_eBH"]=focused_calibrator(pc,2*g,threshold,fraction)
        if "active" not in label:
            for fraction in [0.,.2,.5]:
                # Blanchard-Roquain fixed reshaping measure, NOT an outcome-
                # estimated number of discoveries. Retains singleton evidence.
                cap=max(1,(2*g)//8)
                p[f"capped_{label}_f{fraction}_eBH"]=focused_calibrator(pc,2*g,.001,fraction,cap=cap)
    grid_info={"not_run":True,"reason":"core profile excludes stalled/secondary development routes"}
    if profile=="full":
        p["conditional_efilter_assumption_reference_decision"]=np.stack([
            efilter(p["conditional_t_crossfit"][:,sign,:],2,.025,.5) for sign in range(2)],axis=1)
        p["conditional_t_hunter_PC"]=hunter_partial_conjunction(statistic,shapes,df)
        grid,weights,grid_info=radial_grid_fit(x,fit.rho,fit.scatter)
        p["conditional_mixture_crossfit"]=mixture_tail(mean,full,s*(k-1),v,grid,weights)
    # Deliberately misspecified radial pooling ablation, not a protected method.
    q_identity=np.sum(local,axis=1)
    p["conditional_t_identity_ablation"]=student_tail(mean,v,fit,q_identity[:,None,None],s*(k-1))
    if profile=="frontier":
        p={name:values for name,values in p.items() if
           (not name.startswith(("weighted_cone_a","efilter_","active_face_","gaussian_active_","power_weighted_active","focused_"))
            or name.startswith("focused_") and "active" not in name and "t0.001_f0.8" in name)}
    for name,values in p.items():
        expected=(g,2) if name.endswith(("_PC","_eBH","_decision")) else (g,2,s)
        invalid_range=(values<0).any() or (not name.endswith("_eBH") and (values>1).any())
        if values.shape!=expected or not np.isfinite(values).all() or invalid_range:
            raise ValueError(f"Invalid p-values: {name}")
    return p,{"fit":fit.__dict__,"shape":shape_info,"radial_grid":grid_info,"angular_scale_upper":upper,"limma":limma_info,"power_weighting":weight_info,"simes":simes_info}
