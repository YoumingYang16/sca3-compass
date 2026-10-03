"""Multivariate covariance-mixture empirical Bayes, a labeled reference model.

Beta_g ~ sum pi_j N(0,U_j); Bhat_g | beta_g ~ N(beta_g,V_g).
Estimation likelihood is a composite likelihood when genes are dependent.
This is not a new mash algorithm or an assertion of equivalence to mashr.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, ndtr

from .molecular_data import PROJECT_ROOT, digest, write_json

CONFIG = PROJECT_ROOT / "configs/molecular_shrinkage_protocol.json"


def validate_effects(b: np.ndarray, v: np.ndarray) -> None:
    if b.ndim != 2 or v.shape != (len(b), b.shape[1], b.shape[1]):
        raise ValueError("Effect/covariance dimensions do not match")
    if not np.isfinite(b).all() or not np.isfinite(v).all():
        raise ValueError("Nonfinite effect or covariance")
    if not np.allclose(v, v.swapaxes(-1, -2)) or np.linalg.eigvalsh(v).min() <= 0:
        raise ValueError("Observation covariances must be symmetric positive definite")


def covariance_dictionary(b: np.ndarray, v: np.ndarray, scales: list[float], identity_only: bool = False) -> tuple[np.ndarray, list[str]]:
    dimensions = b.shape[1]
    templates = [np.eye(dimensions)]
    names = ["independent"]
    if not identity_only:
        templates.append(np.ones((dimensions, dimensions)))
        names.append("shared")
        for i in range(dimensions):
            template = np.zeros((dimensions, dimensions))
            template[i, i] = 1
            templates.append(template)
            names.append(f"region_{i}")
        # Training effects only. Test effects must never build this dictionary.
        empirical = np.cov(b.T) - v.mean(axis=0)
        eigenvalues, vectors = np.linalg.eigh(empirical)
        for i in np.argsort(eigenvalues)[-2:]:
            if eigenvalues[i] > 1e-8:
                template = np.outer(vectors[:, i], vectors[:, i])
                templates.append(template / np.max(np.diag(template)))
                names.append(f"learned_rank1_{i}")
    matrices = [np.zeros((dimensions, dimensions))]
    labels = ["point_null"]
    for template, name in zip(templates, names, strict=True):
        for scale in scales:
            matrices.append(template * scale)
            labels.append(f"{name}@{scale}")
    return np.asarray(matrices), labels


def component_likelihood(b: np.ndarray, v: np.ndarray, priors: np.ndarray) -> np.ndarray:
    validate_effects(b, v)
    if priors.ndim != 3 or priors.shape[1:] != v.shape[1:] or not np.isfinite(priors).all():
        raise ValueError("Invalid prior covariance shape")
    if not np.allclose(priors, priors.swapaxes(-1, -2)) or np.linalg.eigvalsh(priors).min() < -1e-8:
        raise ValueError("Prior covariances must be positive semidefinite")
    likelihood = []
    for prior in priors:
        cov = v + prior
        sign, logdet = np.linalg.slogdet(cov)
        if (sign <= 0).any():
            raise ValueError("Non-positive observation covariance")
        solved = np.linalg.solve(cov, b[..., None])[..., 0]
        likelihood.append(-0.5 * (b.shape[1] * np.log(2 * np.pi) + logdet + (b * solved).sum(axis=1)))
    return np.stack(likelihood, axis=1)


def fit_mixture(b: np.ndarray, v: np.ndarray, priors: np.ndarray, max_iter: int = 500, tolerance: float = 1e-7) -> dict:
    ll = component_likelihood(b, v, priors)
    weights = np.full(len(priors), 1 / len(priors))
    history = []
    converged = False
    for iteration in range(max_iter):
        joint = ll + np.log(np.maximum(weights, 1e-300))
        total = logsumexp(joint, axis=1)
        objective = float(total.sum())
        history.append(objective)
        if len(history) > 1:
            if history[-1] < history[-2] - 1e-7 * (1 + abs(history[-2])):
                raise ValueError("EM objective decreased beyond numerical tolerance")
            if abs(history[-1] - history[-2]) <= tolerance * (1 + abs(history[-2])):
                converged = True
                break
        if iteration < max_iter - 1:
            weights = np.exp(joint - total[:, None]).mean(axis=0)
            weights /= weights.sum()
    # Mixture-weight maximum likelihood is convex on the probability simplex.
    # Direct refinement avoids EM's very slow approach to boundary weights.
    a = np.exp(ll - ll.max(axis=1, keepdims=True))
    def objective(w):
        return -float(np.log(np.maximum(a @ w, 1e-300)).mean())
    def gradient(w):
        return -(a / np.maximum(a @ w, 1e-300)[:, None]).mean(axis=0)
    optimization = minimize(objective, weights, jac=gradient, method="SLSQP",
        bounds=[(0, 1)] * len(weights),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones_like(w)},
        options={"maxiter": 2000, "ftol": 1e-12})
    refined = np.maximum(optimization.x, 0)
    refined /= refined.sum()
    refined_ll = float(logsumexp(ll + np.log(np.maximum(refined, 1e-300)), axis=1).sum())
    if refined_ll >= history[-1] - 1e-7:
        weights = refined
        history.append(refined_ll)
    grad = gradient(weights)
    multiplier = float(weights @ grad)
    active = weights > 1e-6
    kkt = max(float(np.max(np.abs(grad[active] - multiplier))),
              float(np.maximum(multiplier - grad[~active], 0).max()) if (~active).any() else 0.0)
    converged = bool(optimization.success and kkt < 1e-4)
    return {"weights": weights, "objective": history, "converged": converged,
            "iterations": len(history), "optimizer": "EM + analytic-gradient simplex SLSQP",
            "simplex_kkt_residual": kkt, "optimizer_message": str(optimization.message)}


def posterior(b: np.ndarray, v: np.ndarray, priors: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    weights = np.asarray(weights)
    if len(weights) != len(priors) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Invalid mixture weights")
    joint = component_likelihood(b, v, priors) + np.log(np.maximum(weights, 1e-300))
    responsibilities = np.exp(joint - logsumexp(joint, axis=1)[:, None])
    mean = np.zeros_like(b)
    second = np.zeros_like(b)
    positive = np.zeros_like(b)
    negative = np.zeros_like(b)
    for j, prior in enumerate(priors):
        cov = v + prior
        gain = np.swapaxes(np.linalg.solve(cov, np.broadcast_to(prior, cov.shape)), -1, -2)
        component_mean = np.einsum("gij,gj->gi", gain, b)
        component_cov = prior - gain @ prior
        variance = np.maximum(np.diagonal(component_cov, axis1=-2, axis2=-1), 0)
        weight = responsibilities[:, j, None]
        mean += weight * component_mean
        second += weight * (variance + component_mean**2)
        scale = np.sqrt(np.maximum(variance, 1e-30))
        p_positive = np.where(variance > 1e-15, ndtr(component_mean / scale), (component_mean > 0).astype(float))
        p_negative = np.where(variance > 1e-15, ndtr(-component_mean / scale), (component_mean < 0).astype(float))
        positive += weight * p_positive
        negative += weight * p_negative
    return {"mean": mean, "sd": np.sqrt(np.maximum(second - mean**2, 0)),
            "lfsr": np.clip(1 - np.maximum(positive, negative), 0, 1),
            "log_predictive": logsumexp(joint, axis=1)}


def independent_simulation(config: dict) -> dict:
    def sample(seed: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        truth = np.zeros((count, 4))
        types = rng.integers(0, 4, size=count)
        shared = rng.normal(0, 0.9, size=count)
        truth[types == 1] = shared[types == 1, None]
        truth[types == 2] = rng.normal(0, 0.6, size=((types == 2).sum(), 4))
        truth[types == 3, 0] = shared[types == 3]
        se = rng.uniform(0.25, 0.75, size=(count, 4))
        correlation = 0.65 * np.eye(4) + 0.35 * np.ones((4, 4))
        cov = se[:, :, None] * correlation * se[:, None, :]
        observed = truth + np.einsum("gij,gj->gi", np.linalg.cholesky(cov), rng.normal(size=truth.shape))
        return observed, cov, truth
    train, train_v, _ = sample(config["simulation_seed"], config["simulation_training_genes"])
    test, test_v, truth = sample(config["simulation_test_seed"], config["simulation_test_genes"])
    rows = [{"method": "unshrunk", "mse": float(np.mean((test - truth)**2))}]
    for simple in (True, False):
        priors, _ = covariance_dictionary(train, train_v, config["prior_variance_scales"], simple)
        fitted = fit_mixture(train, train_v, priors, config["max_em_iterations"], config["tolerance"])
        evaluated = posterior(test, test_v, priors, fitted["weights"])
        rows.append({"method": "identity_mixture" if simple else "structured_mixture",
                     "mse": float(np.mean((evaluated["mean"] - truth)**2)),
                     "mean_test_log_predictive": float(evaluated["log_predictive"].mean()),
                     "training_converged": fitted["converged"], "iterations": fitted["iterations"]})
    return {"provenance": "SIMULATION_NOT_PATIENT_DATA", "training_genes": len(train), "test_genes": len(test),
            "rows": rows, "independent_biological_validation": False,
            "scope": "Single seeded synthetic generator; no general superiority claim"}


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    qc_path = PROJECT_ROOT / "artifacts/molecular-qc.json"
    qc = json.loads(qc_path.read_text(encoding="utf-8"))
    if qc["acquisition_sha256"] != digest(PROJECT_ROOT / "artifacts/molecular-data-audit.json") or qc["source_sha256"] != digest(Path(__file__).with_name("molecular_qc.py")):
        raise ValueError("QC is stale; rerun acquisition/QC before fitting")
    dataset = next(d for d in qc["datasets"] if d["accession"] == config["development_accession"])
    if dataset["resolved_units"] != 14 or not dataset["status"].startswith("matrix_qc_complete"):
        raise ValueError("Source sample/animal audit does not match registered development design")
    matrix_path = PROJECT_ROOT / dataset["derived_path"]
    if digest(matrix_path) != dataset["derived_sha256"]:
        raise ValueError("Processed matrix checksum mismatch")
    with np.load(matrix_path, allow_pickle=False) as archive:
        counts, features, ids = archive["counts"], archive["features"], archive["samples"]
    metadata = {s["sample_id"]: s for s in dataset["samples"]}
    samples = [metadata[x] for x in ids]
    normalized = np.log2(counts / counts.sum(axis=0) * 1e6 + 0.5)
    regions = config["regions"]
    effects, errors, indices, residuals = [], [], [], []
    for region in regions:
        group_indices = [i for i, s in enumerate(samples) if s["tissue"] == region]
        units = [samples[i]["unit_id"] for i in group_indices]
        if len(set(units)) != len(units):
            raise ValueError("Duplicate animal within a region")
        disease = [i for i in group_indices if samples[i]["genotype"] == "SCA3 transgenic"]
        controls = [i for i in group_indices if samples[i]["genotype"] == "wildtype"]
        if len(disease) < 3 or len(controls) < 3 or len(disease) + len(controls) != len(group_indices):
            raise ValueError("Unexpected genotype design")
        effects.append(normalized[:, disease].mean(axis=1) - normalized[:, controls].mean(axis=1))
        errors.append(np.sqrt(normalized[:, disease].var(axis=1, ddof=1) / len(disease) + normalized[:, controls].var(axis=1, ddof=1) / len(controls)))
        residual = {}
        for group in (disease, controls):
            centered = normalized[:, group] - normalized[:, group].mean(axis=1, keepdims=True)
            for col, i in enumerate(group):
                residual[samples[i]["unit_id"]] = centered[:, col]
        residuals.append(residual)
        indices.append({"region": region, "cases": len(disease), "controls": len(controls), "units": units})
    b, se = np.stack(effects, axis=1), np.stack(errors, axis=1)
    keep = np.isfinite(b).all(axis=1) & np.isfinite(se).all(axis=1) & (se > 1e-6).all(axis=1)
    b, se, features = b[keep], se[keep], features[keep]
    correlation = np.eye(4)
    overlap = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(i, 4):
            common = sorted(set(residuals[i]) & set(residuals[j]))
            overlap[i, j] = overlap[j, i] = len(common)
            if i != j:
                x = np.stack([residuals[i][u][keep] for u in common]).ravel()
                y = np.stack([residuals[j][u][keep] for u in common]).ravel()
                value = float(np.corrcoef(x, y)[0, 1])
                correlation[i, j] = correlation[j, i] = value
    eig, vec = np.linalg.eigh(correlation)
    correlation = vec @ np.diag(np.maximum(eig, 1e-6)) @ vec.T
    correlation /= np.sqrt(np.outer(np.diag(correlation), np.diag(correlation)))
    correlation = 0.5 * correlation + 0.5 * np.eye(4)
    v = se[:, :, None] * correlation * se[:, None, :]
    priors, labels = covariance_dictionary(b, v, config["prior_variance_scales"])
    fit = fit_mixture(b, v, priors, config["max_em_iterations"], config["tolerance"])
    post = posterior(b, v, priors, fit["weights"])
    destination = PROJECT_ROOT / "data/processed/molecular/GSE107958-shrinkage.npz"
    np.savez_compressed(destination, features=features, effects=b, standard_errors=se,
                        posterior_mean=post["mean"], posterior_sd=post["sd"], lfsr=post["lfsr"],
                        priors=priors, weights=fit["weights"], noise_correlation=correlation)
    order = np.argsort(np.linalg.norm(post["mean"], axis=1))[::-1][:40]
    examples = [{"feature": str(features[i]), "effects": b[i].tolist(), "standard_errors": se[i].tolist(),
                 "posterior_mean": post["mean"][i].tolist(), "posterior_sd": post["sd"][i].tolist(),
                 "lfsr": post["lfsr"][i].tolist()} for i in order]
    result = {"created_at": datetime.now(UTC).isoformat(), "provenance": "PUBLIC_REAL_DERIVED",
              "protocol_sha256": digest(CONFIG), "source_sha256": digest(Path(__file__)),
              "input_sha256": dataset["derived_sha256"], "qc_sha256": digest(qc_path), "result_sha256": digest(destination),
              "accession": dataset["accession"], "regions": regions, "region_design": indices,
              "genes": len(b), "independent_animals": dataset["resolved_units"], "regions_are_not_independent_studies": True,
              "overlap_matrix": overlap.tolist(), "estimated_noise_correlation": correlation.tolist(),
              "mixture": [{"component": label, "weight": float(w)} for label, w in zip(labels, fit["weights"], strict=True)],
              "em": {k: value for k, value in fit.items() if k != "weights"}, "examples": examples,
              "simulation": independent_simulation(config), "novelty_status": "reference_model_not_original_method",
              "limits": ["Source units conflict (metadata says CPM; values look like counts); column-sum rescaling is exploratory", "Plug-in Gaussian empirical Bayes; noise covariance uncertainty not propagated", "Gene-dependent composite likelihood", "Gene universe already filtered upstream", "Same animals across brain regions, not independent replication", "Model-based local false-sign rates are not certified frequentist FDR", "No causal, treatment or patient prediction claim"]}
    report_path = PROJECT_ROOT / "artifacts/molecular-shrinkage.json"
    if report_path.exists():
        old = json.loads(report_path.read_text(encoding="utf-8"))
        write_json(PROJECT_ROOT / "artifacts/molecular-shrinkage-runs" / f"{digest(report_path)[:20]}.json", old)
    write_json(report_path, result)
    print(json.dumps({"genes": len(b), "em_converged": fit["converged"], "iterations": fit["iterations"], "simulation": result["simulation"]}))


if __name__ == "__main__":
    main()
