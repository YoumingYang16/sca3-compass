"""Advanced, reproducible evidence synthesis and longitudinal design experiments."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import norm

from .analytics import StudyEstimate


def _validate_estimates(estimates: list[StudyEstimate]) -> tuple[np.ndarray, np.ndarray]:
    if len(estimates) < 2:
        raise ValueError("At least two independent estimates are required")
    effects = np.asarray([item.effect for item in estimates], dtype=float)
    variances = np.asarray([item.variance for item in estimates], dtype=float)
    if not np.all(np.isfinite(effects)) or not np.all(np.isfinite(variances)):
        raise ValueError("Effects and variances must be finite")
    if np.any(variances <= 0):
        raise ValueError("Sampling variances must be positive")
    return effects, variances


def _posterior_grid(
    effects: np.ndarray,
    variances: np.ndarray,
    *,
    mu_prior_mean: float,
    mu_prior_sd: float,
    tau_prior_scale: float,
    grid_points: int,
) -> dict[str, np.ndarray]:
    if mu_prior_sd <= 0 or tau_prior_scale <= 0:
        raise ValueError("Prior scales must be positive")
    if grid_points < 501:
        raise ValueError("grid_points must be at least 501")
    tau_upper = max(
        tau_prior_scale * 6.0,
        float(np.ptp(effects)) * 3.0,
        float(np.sqrt(np.max(variances))) * 5.0,
        1.0,
    )
    tau = np.linspace(0.0, tau_upper, grid_points)
    augmented_variance = variances[None, :] + tau[:, None] ** 2
    precision = 1.0 / augmented_variance
    prior_precision = 1.0 / mu_prior_sd**2
    conditional_variance = 1.0 / (precision.sum(axis=1) + prior_precision)
    conditional_mean = conditional_variance * (
        (precision * effects[None, :]).sum(axis=1)
        + mu_prior_mean * prior_precision
    )

    log_likelihood_constant = -0.5 * (
        np.log(2.0 * np.pi * augmented_variance).sum(axis=1)
        + (effects[None, :] ** 2 / augmented_variance).sum(axis=1)
    )
    log_integrated_mean = 0.5 * (
        np.log(conditional_variance / mu_prior_sd**2)
        + conditional_mean**2 / conditional_variance
        - mu_prior_mean**2 / mu_prior_sd**2
    )
    log_half_normal_prior = -0.5 * (tau / tau_prior_scale) ** 2
    log_posterior = (
        log_likelihood_constant + log_integrated_mean + log_half_normal_prior
    )
    # Trapezoid weights: tau=0 is a boundary, not a full-width grid cell.
    log_posterior[[0, -1]] += np.log(0.5)
    probabilities = np.exp(log_posterior - logsumexp(log_posterior))
    return {
        "tau": tau,
        "probabilities": probabilities,
        "conditional_mean": conditional_mean,
        "conditional_sd": np.sqrt(conditional_variance),
    }


def bayesian_random_effects_meta(
    estimates: list[StudyEstimate],
    *,
    mu_prior_mean: float = 1.5,
    mu_prior_sd: float = 2.0,
    tau_prior_scale: float = 0.5,
    grid_points: int = 4001,
    posterior_draws: int = 30_000,
    seed: int = 202709,
    include_sensitivity: bool = True,
) -> dict[str, Any]:
    """Normal-normal hierarchical model fitted by deterministic quadrature.

    The finite grid integrates the heterogeneity parameter instead of relying on
    an MCMC chain. Seeded draws are used only to summarize the resulting mixture
    posterior and posterior predictive distribution.
    """
    effects, variances = _validate_estimates(estimates)
    if posterior_draws < 2_000 or posterior_draws > 500_000:
        raise ValueError("posterior_draws must be between 2,000 and 500,000")
    grid = _posterior_grid(
        effects,
        variances,
        mu_prior_mean=mu_prior_mean,
        mu_prior_sd=mu_prior_sd,
        tau_prior_scale=tau_prior_scale,
        grid_points=grid_points,
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        len(grid["tau"]), size=posterior_draws, p=grid["probabilities"]
    )
    tau_draws = grid["tau"][indices]
    mu_draws = rng.normal(
        grid["conditional_mean"][indices], grid["conditional_sd"][indices]
    )
    predictive_draws = rng.normal(mu_draws, tau_draws)

    def interval(values: np.ndarray) -> dict[str, float]:
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        return {
            "mean": float(np.mean(values)),
            "median": float(median),
            "low": float(low),
            "high": float(high),
        }

    tau_summary = interval(tau_draws)
    pooled_summary = interval(mu_draws)
    predictive_summary = interval(predictive_draws)
    result: dict[str, Any] = {
        "method": "normal-normal hierarchical meta-analysis by grid quadrature",
        "k": len(estimates),
        "posterior_draws": posterior_draws,
        "seed": seed,
        "priors": {
            "mu": f"Normal({mu_prior_mean}, {mu_prior_sd}^2)",
            "tau": f"HalfNormal({tau_prior_scale})",
        },
        "pooled_effect": pooled_summary,
        "heterogeneity_tau": tau_summary,
        "new_setting_prediction": predictive_summary,
        "probability_progression_positive": float(np.mean(mu_draws > 0)),
        "probability_progression_above_one": float(np.mean(mu_draws > 1.0)),
        "studies": [asdict(item) for item in estimates],
        "diagnostics": {
            "integration_grid_points": grid_points,
            "tau_grid_upper": float(grid["tau"][-1]),
            "posterior_boundary_mass": float(grid["probabilities"][-1]),
            "mcmc_required": False,
            "algorithm": "deterministic one-dimensional quadrature plus seeded mixture draws",
        },
        "provenance_tier": "PUBLIC_REAL_DERIVED",
        "interpretation_boundary": (
            "This publication-level posterior represents uncertainty across cohorts; "
            "it is not an individual prognosis."
        ),
    }
    if include_sensitivity:
        result["prior_sensitivity"] = []
        for scale in (0.25, 0.5, 1.0, 2.0):
            sensitivity = bayesian_random_effects_meta(
                estimates,
                mu_prior_mean=mu_prior_mean,
                mu_prior_sd=mu_prior_sd,
                tau_prior_scale=scale,
                grid_points=2001,
                posterior_draws=8_000,
                seed=seed + int(scale * 100),
                include_sensitivity=False,
            )
            result["prior_sensitivity"].append(
                {
                    "tau_prior_scale": scale,
                    "pooled_median": sensitivity["pooled_effect"]["median"],
                    "pooled_low": sensitivity["pooled_effect"]["low"],
                    "pooled_high": sensitivity["pooled_effect"]["high"],
                    "tau_median": sensitivity["heterogeneity_tau"]["median"],
                }
            )
    return result


def _hc3_regression_effect(
    outcome: np.ndarray, treatment: np.ndarray, baseline: np.ndarray, sex: np.ndarray
) -> tuple[float, float]:
    design = np.column_stack(
        [
            np.ones(len(outcome)),
            treatment,
            (baseline - np.mean(baseline)) / max(np.std(baseline), 1e-8),
            sex,
        ]
    )
    xtx_inverse = np.linalg.pinv(design.T @ design)
    beta = xtx_inverse @ design.T @ outcome
    residual = outcome - design @ beta
    leverage = np.sum((design @ xtx_inverse) * design, axis=1)
    adjusted = residual / np.clip(1.0 - leverage, 1e-6, None)
    meat = design.T @ (design * adjusted[:, None] ** 2)
    covariance = xtx_inverse @ meat @ xtx_inverse
    return float(-beta[1]), float(np.sqrt(max(covariance[1, 1], 0.0)))


def simulate_longitudinal_trial(
    *,
    annual_slope: float,
    participants_per_arm: int = 80,
    followup_months: int = 24,
    visit_interval_months: int = 6,
    treatment_reduction: float = 0.3,
    baseline_mean: float = 15.0,
    baseline_sd: float = 6.0,
    slope_sd: float = 1.5,
    measurement_sd: float = 0.8,
    measurement_ar1: float = 0.45,
    baseline_slope_correlation: float = 0.25,
    attrition_rate: float = 0.15,
    missingness: str = "MAR",
    mar_strength: float = 0.35,
    simulations: int = 1000,
    seed: int = 202709,
) -> dict[str, Any]:
    """Simulate repeated SARA measurements and analyze donor-like subject slopes.

    The data-generating model includes correlated random baseline and slope,
    AR(1) measurement error, monotone attrition, and baseline/sex adjustment.
    The estimand is the annual control-minus-treatment slope difference.
    """
    if participants_per_arm < 10 or participants_per_arm > 1000:
        raise ValueError("participants_per_arm must be between 10 and 1,000")
    if followup_months < 12 or followup_months > 60:
        raise ValueError("followup_months must be between 12 and 60")
    if visit_interval_months not in {3, 4, 6, 12}:
        raise ValueError("visit_interval_months must be 3, 4, 6, or 12")
    if followup_months % visit_interval_months:
        raise ValueError("follow-up must be divisible by visit interval")
    if followup_months // visit_interval_months < 2:
        raise ValueError("At least three scheduled visits are required")
    if missingness not in {"MCAR", "MAR"}:
        raise ValueError("missingness must be MCAR or MAR")
    if not 0 <= measurement_ar1 < 0.95:
        raise ValueError("measurement_ar1 must be in [0, 0.95)")
    if not -0.9 < baseline_slope_correlation < 0.9:
        raise ValueError("baseline_slope_correlation must be in (-0.9, 0.9)")
    if not 0 <= attrition_rate < 0.7:
        raise ValueError("attrition_rate must be in [0, 0.7)")
    if simulations < 100 or simulations > 20_000:
        raise ValueError("simulations must be between 100 and 20,000")

    rng = np.random.default_rng(seed)
    visit_months = np.arange(0, followup_months + 1, visit_interval_months)
    times = visit_months / 12.0
    number_visits = len(times)
    correlation = measurement_ar1 ** np.abs(
        np.subtract.outer(np.arange(number_visits), np.arange(number_visits))
    )
    error_covariance = measurement_sd**2 * correlation
    interval_hazard = 1.0 - (1.0 - attrition_rate) ** (1.0 / (number_visits - 1))
    base_logit = (
        np.log(interval_hazard / (1.0 - interval_hazard))
        if interval_hazard > 0 else -np.inf
    )
    target_effect = annual_slope * treatment_reduction

    estimates: list[float] = []
    standard_errors: list[float] = []
    analyzed: list[int] = []
    completers: list[int] = []
    significant = 0
    covered = 0
    for _ in range(simulations):
        total = participants_per_arm * 2
        treatment = np.repeat([0.0, 1.0], participants_per_arm)
        sex = rng.binomial(1, 0.5, total).astype(float)
        baseline_z = rng.normal(size=total)
        independent_z = rng.normal(size=total)
        slope_z = (
            baseline_slope_correlation * baseline_z
            + np.sqrt(1.0 - baseline_slope_correlation**2) * independent_z
        )
        latent_baseline = baseline_mean + baseline_sd * baseline_z
        subject_slope = (
            annual_slope * (1.0 - treatment * treatment_reduction)
            + slope_sd * slope_z
        )
        errors = rng.multivariate_normal(
            np.zeros(number_visits), error_covariance, size=total
        )
        outcomes = latent_baseline[:, None] + subject_slope[:, None] * times + errors
        observed = np.ones_like(outcomes, dtype=bool)
        for visit in range(1, number_visits):
            previous = outcomes[:, visit - 1]
            severity = (previous - baseline_mean) / max(baseline_sd, 1e-8)
            probability = expit(
                base_logit + (mar_strength * severity if missingness == "MAR" else 0.0)
            )
            newly_missing = (rng.random(total) < probability) & observed[:, visit - 1]
            observed[newly_missing, visit:] = False

        slope_estimates = np.full(total, np.nan)
        for subject in range(total):
            mask = observed[subject]
            if int(mask.sum()) < 3:
                continue
            subject_design = np.column_stack([np.ones(mask.sum()), times[mask]])
            slope_estimates[subject] = np.linalg.lstsq(
                subject_design, outcomes[subject, mask], rcond=None
            )[0][1]
        valid = np.isfinite(slope_estimates)
        if int(valid.sum()) < 10 or len(np.unique(treatment[valid])) < 2:
            continue
        effect, standard_error = _hc3_regression_effect(
            slope_estimates[valid],
            treatment[valid],
            outcomes[valid, 0],
            sex[valid],
        )
        if standard_error <= 0 or not np.isfinite(standard_error):
            continue
        low = effect - norm.ppf(0.975) * standard_error
        high = effect + norm.ppf(0.975) * standard_error
        significant += int(low > 0)
        covered += int(low <= target_effect <= high)
        estimates.append(effect)
        standard_errors.append(standard_error)
        analyzed.append(int(valid.sum()))
        completers.append(int(observed[:, -1].sum()))

    completed = len(estimates)
    if completed == 0:
        raise ValueError("No longitudinal replicate could be analyzed")
    estimates_array = np.asarray(estimates)
    return {
        "design": "two-arm longitudinal random-slope simulation",
        "analysis": "subject slopes with HC3 baseline- and sex-adjusted regression",
        "estimand": "annual control-minus-treatment SARA slope difference",
        "seed": seed,
        "requested_simulations": simulations,
        "completed_simulations": completed,
        "power": significant / completed,
        "coverage_95": covered / completed,
        "target_effect": target_effect,
        "mean_estimate": float(np.mean(estimates_array)),
        "median_estimate": float(np.median(estimates_array)),
        "bias": float(np.mean(estimates_array) - target_effect),
        "empirical_sd": float(np.std(estimates_array, ddof=1)),
        "mean_model_se": float(np.mean(standard_errors)),
        "mean_analyzed_participants": float(np.mean(analyzed)),
        "mean_final_visit_completers": float(np.mean(completers)),
        "visit_months": visit_months.tolist(),
        "assumptions": {
            "annual_control_slope": annual_slope,
            "participants_per_arm": participants_per_arm,
            "followup_months": followup_months,
            "visit_interval_months": visit_interval_months,
            "treatment_reduction": treatment_reduction,
            "baseline_mean": baseline_mean,
            "baseline_sd": baseline_sd,
            "individual_slope_sd": slope_sd,
            "measurement_sd": measurement_sd,
            "measurement_ar1": measurement_ar1,
            "baseline_slope_correlation": baseline_slope_correlation,
            "target_attrition_rate": attrition_rate,
            "missingness": missingness,
            "mar_strength": mar_strength,
        },
        "diagnostics": {
            "monotone_dropout": True,
            "minimum_visits_for_slope": 3,
            "robust_covariance": "HC3",
        },
        "provenance_tier": "SIMULATION_OUTPUT_FROM_PUBLIC_REAL_PARAMETERS",
        "interpretation_boundary": (
            "Operating characteristics depend on stated data-generating assumptions and "
            "do not demonstrate efficacy of any treatment."
        ),
    }
