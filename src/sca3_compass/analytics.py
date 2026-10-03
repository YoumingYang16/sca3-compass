"""Reproducible statistical synthesis and rare-disease trial simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


@dataclass(frozen=True)
class StudyEstimate:
    study_id: str
    cohort: str
    effect: float
    standard_error: float
    sample_size: int | None
    source_url: str

    @property
    def variance(self) -> float:
        return self.standard_error**2


def _restricted_log_likelihood(
    tau2: float, y: np.ndarray, variances: np.ndarray
) -> float:
    weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(weights * y) / np.sum(weights))
    residual = float(np.sum(weights * (y - pooled) ** 2))
    return 0.5 * (
        float(np.sum(np.log(variances + tau2)))
        + float(np.log(np.sum(weights)))
        + residual
    )


def random_effects_meta(estimates: list[StudyEstimate]) -> dict[str, Any]:
    """Intercept-only REML synthesis with heterogeneity and influence analysis."""
    if len(estimates) < 2:
        raise ValueError("At least two independent estimates are required")
    y = np.asarray([item.effect for item in estimates], dtype=float)
    se = np.asarray([item.standard_error for item in estimates], dtype=float)
    if np.any(se <= 0):
        raise ValueError("Standard errors must be positive")
    variances = se**2
    upper = max(float(np.var(y, ddof=1) * 20), 1.0)
    fit = minimize_scalar(
        _restricted_log_likelihood,
        bounds=(0.0, upper),
        args=(y, variances),
        method="bounded",
        options={"xatol": 1e-12},
    )
    tau2 = max(float(fit.x), 0.0)
    weights = 1.0 / (variances + tau2)
    pooled = float(np.sum(weights * y) / np.sum(weights))
    pooled_se = float(np.sqrt(1.0 / np.sum(weights)))
    z = float(norm.ppf(0.975))
    fixed_weights = 1.0 / variances
    fixed_mean = float(np.sum(fixed_weights * y) / np.sum(fixed_weights))
    q = float(np.sum(fixed_weights * (y - fixed_mean) ** 2))
    df = len(estimates) - 1
    i2 = max(0.0, (q - df) / q * 100.0) if q > 0 else 0.0
    prediction_se = float(np.sqrt(tau2 + pooled_se**2))
    forest = [
        {
            "study_id": item.study_id,
            "cohort": item.cohort,
            "effect": item.effect,
            "standard_error": item.standard_error,
            "ci_low": item.effect - z * item.standard_error,
            "ci_high": item.effect + z * item.standard_error,
            "weight_percent": float(weight / np.sum(weights) * 100),
            "sample_size": item.sample_size,
            "source_url": item.source_url,
        }
        for item, weight in zip(estimates, weights, strict=True)
    ]
    leave_one_out = []
    if len(estimates) > 2:
        for index, item in enumerate(estimates):
            subset = estimates[:index] + estimates[index + 1 :]
            reduced = random_effects_meta(subset)
            leave_one_out.append(
                {
                    "omitted_study_id": item.study_id,
                    "pooled_effect": reduced["pooled_effect"],
                    "ci_low": reduced["ci_low"],
                    "ci_high": reduced["ci_high"],
                }
            )
    return {
        "method": "intercept-only random-effects meta-analysis (REML)",
        "k": len(estimates),
        "pooled_effect": pooled,
        "standard_error": pooled_se,
        "ci_low": pooled - z * pooled_se,
        "ci_high": pooled + z * pooled_se,
        "prediction_low": pooled - z * prediction_se,
        "prediction_high": pooled + z * prediction_se,
        "tau_squared": tau2,
        "q": q,
        "degrees_of_freedom": df,
        "i_squared_percent": i2,
        "forest": forest,
        "leave_one_out": leave_one_out,
        "interpretation_boundary": (
            "Publication-level estimates describe cohorts, not an individual's future course."
        ),
    }


def simulate_parallel_trial(
    *,
    annual_slope: float,
    participants_per_arm: int,
    followup_months: int,
    treatment_reduction: float,
    slope_sd: float,
    measurement_sd: float,
    attrition_rate: float,
    simulations: int,
    seed: int,
) -> dict[str, Any]:
    """Simulate a two-arm change-score trial with transparent assumptions.

    The simulator is intentionally a design experiment, not an efficacy model.
    It uses a public cohort slope as an input and samples hypothetical outcomes.
    """
    if participants_per_arm < 5:
        raise ValueError("participants_per_arm must be at least 5")
    if not 6 <= followup_months <= 60:
        raise ValueError("followup_months must be between 6 and 60")
    if not 0 <= treatment_reduction <= 1:
        raise ValueError("treatment_reduction must be between 0 and 1")
    if not 0 <= attrition_rate < 0.8:
        raise ValueError("attrition_rate must be in [0, 0.8)")
    if simulations < 100 or simulations > 20_000:
        raise ValueError("simulations must be between 100 and 20,000")
    rng = np.random.default_rng(seed)
    duration = followup_months / 12.0
    significant = 0
    effects: list[float] = []
    standard_errors: list[float] = []
    analyzed_counts: list[int] = []
    for _ in range(simulations):
        control_slopes = rng.normal(annual_slope, slope_sd, participants_per_arm)
        treatment_slopes = rng.normal(
            annual_slope * (1.0 - treatment_reduction), slope_sd, participants_per_arm
        )
        control_change = control_slopes * duration + rng.normal(
            0.0, measurement_sd, participants_per_arm
        )
        treatment_change = treatment_slopes * duration + rng.normal(
            0.0, measurement_sd, participants_per_arm
        )
        keep_control = rng.random(participants_per_arm) >= attrition_rate
        keep_treatment = rng.random(participants_per_arm) >= attrition_rate
        control = control_change[keep_control]
        treatment = treatment_change[keep_treatment]
        if len(control) < 2 or len(treatment) < 2:
            continue
        effect = float(np.mean(control) - np.mean(treatment))
        se = float(
            np.sqrt(
                np.var(control, ddof=1) / len(control)
                + np.var(treatment, ddof=1) / len(treatment)
            )
        )
        z_score = effect / se if se > 0 else 0.0
        significant += int(z_score > norm.ppf(0.975))
        effects.append(effect)
        standard_errors.append(se)
        analyzed_counts.append(len(control) + len(treatment))
    completed = len(effects)
    if completed == 0:
        raise ValueError("No simulation replicate retained enough participants")
    target_effect = annual_slope * treatment_reduction * duration
    return {
        "design": "two-arm parallel change-score simulation",
        "seed": seed,
        "requested_simulations": simulations,
        "completed_simulations": completed,
        "power": significant / completed,
        "target_mean_difference": target_effect,
        "mean_estimated_difference": float(np.mean(effects)),
        "bias": float(np.mean(effects) - target_effect),
        "mean_standard_error": float(np.mean(standard_errors)),
        "mean_analyzed_participants": float(np.mean(analyzed_counts)),
        "assumptions": {
            "annual_control_slope": annual_slope,
            "participants_per_arm": participants_per_arm,
            "followup_months": followup_months,
            "treatment_reduction": treatment_reduction,
            "individual_slope_sd": slope_sd,
            "endpoint_measurement_sd": measurement_sd,
            "independent_attrition_rate": attrition_rate,
        },
        "provenance_tier": "SIMULATION_OUTPUT_FROM_PUBLIC_REAL_PARAMETERS",
        "interpretation_boundary": (
            "This estimates operating characteristics under assumptions; it does not show that a therapy works."
        ),
    }
