"""Publication-calibrated design assurance with independent null calibration.

This is a canonical-normal design experiment, not a fitted patient model.
Natural-history uncertainty comes from publications; treatment, variability,
dropout and information-time assumptions are explicitly investigator-specified.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.stats import norm

from .advanced_analytics import _posterior_grid, _validate_estimates
from .analytics import StudyEstimate

METHOD_VERSION = "assurance-1.1.0"


class AssuranceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sample_sizes: list[int] = Field(
        default_factory=lambda: [40, 60, 80, 100, 140, 180], min_length=1, max_length=10
    )
    durations: list[int] = Field(
        default_factory=lambda: [12, 24, 36], min_length=1, max_length=5
    )
    treatment_reduction: float = Field(0.30, ge=0, le=1)
    effect_prior_concentration: float = Field(30.0, ge=2, le=1000)
    slope_sd: float = Field(1.5, gt=0, le=10)
    measurement_sd: float = Field(0.8, ge=0, le=10)
    measurement_ar1: float = Field(0.45, ge=0, lt=0.95)
    attrition_rate: float = Field(0.15, ge=0, lt=0.5)
    information_fractions: list[float] = Field(
        default_factory=lambda: [0.5, 0.75, 1.0], min_length=1, max_length=5
    )
    simulations: int = Field(20_000, ge=2000, le=100_000)
    calibration_simulations: int = Field(100_000, ge=10_000, le=500_000)
    target_assurance: float = Field(0.80, gt=0.05, lt=1)
    seed: int = Field(202709, ge=0, le=2_147_483_647)

    @model_validator(mode="after")
    def design_constraints(self) -> AssuranceRequest:
        if any(n < 10 or n > 1000 for n in self.sample_sizes):
            raise ValueError("Each sample size must be 10–1000 per arm")
        if any(t < 12 or t > 60 or t % 6 for t in self.durations):
            raise ValueError("Durations must be multiples of 6 months from 12 to 60")
        if self.sample_sizes != sorted(
            set(self.sample_sizes)
        ) or self.durations != sorted(set(self.durations)):
            raise ValueError("Design grids must be unique and increasing")
        rates = np.asarray(self.information_fractions)
        if (
            not np.all(np.isfinite(rates))
            or rates[0] <= 0
            or not np.all(np.diff(rates) > 0)
            or rates[-1] != 1
        ):
            raise ValueError(
                "Information fractions must increase strictly from >0 to exactly 1"
            )
        if np.min(np.diff(np.r_[0, rates])) < 0.05 - 1e-12:
            raise ValueError(
                "Information increments must be at least .05 for the verified quadrature resolution"
            )
        return self


def wilson_interval(successes: int, total: int) -> dict[str, float]:
    """Binomial Monte Carlo uncertainty, NOT a clinical effect interval."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    p = successes / total
    z = float(norm.ppf(0.975))
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {
        "estimate": p,
        "low": 0.0 if successes == 0 else float(max(0, center - half)),
        "high": 1.0 if successes == total else float(min(1, center + half)),
        "mc_se": float(np.sqrt(p * (1 - p) / total)),
    }


def brownian_paths(
    rng: np.random.Generator, draws: int, fractions: np.ndarray
) -> np.ndarray:
    return np.cumsum(
        rng.normal(size=(draws, len(fractions)))
        * np.sqrt(np.diff(np.r_[0, fractions])),
        axis=1,
    )


@lru_cache(maxsize=8)
def _legendre_rule(points: int) -> tuple[np.ndarray, np.ndarray]:
    return roots_legendre(points)


def sequential_null_rejection(
    constant: float, fractions: list[float], points: int = 256
) -> float:
    """Survivor-density recursion using Gauss–Legendre quadrature.

    Integrate W(t) only below constant at each look. Omitted lower-tail mass at
    W=-9 is below 5*Phi(-9) for up to five looks ending at unit information.
    """
    nodes, raw_weights = _legendre_rule(points)
    lower = -9.0
    positions = lower + (nodes + 1) * (constant - lower) / 2
    weights = raw_weights * (constant - lower) / 2
    intervals = np.diff(np.r_[0, fractions])
    density = norm.pdf(positions, scale=np.sqrt(intervals[0]))
    differences = positions[:, None] - positions[None, :]
    for interval in intervals[1:]:
        transition = norm.pdf(differences, scale=np.sqrt(interval))
        density = transition @ (weights * density)
    return float(1 - weights @ density)


@lru_cache(maxsize=32)
def _quadrature_boundary(fractions: tuple[float, ...]) -> tuple[float, float]:
    times = list(fractions)
    constant = float(
        brentq(
            lambda value: sequential_null_rejection(value, times, 256) - 0.025,
            1.5,
            3.5,
            xtol=1e-11,
        )
    )
    discrepancy = abs(sequential_null_rejection(constant, times, 512) - 0.025)
    if discrepancy > 1e-8:
        raise ValueError(
            "Sequential-boundary quadrature failed the doubled-resolution gate"
        )
    return constant, discrepancy


def calibrate_sequential_boundary(
    fractions: list[float], draws: int, seed: int
) -> dict[str, Any]:
    """One-sided O'Brien–Fleming SHAPE; c solved by numerical integration.

    Z(t)=W(t)/sqrt(t); crossing c/sqrt(t) is equivalent to W(t)>c.
    Two Monte Carlo checks use disjoint SeedSequence streams. This is not
    a Lan–DeMets alpha-spending implementation or a clinical-grade boundary.
    """
    times = np.asarray(fractions)
    streams = np.random.SeedSequence(seed).spawn(2)
    paths = brownian_paths(np.random.default_rng(streams[0]), draws, times)
    constant, discrepancy = _quadrature_boundary(tuple(fractions))
    empirical_constant = float(np.quantile(paths.max(axis=1), 0.975, method="higher"))
    validation = brownian_paths(np.random.default_rng(streams[1]), draws, times)
    crossing = validation > constant
    null = wilson_interval(int(np.any(crossing, axis=1).sum()), draws)
    naive = wilson_interval(
        int(np.any(validation / np.sqrt(times) > norm.ppf(0.975), axis=1).sum()), draws
    )
    return {
        "method": "Gauss–Legendre calibrated O'Brien–Fleming-shaped canonical-normal boundary",
        "alpha_one_sided": 0.025,
        "constant": constant,
        "independent_mc_boundary_estimate": empirical_constant,
        "quadrature_nodes": 256,
        "doubled_resolution_alpha_discrepancy": discrepancy,
        "integrated_null_rejection": sequential_null_rejection(
            constant, fractions, 512
        ),
        "information_fractions": fractions,
        "z_boundaries": (constant / np.sqrt(times)).tolist(),
        "calibration_draws": draws,
        "independent_validation_draws": draws,
        "null_type1": null,
        "naive_repeated_testing_type1": naive,
        "cumulative_null_rejection": [
            float(np.any(crossing[:, : i + 1], axis=1).mean())
            for i in range(len(times))
        ],
        "seed": seed,
        "boundary": "Null intervals quantify Monte Carlo check error, not the quadrature boundary error. No futility, adaptation, recruitment lag or estimated-covariance uncertainty is modeled.",
    }


def subject_slope_variance(
    months: int, slope_sd: float, measurement_sd: float, ar1: float
) -> float:
    """Exact variance of a complete-case OLS slope for the declared covariance.

    Random intercept drops out because slope weights sum to zero. This does not
    use an invented empirical covariance from patient records.
    """
    times = np.arange(0, months + 1, 6) / 12
    centered = times - times.mean()
    weights = centered / (centered @ centered)
    correlation = ar1 ** np.abs(
        np.subtract.outer(np.arange(len(times)), np.arange(len(times)))
    )
    return float(slope_sd**2 + measurement_sd**2 * weights @ correlation @ weights)


def draw_progression(
    estimates: list[StudyEstimate], tau_scale: float, draws: int, seed: int
) -> tuple[np.ndarray, dict[str, float]]:
    effects, variances = _validate_estimates(estimates)
    grid = _posterior_grid(
        effects,
        variances,
        mu_prior_mean=1.5,
        mu_prior_sd=2,
        tau_prior_scale=tau_scale,
        grid_points=4001,
    )
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(grid["tau"]), size=draws, p=grid["probabilities"])
    mean = rng.normal(
        grid["conditional_mean"][indices], grid["conditional_sd"][indices]
    )
    prediction = rng.normal(mean, grid["tau"][indices])
    return prediction, {
        "mean": float(prediction.mean()),
        "low": float(np.quantile(prediction, 0.025)),
        "high": float(np.quantile(prediction, 0.975)),
        "probability_nonpositive": float(np.mean(prediction <= 0)),
        "grid_tail_mass": float(grid["probabilities"][-20:].sum()),
    }


def run_design_assurance(
    estimates: list[StudyEstimate], request: AssuranceRequest
) -> dict[str, Any]:
    streams = np.random.SeedSequence(request.seed).generate_state(5).tolist()
    calibration = calibrate_sequential_boundary(
        request.information_fractions, request.calibration_simulations, streams[0]
    )
    times = np.asarray(request.information_fractions)
    noise = brownian_paths(
        np.random.default_rng(streams[1]), request.simulations, times
    )
    rng = np.random.default_rng(streams[2])
    mean = request.treatment_reduction
    # A design prior, not an estimated drug effect. Zero/one are point masses.
    reduction = (
        rng.beta(
            mean * request.effect_prior_concentration,
            (1 - mean) * request.effect_prior_concentration,
            request.simulations,
        )
        if 0 < mean < 1
        else np.full(request.simulations, mean)
    )
    predictions = {}
    posterior_summaries = []
    for scale in (0.25, 0.5, 1.0, 2.0):
        prediction, summary = draw_progression(
            estimates, scale, request.simulations, streams[3]
        )
        predictions[scale] = prediction
        posterior_summaries.append({"tau_prior_scale": scale, **summary})
    central = predictions[0.5]
    critical = float(norm.ppf(0.975))
    cells = []
    for months in request.durations:
        for n in request.sample_sizes:
            variance = subject_slope_variance(
                months,
                request.slope_sd,
                request.measurement_sd,
                request.measurement_ar1,
            )
            se = np.sqrt(2 * variance / (n * (1 - request.attrition_rate)))
            drift = central * reduction / se
            fixed_probabilities = norm.sf(critical - drift)
            crossing = noise + drift[:, None] * times > calibration["constant"]
            success = np.any(crossing, axis=1)
            first = np.where(success, np.argmax(crossing, axis=1), len(times) - 1)
            sequential = wilson_interval(int(success.sum()), request.simulations)
            sensitivity = []
            for scale, prediction in predictions.items():
                for multiplier in (0.8, 1.0, 1.2):
                    scenario_variance = subject_slope_variance(
                        months,
                        request.slope_sd * multiplier,
                        request.measurement_sd,
                        request.measurement_ar1,
                    )
                    scenario_se = np.sqrt(
                        2 * scenario_variance / (n * (1 - request.attrition_rate))
                    )
                    scenario_probability = norm.sf(
                        critical - prediction * reduction / scenario_se
                    )
                    sensitivity.append(
                        {
                            "tau_prior_scale": scale,
                            "slope_sd_multiplier": multiplier,
                            "fixed_assurance": float(scenario_probability.mean()),
                            "mc_se": float(
                                scenario_probability.std(ddof=1)
                                / np.sqrt(request.simulations)
                            ),
                        }
                    )
            cells.append(
                {
                    "id": f"n{n}-m{months}",
                    "participants_per_arm": n,
                    "duration_months": months,
                    "maximum_participants": 2 * n,
                    "planned_participant_months": 2 * n * months,
                    "slope_difference_se": float(se),
                    "plug_in_power": float(
                        norm.sf(critical - central.mean() * mean / se)
                    ),
                    "fixed_assurance": float(fixed_probabilities.mean()),
                    "fixed_assurance_mc_se": float(
                        fixed_probabilities.std(ddof=1) / np.sqrt(request.simulations)
                    ),
                    "sequential_assurance": sequential,
                    "expected_information_fraction": float(times[first].mean()),
                    "early_efficacy_stop_probability": float(
                        np.mean(success & (first < len(times) - 1))
                    ),
                    "minimum_scenario_fixed_assurance": min(
                        item["fixed_assurance"] for item in sensitivity
                    ),
                    "sensitivity": sensitivity,
                }
            )
    # Three-objective nondomination: participants, follow-up, scenario-min assurance.
    for cell in cells:
        cell["pareto_efficient"] = not any(
            other["maximum_participants"] <= cell["maximum_participants"]
            and other["duration_months"] <= cell["duration_months"]
            and other["minimum_scenario_fixed_assurance"]
            >= cell["minimum_scenario_fixed_assurance"]
            and (
                other["maximum_participants"] < cell["maximum_participants"]
                or other["duration_months"] < cell["duration_months"]
                or other["minimum_scenario_fixed_assurance"]
                > cell["minimum_scenario_fixed_assurance"]
            )
            for other in cells
        )
    qualifying = [
        cell
        for cell in cells
        if cell["minimum_scenario_fixed_assurance"] >= request.target_assurance
    ]
    candidate = (
        min(
            qualifying,
            key=lambda c: (c["planned_participant_months"], c["maximum_participants"]),
        )
        if qualifying
        else None
    )
    return {
        "method_version": METHOD_VERSION,
        "provenance_tier": "SIMULATION_OUTPUT_WITH_PUBLIC_DERIVED_AND_ASSUMED_INPUTS",
        "request": request.model_dump(),
        "sources": [asdict(item) for item in estimates],
        "input_contract": {
            "public_derived": "Four curated, non-overlapping publication-level annual SARA slope estimates and standard errors; extraction accuracy remains a review responsibility.",
            "assumed": "Treatment reduction Beta prior, within-person slope SD, AR(1) measurement covariance, dropout information discount and information fractions.",
            "not_available": "No individual longitudinal SCA3 records; no observed treatment response or participant-level validation.",
        },
        "progression_prior_sensitivity": posterior_summaries,
        "calibration": calibration,
        "cells": cells,
        "candidate": {
            "design_id": candidate["id"],
            "rule": "Lowest planned participant-months meeting scenario-minimum fixed assurance target within this grid",
        }
        if candidate
        else None,
        "candidate_message": "No design in this grid meets the target under every specified sensitivity scenario."
        if candidate is None
        else "Scenario-conditional candidate, not a clinical protocol recommendation.",
        "assumptions": [
            "Known-variance Gaussian subject slopes; six-month scheduled visits; complete-profile variance computed analytically.",
            "Attrition discounts expected information; informative dropout is NOT corrected. Use the separate longitudinal simulator to investigate missingness.",
            "New-setting natural-history posterior retains negative draws rather than silently truncating them.",
            "Beta treatment-reduction prior is optimistic by construction: it excludes harmful effects and is user-assumed, not learned from efficacy data.",
            "Canonical independent increments at prespecified information looks; information savings are NOT equivalent to recruitment or calendar savings.",
            "Common random numbers couple scenario comparisons; intervals are marginal Monte Carlo intervals, not simultaneous uncertainty bands.",
        ],
        "interpretation_boundary": "Design assurance integrates a stated design prior. It is neither probability that a drug works nor evidence of clinical benefit. Method is not novel or externally certified.",
        "method_references": [
            {
                "title": "rpact: group sequential design boundary documentation (independent methodological reference; not the computation engine)",
                "url": "https://www.rpact.org/vignettes/planning/rpact_boundary_examples/",
            },
        ],
    }
