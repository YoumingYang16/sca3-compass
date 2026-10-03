"""Seeded operating-characteristic experiments, with explicit synthetic truth."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from .molecular_data import PROTOCOL, digest, write_json
from .molecular_methods import (
    GaussianMaxT,
    covariance_root,
    equicorrelation,
    evaluate_methods,
)
from .repository import PROJECT_ROOT


def truth_effects(genes: int, studies: int, effect: float) -> np.ndarray:
    """Null, single-study-only, replicated positive, negative and sign-conflict."""
    mu = np.zeros((genes, studies))
    for g in range(genes):
        kind = g % 5
        if kind == 1:
            mu[g, g % studies] = effect  # Not a replicated signal for r>=2.
        elif kind == 2:
            mu[g, :2] = effect
        elif kind == 3:
            mu[g, :] = -effect
        elif kind == 4:
            mu[g, 0], mu[g, 1] = effect, -effect  # Not directional replication.
    return mu


def generate_statistics(rng: np.random.Generator, mu: np.ndarray, pipelines: int, scenario: dict) -> np.ndarray:
    g, s = mu.shape
    root_s = covariance_root(equicorrelation(s, scenario["study_rho"]))
    root_p = covariance_root(equicorrelation(pipelines, scenario["pipeline_rho"]))
    rho_g = scenario["gene_rho"]
    independent = rng.normal(size=(g, s, pipelines))
    shared = rng.normal(size=(1, s, pipelines))
    noise = np.sqrt(1 - rho_g) * independent + np.sqrt(rho_g) * shared
    noise = np.einsum("ab,gbk,lk->gal", root_s, noise, root_p, optimize=True)
    if scenario["distribution"] == "t5":
        noise *= np.sqrt(3 / rng.chisquare(5, size=(g, 1, 1)))
    return noise + mu[..., None]


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(len(array))) if len(array) > 1 else None
    # Bounded empirical-Bernstein interval: unlike a Wald interval it does not
    # report a zero-width [0,0] bound when no false discoveries were observed.
    variance = float(array.var(ddof=1)) if len(array) > 1 else 0.0
    # Apply the one-sided bound to X and 1-X with delta=.025 each:
    # log(2/delta)=log(80); a union bound gives two-sided coverage >=.95.
    width = np.sqrt(2 * variance * np.log(80) / len(array)) + 7 * np.log(80) / (3 * (len(array) - 1)) if len(array) > 1 else 1.0
    return {"mean": mean, "mc_se": se, "ci_method": "bounded empirical Bernstein, pointwise 95%",
            "ci95": [max(0, mean - float(width)), min(1, mean + float(width))]}


def run_benchmark(protocol: dict, phase: str, repetitions: int | None = None) -> dict:
    settings = dict(protocol["simulation"])
    if repetitions is not None:
        settings["repetitions"] = repetitions
    if settings["repetitions"] < 2:
        raise ValueError("At least two repetitions required")
    seed = protocol["development_seed" if phase == "development" else "validation_seed"]
    started = time.perf_counter()
    sources = {Path(__file__).name: digest(Path(__file__)),
               "molecular_methods.py": digest(Path(__file__).with_name("molecular_methods.py"))}
    effective = {"protocol": protocol, "settings": settings, "phase": phase, "seed": seed,
                 "source_sha256": sources}
    content_hash = hashlib.sha256(json.dumps(effective, sort_keys=True).encode()).hexdigest()
    rows = []
    g, s, k = settings["genes"], settings["studies"], settings["pipelines"]
    mu = truth_effects(g, s, settings["effect_z"])
    truth = np.stack(((mu > 0).sum(axis=1) >= protocol["replication_r"],
                      (mu < 0).sum(axis=1) >= protocol["replication_r"]), axis=1)
    for index, scenario in enumerate(settings["scenarios"]):
        rng = np.random.default_rng(np.random.SeedSequence([seed, index, 0]))
        assumed = scenario.get("assumed_pipeline_rho", scenario["pipeline_rho"])
        calibration = GaussianMaxT(equicorrelation(k, assumed), settings["null_draws"], seed + index + 5000)
        records: dict[str, dict[str, list]] = {}
        scenario_start = time.perf_counter()
        for repetition in range(settings["repetitions"]):
            z = generate_statistics(rng, mu, k, scenario)
            results = evaluate_methods(z, protocol["replication_r"], protocol["alpha"], calibration)
            for name, reject in results.items():
                record = records.setdefault(name, {"fdp": [], "power": [], "discoveries": []})
                discoveries = int(reject.sum())
                record["fdp"].append(float((reject & ~truth).sum() / max(discoveries, 1)))
                record["power"].append(float((reject & truth).sum() / max(truth.sum(), 1)))
                record["discoveries"].append(discoveries)
        for name, values in records.items():
            baseline_power = np.array(records["pipeline_bonferroni_PC_BY"]["power"])
            paired = np.array(values["power"]) - baseline_power
            se = float(paired.std(ddof=1) / np.sqrt(len(paired)))
            mean_difference = float(paired.mean())
            rows.append({"scenario": scenario["name"], "method": name,
                         "within_candidate_assumptions": scenario["distribution"] == "normal" and assumed == scenario["pipeline_rho"],
                         "fdr": summarize(values["fdp"]), "power": summarize(values["power"]),
                         "mean_discoveries": float(np.mean(values["discoveries"])),
                         "paired_power_vs_bonferroni": {"mean": mean_difference, "mc_se": se,
                            "ci95": [mean_difference - 1.96 * se, mean_difference + 1.96 * se]},
                         "repetitions": settings["repetitions"],
                         "fdp_by_repetition": values["fdp"], "power_by_repetition": values["power"]})
        print(f"{phase}: {scenario['name']} finished in {time.perf_counter()-scenario_start:.1f}s", flush=True)
    return {"schema_version": "1.0", "created_at": datetime.now(UTC).isoformat(),
            "phase": phase, "provenance": "SIMULATION_NOT_PATIENT_DATA", "run_digest": content_hash,
            "protocol_sha256": digest(PROTOCOL), "effective_settings": effective,
            "source_sha256": sources,
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "elapsed_seconds": time.perf_counter() - started, "rows": rows,
            "inference_family": "2G signed hypotheses, not collapsed gene-level discoveries",
            "uncertainty": "Bounded empirical-Bernstein intervals for mean FDP/power conditional on the calibration bank; paired power differences use a normal approximation. No multiplicity adjustment across scenarios. Separate-seed validation is not external biological validation.",
            "calibration_minimum_p": 1 / (settings["null_draws"] + 1),
            "novelty_status": "not_established", "paper_ready": False,
            "missing_comparators": ["e-Filter author implementation", "PIMAX where target and assumptions match"],
            "limits": ["Known Gaussian covariance calibration only", "Misspecification scenarios intentionally outside guarantee", "No claim of new max-T/PC/BY/e-BH theorem", "Real external validation not completed"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["development", "validation"], default="development")
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if args.phase == "validation":
        development = PROJECT_ROOT / "artifacts/molecular-benchmark-development.json"
        if not development.exists():
            raise SystemExit("Validation requires a completed development run")
        prior = json.loads(development.read_text(encoding="utf-8"))
        if prior["protocol_sha256"] != digest(PROTOCOL) or any(
            digest(Path(__file__).with_name(name)) != value for name, value in prior["source_sha256"].items()
        ):
            raise SystemExit("Code or protocol changed: register and rerun development before validation")
    result = run_benchmark(protocol, args.phase, args.repetitions)
    destination = PROJECT_ROOT / "artifacts/molecular-runs" / f"{args.phase}-{result['run_digest'][:16]}.json"
    if destination.exists():
        # Keep immutable previous runs instead of overwriting timestamps/results.
        old = json.loads(destination.read_text(encoding="utf-8"))
        if old["source_sha256"] != result["source_sha256"]:
            destination = destination.with_name(destination.stem + "-" + result["source_sha256"]["molecular_methods.py"][:8] + ".json")
    if not destination.exists():
        write_json(destination, result)
    write_json(PROJECT_ROOT / f"artifacts/molecular-benchmark-{args.phase}.json", result)
    print(json.dumps({"run": str(destination), "rows": len(result["rows"]), "seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
