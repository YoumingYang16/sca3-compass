"""Bounded known-model development diagnostic; never touches the main runner.

Run from the repository root with the existing environment, Python -B and
PYTHONPATH=src. Writes exclusively to a NEW child of pc-weight-development.
Seeds, effects, shape and profiles below are fixed before inspecting output.
"""
import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import f, t

from sca3_compass.robustness_cone import cone_partial_conjunction
from sca3_compass.robustness_pc_weight import (
    DEFAULT_SEED,
    METHODS,
    WEIGHT_GRID,
    _conditional_variance,
    _PCScorer,
    _student_bank,
    build_pc_weight_tables,
)
from sca3_compass.robustness_projection import projection_pc
from sca3_compass.robustness_simes import simes_pc
from sca3_compass.robustness_weighting import power_weight_table

EVALUATION_SEED = 731906
U_SEED = 731907
DESIGN_EFFECTS = (2.5, 3.5, 4.5)
OFF_GRID_EFFECTS = (3., 4.)


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, default=plain, allow_nan=False)


def digest(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def direct_pc(means, shape, df, profiles):
    result = {
        "projection": projection_pc(means, 1., shape, df, profiles),
        "support_simes": projection_pc(means, 1., shape, df, profiles, support_simes=True),
        "profile_bonferroni": projection_pc(means, 1., shape, df, profiles, bonferroni=True),
        "cone": cone_partial_conjunction(means, np.ones(len(means)), [shape, shape], df),
    }
    z = means / np.sqrt(np.diag(shape))
    marginal = np.where(z > 0, t.sf(z, df), 1.)
    result["simes"] = simes_pc(marginal, shape)
    result["bonferroni"] = np.max([np.minimum(1., 3 * marginal[..., list(j)].min(axis=-1))
                                  for j in combinations(range(4), 3)], axis=0)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="D001-fixed-known-model")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--evaluation-samples", type=int, default=16384)
    args = parser.parse_args()
    if (not args.run_id or args.run_id in (".", "..")
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id)):
        parser.error("run-id must be one plain child directory name")
    if not 2 <= args.samples <= 65536 or not 2 <= args.evaluation_samples <= 65536:
        parser.error("sample counts must be in [2, 65536]")
    repo = Path(__file__).resolve().parents[1]
    output_root = repo / "artifacts" / "robustness" / "pc-weight-development"
    output = output_root / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    shape = .65 * np.ones((4, 4)) + .35 * np.eye(4)
    profiles = np.array([[1., 1., 0., 0.], [1., .7, 1., .3]])
    fixed = {"dimension": 20, "prior_df": 5., "prior_scale": .6, "projection_variance": .8}
    plan = {
        "purpose": "bounded known-model implementation/power diagnostic, not FDR confirmation",
        "created_utc": datetime.now(UTC).isoformat(),
        "parameters": fixed, "shape": shape, "profiles": profiles,
        "profile_convention": "each row divided by maximum; amplitude in original location units",
        "samples": args.samples, "evaluation_samples": args.evaluation_samples,
        "allocation_seed": DEFAULT_SEED, "evaluation_seed": EVALUATION_SEED, "u_seed": U_SEED,
        "design_effects": DESIGN_EFFECTS, "off_grid_effects": OFF_GRID_EFFECTS,
        "bins": 64, "weight_grid": WEIGHT_GRID, "reference_level": .0003,
        "methods": METHODS, "allocations": ("unweighted", "marginal_proxy", "height", "subbin"),
        "evaluation_U": "independent IID continuous uniform draws, not fitted bin midpoints",
        "evaluation_SE": "independent joint (noise,U) rows; amplitudes clustered within row; paired differences",
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
    }
    # Freeze the plan receipt before the first synthetic draw or table fit.
    write_json(output / "plan.json", plan)
    start = time.perf_counter()
    result = build_pc_weight_tables(**fixed, shape=shape, profiles=profiles,
                                   noise_samples=args.samples, seed=DEFAULT_SEED)
    build_seconds = time.perf_counter() - start
    baseline = power_weight_table(**fixed)
    scorer = _PCScorer(shape, result.profiles, 25.)
    evaluation_start = time.perf_counter()
    bank = _student_bank(args.evaluation_samples, 25., shape, EVALUATION_SEED)
    u = np.random.default_rng(U_SEED).uniform(size=args.evaluation_samples)
    q = 20 * .6 * f.ppf(u, 20, 5.)
    variance = _conditional_variance(q, **fixed)
    methods, modes = METHODS, plan["allocations"]
    all_effects = np.array(DESIGN_EFFECTS + OFF_GRID_EFFECTS)
    indicators = {m: np.zeros((len(all_effects), args.evaluation_samples, 2, len(modes)), dtype=bool)
                  for m in methods}
    baseline_weights = baseline[np.minimum((u * 64).astype(int), 63)]
    weights = {m: np.stack((np.ones((len(u), 2)), np.broadcast_to(baseline_weights[:, None], (len(u), 2)),
                           result.lookup(q, m), result.lookup(q, m, allocation="subbin")), axis=-1)
               for m in methods}
    for begin in range(0, len(u), 512):
        end = min(len(u), begin + 512)
        means = (bank[None, begin:end, None, :]
                 + all_effects[:, None, None, None] * result.profiles[None, None]
                 / np.sqrt(variance[None, begin:end, None, None]))
        pvalues = scorer.pvalues(means.reshape(-1, 2, 4))
        for method in methods:
            pvalue = pvalues[method].reshape(len(all_effects), end - begin, 2)
            indicators[method][:, begin:end] = pvalue[..., None] <= .0003 * weights[method][None, begin:end]
    evaluation_seconds = time.perf_counter() - evaluation_start
    power = {}
    for method in methods:
        power[method] = {}
        for effect_group, selection in (("design", slice(0, 3)), ("off_grid", slice(3, 5))):
            per_row = indicators[method][selection].mean(axis=0)
            group = {}
            for index, mode in enumerate(modes):
                values = per_row[..., index]
                delta = values - per_row[..., 1]
                group[mode] = {"power_by_sign": values.mean(axis=0),
                               "se_by_sign": values.std(axis=0, ddof=1) / np.sqrt(len(u)),
                               "delta_vs_marginal_pp": 100 * delta.mean(axis=0),
                               "paired_se_delta_pp": 100 * delta.std(axis=0, ddof=1) / np.sqrt(len(u))}
            power[method][effect_group] = group
    agreement_start = time.perf_counter()
    probe = bank[:256, None, :] + np.array([2.5, 4.5])[None, :, None] * result.profiles[None]
    actual = scorer.pvalues(probe)
    direct = direct_pc(probe, shape, 25., profiles)
    agreement = {}
    for method in methods:
        max_error = float(np.max(np.abs(actual[method] - direct[method])))
        mismatches = int(np.count_nonzero((actual[method][..., None] <= .0003 * np.array(WEIGHT_GRID))
                                          != (direct[method][..., None] <= .0003 * np.array(WEIGHT_GRID))))
        agreement[method] = {"max_absolute_p_error": max_error, "rejection_mismatches": mismatches}
        if max_error > 1e-11 or mismatches:
            raise AssertionError(f"Direct PC agreement failed: {method}")
    agreement_seconds = time.perf_counter() - agreement_start
    integrals = {m: {mode: result.model_integral(m, allocation=mode) for mode in ("height", "subbin")}
                 for m in methods}
    for row in integrals.values():
        for values in row.values():
            np.testing.assert_allclose(values, 1., rtol=0, atol=1e-14)
    source_files = ["src/sca3_compass/robustness_pc_weight.py", "tests/test_robustness_pc_weight.py",
                    "scripts/robustness_pc_weight_diagnostic.py", "src/sca3_compass/robustness_weighting.py",
                    "src/sca3_compass/robustness_projection.py", "src/sca3_compass/robustness_cone.py",
                    "src/sca3_compass/robustness_simes.py", "src/sca3_compass/robustness_universal.py"]
    receipt = {
        "runtime_seconds": {"build": build_seconds, "continuous_u_evaluation": evaluation_seconds,
                            "direct_agreement": agreement_seconds, "total": time.perf_counter() - start},
        "agreement": agreement, "model_integrals": integrals,
        "evaluation_bank_sha256": digest(bank), "evaluation_u_sha256": digest(u),
        "evaluation_midpoint_hits": int(np.count_nonzero(u * 64 % 1 == .5)),
        "builder": dict(result.diagnostics), "power": power,
        "source_sha256": {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in source_files},
        "limits": ["Known nuisance common-location IG radial model only; no fitted-nuisance guarantee",
                   "Gene dependence and informative/noncentral Q remain unproved",
                   "One predeclared shape/profile pair and model; no superiority or FDR claim",
                   "Independent evaluation uses same proposed profiles; off-grid effects do not establish generalization",
                   "Monte Carlo and continuous-U integration error; selected allocation-bank SE is not a confirmation interval",
                   "Midpoint convexification does not certify exact continuous-Q optimality"],
    }
    write_json(output / "result.json", receipt)
    arrays = {"marginal_proxy": baseline, "normalized_profiles": result.profiles}
    for name, mapping in (("height", result.tables), ("over", result.overspend), ("under", result.underspend),
                          ("fraction", result.fractions), ("prediction", result.predictions), ("se", result.prediction_se)):
        arrays.update({f"{method}_{name}": array for method, array in mapping.items()})
    with (output / "tables.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(json.dumps({"output": str(output), "runtime_seconds": receipt["runtime_seconds"],
                      "agreement": agreement, "projection": power["projection"],
                      "model_integrals": integrals}, default=plain, indent=2))


if __name__ == "__main__":
    main()
