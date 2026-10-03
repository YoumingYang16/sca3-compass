"""Frozen, TRAIN-only repeated CF scale development diagnostic.

The existing CF, central and point-joint implementations are called unchanged.
No testing probabilities, effect labels, FDR or power are computed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import warnings

# Set before NumPy/SciPy import, including inside the frozen child process.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
                  "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
sys.dont_write_bytecode = True

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from robustness_cf_scale_diagnostic import cf_fit
from robustness_screen import data
from sca3_compass.robustness_domain import domain_variance
from sca3_compass.robustness_joint_scale import fit_joint_pattern_scale
from sca3_compass.robustness_methods import contrasts, fit_calibration, tyler_shape
from sca3_compass.robustness_prior import fit_energy_prior

INDICES = (0, 2, 3, 4, 6, 9, 10)
DEVSEED = 7105211
CONFIG = "configs/robustness_central_anchor_development.json"
METHODS = ("fixed_reference", "fixed_central", "free_central", "cf_base",
           "cf_observed_scale", "point_joint")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_value(value):
    """Preserve nonfinite raw receipts explicitly, never silently replace them."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"__nonfinite_float__": repr(value)}
    return value


def write_new(path, value, compressed=False):
    """Exclusive creation prevents accidental overwrites or run resumption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(json_value(value), sort_keys=True, allow_nan=False,
                          separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        if compressed:
            with gzip.GzipFile(filename="", mode="wb", fileobj=stream,
                               mtime=0, compresslevel=3) as archive:
                archive.write(payload)
        else:
            stream.write(payload)


def seed_receipt(index, repeat, seed=DEVSEED):
    sequence = np.random.SeedSequence([seed, index, repeat])
    return {"entropy": [seed, index, repeat], "bit_generator": "PCG64",
            "state_u32": sequence.generate_state(8).tolist()}


def generate_observations(case, index, repeat, seed=DEVSEED):
    generated = data(np.random.default_rng(np.random.SeedSequence([seed, index, repeat])), case)
    # Never unpack or index the generator's third (truth) return value.
    return generated[0], generated[1]


def extract_train(z, calibration):
    """Exactly reproduce reference fold 0 inputs, slicing TRAIN before arithmetic.

    Tyler uses uncentered contrasts, as crossfit_residual_energy does. The
    energy prior uses centered contrasts, as energy_prior_candidates does.
    No operation other than row selection touches held observations.
    """
    rows = np.arange(len(z))[np.arange(len(z)) % 2 != 0]
    training = np.asarray(z[rows], float)
    calibration = np.asarray(calibration, float)
    if (training.ndim != 3 or training.shape[1] != 4 or len(training) < 16
            or not np.isfinite(training).all() or not np.isfinite(calibration).all()):
        raise ValueError("Finite TRAIN G x 4 x K and calibration arrays required")
    _, studies, pipelines = training.shape
    calibration_fit = fit_calibration(calibration.reshape(-1, pipelines))
    y = training @ contrasts(pipelines)
    shape, shape_receipt = tyler_shape(y.transpose(0, 2, 1).reshape(-1, studies))
    means = training.mean(-1)
    residual = (training - means[..., None]) @ contrasts(pipelines)
    q = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual) / (1 - calibration_fit.rho)
    dimension = studies * (pipelines - 1)
    projection = (1 + (pipelines - 1) * calibration_fit.rho) / pipelines
    prior = fit_energy_prior(q, dimension)
    df = np.inf if prior["gaussian_bic_selected"] else prior["df"] + dimension
    variance = (np.full(len(training), projection * prior["scatter"]) if np.isinf(df)
                else projection * (prior["df"] * prior["scatter"] + q) / df)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise FloatingPointError("Invalid extracted conditional variance")
    inputs = {"means": means, "base_variance": variance, "shape": shape,
              "df": None if np.isinf(df) else df, "q": q,
              "residual_dimension": dimension, "projection_variance": projection}
    receipt = {"train_indices": rows, "outer_fold": 0, "training_rows": len(rows),
               "calibration_fit": calibration_fit.__dict__, "shape_fit": shape_receipt,
               "target_only_prior": prior, "uses_held_rows": False, "uses_truth": False}
    issues = []
    for label, output in (("calibration", calibration_fit.__dict__),
                          ("shape", shape_receipt), ("prior", prior)):
        if not output["converged"]:
            issues.append(label + ":not_converged")
        if output.get("optimizer_success") is False:
            issues.append(label + ":optimizer_failure_even_if_recovered")
    receipt["numerical_issues"] = issues
    return inputs, receipt


def describe_input(inputs):
    v = inputs["base_variance"]
    energy = np.einsum("ni,ij,nj->n", inputs["means"],
                       np.linalg.inv(inputs["shape"]), inputs["means"])
    return {"gaussian": inputs["df"] is None, "conditional_df": inputs["df"],
            "log_variance_spread": float(np.std(np.log(v))),
            "variance_min": float(np.min(v)), "variance_median": float(np.median(v)),
            "variance_max": float(np.max(v)),
            "observed_to_base_frequency_scale_ratio": float(max(1., np.median(energy) / 4 / np.median(v)))}


def classify(label, raw):
    """Conservative failure policy; keep raw estimates even when ineligible."""
    if label.startswith("cf_"):
        if raw.get("status") == "NO_VARIANCE_CONTRAST_NOT_IDENTIFIED":
            return {"status": "NOT_IDENTIFIED", "tau": None, "boundary": False,
                    "issues": [], "eligible_for_estimate_summary": False}
        tau = raw.get("tau")
        issues = ["local_optimizer_failed"] if any(not f["success"] for f in raw["local_fits"]) else []
        if not np.isfinite(raw["grid_objective"]).all() or not np.isfinite(raw["objective"]):
            issues.append("nonfinite_objective")
        boundary = bool(raw["active_boundary"])
    elif label == "point_joint":
        d = raw["diagnostics"]
        tau = raw.get("tau")
        issues = []
        if not d["converged"]:
            issues.append("selected_fit_not_converged")
        if d["numerical_failure_count"]:
            issues.append("numerical_failure_in_any_start")
        if d["central_initialization_failure"] is not None:
            issues.append("central_initialization_failure")
        boundary = bool(d.get("active_bound", False))
    else:
        tau = raw["variance_multiplier"]
        issues = []
        central = raw.get("receipt", {}).get("central_f", {})
        if central and not central["converged"]:
            issues.append("central_not_converged")
        boundary = bool(central.get("bound_active", False) and central.get("free_scale_adopted", True))
    if tau is None or not np.isfinite(tau) or tau <= 0:
        issues.append("invalid_scale")
    return {"status": "FAILED_NUMERICAL" if issues else "FIXED_REFERENCE" if label == "fixed_reference" else "ESTIMATE",
            "tau": tau, "boundary": boundary, "issues": issues,
            "eligible_for_estimate_summary": not issues}


def fit_one(label, inputs, nuisance_issues=(), include_joint=True):
    if label == "point_joint" and not include_joint:
        return {"status": "NOT_RUN_COST_PROTOCOL", "tau": None, "boundary": False,
                "eligible_for_estimate_summary": False, "elapsed_seconds": 0.}
    x, v, shape = (inputs[k] for k in ("means", "base_variance", "shape"))
    df = np.inf if inputs["df"] is None else inputs["df"]
    started = time.perf_counter()
    raw = None
    caught = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if label == "fixed_reference":
                raw = {"variance_multiplier": 1., "meaning": "unchanged conditional TRAIN variance"}
            elif label.startswith("cf_"):
                raw = cf_fit(x, v, shape, df, frequency_mode=label.removeprefix("cf_"))
            elif label == "point_joint":
                raw = fit_joint_pattern_scale(x, v, shape, df)
            else:
                mode = "anchored_fcentral" if label == "fixed_central" else "transport_fcentral"
                selected, receipt = domain_variance(x, inputs["q"], inputs["residual_dimension"],
                    inputs["projection_variance"], mode, shape=shape)
                raw = {"variance_multiplier": selected / inputs["projection_variance"], "receipt": receipt}
        answer = classify(label, raw)
        numerical_warnings = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
        answer["issues"].extend("runtime_warning:" + message for message in numerical_warnings)
        answer["nuisance_issues"] = list(nuisance_issues)
        if answer["status"] != "NOT_IDENTIFIED":
            if numerical_warnings:
                answer.update(status="FAILED_NUMERICAL", eligible_for_estimate_summary=False)
            if nuisance_issues:
                answer.update(status="FAILED_NUISANCE", eligible_for_estimate_summary=False)
        answer["raw"] = raw
    except Exception as error:
        answer = {"status": "FAILED_EXCEPTION", "tau": None, "boundary": False,
                  "eligible_for_estimate_summary": False, "raw": raw,
                  "error_type": type(error).__name__, "error_message": str(error),
                  "traceback": traceback.format_exc()}
    answer["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in caught]
    answer["elapsed_seconds"] = time.perf_counter() - started
    return answer


def freeze_sources(root, output):
    """Copy all package modules (including transitive/lazy imports) plus scripts."""
    paths = [root / relative for relative in (
        "scripts/robustness_cf_repeated_diagnostic.py", "scripts/robustness_cf_scale_diagnostic.py",
        "scripts/robustness_continuous_floor_diagnostic.py", "scripts/robustness_screen.py",
        "tests/test_robustness_cf_repeated_diagnostic.py", "docs/robustness_cf_diagnostic.md",
        "pyproject.toml", CONFIG)]
    paths += sorted((root / "src/sca3_compass").glob("*.py"))
    manifest = {}
    for path in paths:
        key = path.relative_to(root).as_posix()
        before = sha256(path)
        copied = output / "source" / key
        copied.parent.mkdir(parents=True, exist_ok=True)
        with copied.open("xb") as stream:
            stream.write(path.read_bytes())
        if sha256(copied) != before or sha256(path) != before:
            raise RuntimeError("Source changed during snapshot: " + key)
        manifest[key] = before
    return manifest


def verify_sources(output, expected):
    for relative, digest in expected.items():
        if sha256(output / "source" / relative) != digest:
            raise RuntimeError("Frozen source hash mismatch: " + relative)


def protocol(root, repetitions, include_joint, sources):
    cases = json.loads((root / CONFIG).read_text(encoding="utf-8"))["cases"]
    return {"phase": "DEVELOPMENT_ONLY_NOT_ACCEPTANCE", "frozen_at_utc": utc(),
            "seed": DEVSEED, "case_indices": list(INDICES), "repetitions_per_case": repetitions,
            "planned_train_folds": len(INDICES) * repetitions, "outer_fold": 0,
            "seed_rule": "SeedSequence([7105211, case_index, repetition]); PCG64; independent per repetition",
            "cases": {str(i): cases[i] for i in INDICES}, "source_sha256": sources,
            "methods": list(METHODS), "include_joint": include_joint,
            "joint_budget_rule": "Pilot uses unchanged default 250 iterations and four starts; full run includes joint if pilot total <=120 seconds and individual joint fit <=30 seconds. Decision uses cost only.",
            "cf_bounds": [.001, 1000.], "cf_frequency_modes": ["base", "observed_scale"],
            "fixed_central_definition": "anchored_fcentral: compare fixed projection scale with free central scale on same lower 40% by existing criterion; can retain tau=1",
            "free_central_definition": "transport_fcentral: lower-40% selection-corrected F fit, with quartile anchor/9 to anchor bounds",
            "null_effect_independent_of_variance": "ASSUMPTION_UNPROVED; CF requires full effect vector independent of conditional noise variance",
            "central_assumption": "negligible nonnull contribution to lower 40%; unproved",
            "joint_assumption": "fixed point-pattern effect mixture is a working approximation; local convergence only",
            "failure_policy": "All exceptions, runtime numerical warnings, CF failed local optimizers, joint numerical events in any start, selected nonconvergence, and nuisance numerical failures are failures. Raw outputs retained; no fallback. NOT_IDENTIFIED never counts as an estimate.",
            "reporting": "Counts, finite raw/eligible multipliers, log spread, boundaries and paired observed/base ratio. No FDR/power, accuracy ranking or acceptance.",
            "gaussian_constant_variance": "NO_VARIANCE_CONTRAST_NOT_IDENTIFIED",
            "held_rows_used": False, "truth_return_read": False, "floor_convolution": False,
            "blas_threads": 1, "multiprocessing": False,
            "environment": {"python": sys.version, "executable": sys.executable,
                            "numpy": np.__version__, "scipy": scipy.__version__,
                            "platform": platform.platform()},
            "schedule": [{"case_index": i, "repeat": r, "seed": seed_receipt(i, r)}
                         for i in INDICES for r in range(repetitions)]}


def spread(values):
    values = np.asarray(values, float)
    if not len(values):
        return {"n": 0}
    logs = np.log(values)
    return {"n": len(values), "min": float(values.min()), "q10": float(np.quantile(values, .1)),
            "q25": float(np.quantile(values, .25)), "median": float(np.median(values)),
            "q75": float(np.quantile(values, .75)), "q90": float(np.quantile(values, .9)),
            "max": float(values.max()), "log_sd": float(logs.std()),
            "geometric_sd": float(np.exp(logs.std())), "max_min_ratio": float(values.max() / values.min())}


def summarize(records, plan):
    result = {"phase": plan["phase"], "completed_train_folds": len(records),
              "planned_train_folds": plan["planned_train_folds"], "cases": {}}
    for index in INDICES:
        rows = [r for r in records if r["case_index"] == index]
        methods = {}
        for method in METHODS:
            outputs = [r["fits"][method] for r in rows]
            finite = [o["tau"] for o in outputs if o.get("tau") is not None and np.isfinite(o["tau"]) and o["tau"] > 0]
            eligible = [o["tau"] for o in outputs if o["eligible_for_estimate_summary"]]
            methods[method] = {"denominator": len(rows), "statuses": dict(Counter(o["status"] for o in outputs)),
                               "boundaries": sum(o["boundary"] for o in outputs),
                               "eligible_boundaries": sum(o["boundary"] and o["eligible_for_estimate_summary"] for o in outputs),
                               "all_finite_raw_tau": spread(finite), "eligible_tau": spread(eligible),
                               "elapsed_seconds": sum(o["elapsed_seconds"] for o in outputs)}
        pairs = [(r["fits"]["cf_base"], r["fits"]["cf_observed_scale"]) for r in rows]
        ratios = [b["tau"] / a["tau"] for a, b in pairs if a["eligible_for_estimate_summary"] and b["eligible_for_estimate_summary"]]
        result["cases"][str(index)] = {"name": plan["cases"][str(index)]["name"], "methods": methods,
            "gaussian_folds": sum(r.get("input_summary", {}).get("gaussian", False) for r in rows),
            "nuisance_failure_folds": sum(bool(r.get("nuisance_issues")) for r in rows),
            "cf_observed_over_base": spread(ratios),
            "cf_observed_over_base_gt2": sum(v > 2 for v in ratios),
            "cf_observed_over_base_lt_half": sum(v < .5 for v in ratios),
            "input_log_variance_spread": [r.get("input_summary", {}).get("log_variance_spread") for r in rows]}
    return result


def execute_frozen(output):
    started = time.perf_counter()
    plan = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    protocol_hash = sha256(output / "protocol.json")
    verify_sources(output, plan["source_sha256"])
    expected_root = (output / "source").resolve()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and (getattr(module, "__name__", "").startswith("sca3_compass")
                         or getattr(module, "__name__", "") in ("robustness_screen", "robustness_cf_scale_diagnostic")):
            if not Path(filename).resolve().is_relative_to(expected_root):
                raise RuntimeError("Module escaped frozen snapshot: " + filename)
    records = []
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        if any(p["num_threads"] != 1 for p in pools):
            raise RuntimeError("Thread limit did not apply")
        write_new(output / "execution-start.json", {"utc": utc(), "protocol_sha256": protocol_hash,
                  "threadpools": pools, "pid": os.getpid(), "source_root": str(expected_root)})
        for ordinal, item in enumerate(plan["schedule"], 1):
            index, repeat = item["case_index"], item["repeat"]
            case_dir = output / f"case-{index:02d}" / f"repeat-{repeat:03d}"
            case_start = time.perf_counter()
            record = {"case_index": index, "repeat": repeat, "started_at_utc": utc(),
                      "seed": item["seed"], "fits": {}}
            write_new(case_dir / "started.json", record)
            try:
                z, calibration = generate_observations(plan["cases"][str(index)], index, repeat, plan["seed"])
                record["generation_seconds"] = time.perf_counter() - case_start
                write_new(case_dir / "observations.json.gz", {"z": z, "calibration": calibration,
                          "notice": "full observed arrays for reproducibility only; held rows never passed to fit; generator truth deliberately not read or saved"}, True)
                stamp = time.perf_counter()
                with warnings.catch_warnings(record=True) as nuisance_warnings:
                    warnings.simplefilter("always")
                    inputs, receipt = extract_train(z, calibration)
                receipt["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in nuisance_warnings]
                receipt["numerical_issues"].extend("runtime_warning:" + str(w.message) for w in nuisance_warnings if issubclass(w.category, RuntimeWarning))
                record["extraction_seconds"] = time.perf_counter() - stamp
                record["nuisance_issues"] = receipt["numerical_issues"]
                write_new(case_dir / "inputs.json.gz", inputs, True)
                write_new(case_dir / "extraction.json", receipt)
                record["inputs_sha256"] = sha256(case_dir / "inputs.json.gz")
                record["input_summary"] = describe_input(inputs)
                for label in METHODS:
                    answer = fit_one(label, inputs, receipt["numerical_issues"], plan["include_joint"])
                    write_new(case_dir / (label + ".json.gz"), answer, True)
                    record["fits"][label] = {k: v for k, v in answer.items() if k not in ("raw", "traceback")}
            except Exception as error:
                record["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
                for label in METHODS:
                    record["fits"].setdefault(label, {"status": "FAILED_EXTRACTION", "tau": None,
                        "boundary": False, "eligible_for_estimate_summary": False, "elapsed_seconds": 0.})
            record.update(completed_at_utc=utc(), elapsed_seconds=time.perf_counter() - case_start)
            write_new(case_dir / "result.json", record)
            records.append(record)
            print(json.dumps({"utc": record["completed_at_utc"], "completed": ordinal,
                "planned": plan["planned_train_folds"], "case_index": index, "repeat": repeat,
                "fits": {m: {"status": r["status"], "tau": r["tau"]} for m, r in record["fits"].items()},
                "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
    verify_sources(output, plan["source_sha256"])
    if sha256(output / "protocol.json") != protocol_hash:
        raise RuntimeError("Protocol changed after freeze")
    write_new(output / "summary.json", summarize(records, plan))
    write_new(output / "completion.json", {"utc": utc(), "completed": len(records),
              "planned": plan["planned_train_folds"], "elapsed_seconds": time.perf_counter() - started,
              "protocol_sha256": protocol_hash, "source_hashes_verified_after_run": True,
              "max_point_joint_seconds": max(r["fits"]["point_joint"]["elapsed_seconds"] for r in records)})
    # Manifest includes every source/input/result/progress file, excluding itself.
    files = sorted(p for p in output.rglob("*") if p.is_file())
    manifest = {p.relative_to(output).as_posix(): sha256(p) for p in files}
    write_new(output / "manifest.json", manifest)
    for relative, digest in manifest.items():
        if sha256(output / relative) != digest:
            raise RuntimeError("Output hash mismatch: " + relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, choices=(1, 32), default=32)
    parser.add_argument("--skip-joint", action="store_true", help="Only if pilot violates frozen cost rule")
    parser.add_argument("--execute-frozen", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.execute_frozen:
        execute_frozen(output)
        return
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = freeze_sources(root, output)
    write_new(output / "protocol.json", protocol(output / "source", args.repetitions, not args.skip_joint, sources))
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(output / "source" / p) for p in ("src", "scripts"))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable, "-B", str(output / "source/scripts/robustness_cf_repeated_diagnostic.py"),
               "--execute-frozen", "--output", str(output)]
    completed = subprocess.run(command, cwd=output / "source", env=env, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
