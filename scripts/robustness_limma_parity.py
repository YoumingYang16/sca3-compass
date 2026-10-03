# SPDX-License-Identifier: GPL-2.0-or-later
"""Run one authentic R batch and record standard/robust Python parity.

From repository root:
  .venv/Scripts/python.exe scripts/robustness_limma_parity.py
Install the reference-only statmod dependency if needed:
  .tools/R-4.6.1/bin/Rscript.exe --vanilla external/limma-reference/install-statmod.R

No generated reference values are computed by the Python implementation.
All inputs, R outputs, checksums and comparison results remain reviewable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "external" / "limma-reference"
sys.path.insert(0, str(ROOT / "src"))

from sca3_compass.robustness_limma import squeeze_var


def parity_cases(seed=932761):
    rng = np.random.default_rng(seed)
    cases = []

    def add(name, x, df=20, tails=(0.05, 0.10)):
        cases.append((name, np.asarray(x, float), float(df), tails))

    for prior_df in (2.1, 5, 12, 100):
        for n in (64, 1000):
            add(f"scaled_f_df{prior_df:g}_n{n}", 0.6 * rng.f(20, prior_df, n))
    for n in (4, 8, 10, 11, 37, 512, 4000):
        add(f"chi_square_n{n}", 1.7 * rng.chisquare(20, n) / 20)
    add("constant", np.full(128, 2.75))
    add("nearly_constant", 2.75 * np.exp(np.linspace(-0.001, 0.001, 128)))
    add("tied_high_outliers", np.r_[np.ones(180), np.full(20, 100)])
    add("tied_low_variances", np.repeat([0.001, 0.3, 0.7, 1, 1.5, 2, 20], 41))
    add("one_hypervariable", np.r_[rng.f(20, 5, 999), 1e8])
    add("one_extreme_hypervariable", np.r_[rng.f(20, 5, 999), 1e200])
    add("tiny_variance_floor", np.r_[1e-200, 1e-14, 1e-9, rng.f(20, 5, 197)])
    add("infinite_prior_outlier", np.r_[np.ones(999), 1e9])
    add("infinite_prior_tiny_tails", np.r_[np.ones(980), np.full(20, 100)])
    x = rng.f(20, 7, 800)
    for scale in (1e-12, 1, 1e12):
        add(f"scale_{scale:g}", x * scale)
    add("no_winsorization", x, tails=(0, 0))
    add("custom_tails", x, tails=(0.1, 0.2))
    # The target is d=20; check some nearby scalar-df use as well.
    for df in (1, 4, 40, 100):
        add(f"common_df{df}", rng.f(df, 5, 512), df=df)
    for index in range(24):
        n = int(rng.integers(30, 1200))
        x = rng.f(20, rng.uniform(2.05, 50), n)
        if index % 3 == 0:
            x[:max(1, n//25)] *= 100
        if index % 3 == 1:
            x[:max(1, n//25)] *= 0.0001
        add(f"random_{index:02}", x)
    return cases


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_source_manifest():
    manifest = json.loads((REFERENCE / "source-manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["sources"]:
        data = (REFERENCE / entry["path"]).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != entry["sha256_lf"]:
            raise RuntimeError(f"Reference source checksum mismatch: {entry['path']}")
    return manifest


def compare(actual, expected, *, rtol, atol):
    actual, expected = np.broadcast_arrays(np.asarray(actual, float), np.asarray(expected, float))
    finite = np.isfinite(expected)
    nonfinite_match = np.array_equal(np.isposinf(actual), np.isposinf(expected))
    nonfinite_match &= not np.isnan(actual).any() and not np.isnan(expected).any()
    differences = np.abs(actual[finite] - expected[finite])
    scale = np.maximum(np.abs(expected[finite]), np.finfo(float).tiny)
    passed = nonfinite_match and bool(np.all(differences <= atol + rtol*np.abs(expected[finite])))
    return {"passed": bool(passed), "max_absolute_error": float(np.max(differences, initial=0)),
            "max_relative_error": float(np.max(differences / scale, initial=0)),
            "infinity_pattern_matches": bool(nonfinite_match)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_r = ROOT / ".tools" / "R-4.6.1" / "bin" / "Rscript.exe"
    parser.add_argument("--rscript", default=os.environ.get("RSCRIPT", str(default_r) if default_r.exists() else shutil.which("Rscript")))
    parser.add_argument("--output-dir", type=Path, default=REFERENCE / "parity")
    parser.add_argument("--seed", type=int, default=932761)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    args = parser.parse_args(argv)
    if not args.rscript:
        parser.error("Rscript is required: this check cannot claim parity without executing author R")
    manifest = verify_source_manifest()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = parity_cases(args.seed)
    input_path, output_path = out / "inputs.csv", out / "author-r.csv"
    with input_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "index", "variance", "df", "lower", "upper", "mean", "variance_factor"])
        for name, x, df, tails in cases:
            means = np.sin(np.arange(x.size) * 0.39) * 5
            for index, value in enumerate(x):
                writer.writerow([name, index, repr(float(value)), df, *tails, repr(float(means[index])), 0.8])
    command = [str(args.rscript), "--vanilla", str(REFERENCE / "reference-batch.R"),
               str(REFERENCE), str(input_path), str(output_path)]
    begin = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=180)
    r_seconds = time.perf_counter() - begin
    (out / "author-r.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    rows = {}
    with output_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.setdefault((row["case"], int(row["robust"])), []).append(row)
    comparisons, failures = [], []
    # Fixed numerical tolerances, recorded alongside the measured errors.
    tolerances = {False: {"rtol": 2e-10, "atol": 2e-13},
                  True: {"rtol": 2e-9, "atol": 2e-12}}
    for name, x, df, tails in cases:
        for robust in (False, True):
            reference_rows = rows[name, int(robust)]
            if len(reference_rows) != x.size:
                raise RuntimeError(f"Incomplete author R output for {name}")
            expected = {key: np.array([float(row[key]) for row in reference_rows])
                        for key in ("posterior", "prior_df", "prior_variance", "global_prior_df",
                                    "total_df", "t_one_sided", "t_two_sided")}
            posterior, prior, diagnostic = squeeze_var(x, df=df, robust=robust, winsor_tail_p=tails)
            total = np.minimum(df + prior, x.size * df)
            statistic = np.sin(np.arange(x.size) * 0.39) * 5 / np.sqrt(0.8 * posterior)
            actual = {"posterior": posterior, "prior_df": prior,
                      "prior_variance": diagnostic["prior_variance"],
                      "global_prior_df": diagnostic["global_prior_df"], "total_df": total,
                      "t_one_sided": t.sf(statistic, total), "t_two_sided": 2*t.cdf(-np.abs(statistic), total)}
            errors = {key: compare(actual[key], expected[key], **tolerances[robust]) for key in expected}
            entry = {"case": name, "robust": robust, "n": int(x.size), "df": df,
                     "branch": diagnostic["branch"], "passed": all(v["passed"] for v in errors.values()),
                     "errors": errors}
            comparisons.append(entry)
            if not entry["passed"]:
                failures.append(entry)
    timings = {}
    benchmark_x = np.random.default_rng(672).f(20, 5, 4000)
    for robust in (False, True):
        squeeze_var(benchmark_x, robust=robust)
        samples = []
        for _ in range(args.benchmark_repeats):
            begin = time.perf_counter()
            squeeze_var(benchmark_x, robust=robust)
            samples.append(1000*(time.perf_counter() - begin))
        timings["robust" if robust else "standard"] = {"n": 4000, "repeats": args.benchmark_repeats,
                                                         "median_ms": float(np.median(samples))}
    report = {"created_utc": datetime.now(UTC).isoformat(),
              "author_R_executed": True, "passed": not failures, "seed": args.seed,
              "scope": "classic squeezeVar, no covariate, common df; full vector; primary target d=20",
              "t_check": "squeezeVar plus eBayes df.total=min(d+df.prior,G*d), one/two-sided t tails",
              "not_checked": "lmFit, full eBayes object, coefficient fitting, trend/unequal df, PC/BY or FDR",
              "cases": len(cases), "comparisons": len(comparisons), "tolerances": tolerances,
              "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
              "scipy": scipy.__version__, "R_stdout": result.stdout,
              "R_command": command, "R_batch_seconds": r_seconds, "timing": timings,
              "source_manifest": manifest, "source_manifest_sha256": _sha256(REFERENCE / "source-manifest.json"),
              "python_module_sha256": _sha256(ROOT / "src" / "sca3_compass" / "robustness_limma.py"),
              "r_harness_sha256": _sha256(REFERENCE / "reference-batch.R"),
              "parity_script_sha256": _sha256(__file__),
              "input_sha256": _sha256(input_path), "author_output_sha256": _sha256(output_path),
              "results": comparisons}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": not failures, "cases": len(cases), "comparisons": len(comparisons),
                      "failed": [{"case": entry["case"], "robust": entry["robust"],
                                  "errors": {key: value for key, value in entry["errors"].items() if not value["passed"]}}
                                 for entry in failures], "timing": timings, "report": str(out / "report.json")}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
