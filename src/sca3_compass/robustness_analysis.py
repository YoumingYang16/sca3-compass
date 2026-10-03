"""Deterministic DEVELOPMENT summaries of completed robustness screens.

Pairing is by array position within ONE input/scenario, as written by
robustness_screen.one; genes, methods, and cases are not independent reps.
No simulation, registry update, decision gate, or confirmation is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_BASELINES = (
    "t_mean_simple",
    "limma_standard",
    "limma_robust",
    "radial_studentized_simple",
    "conditional_simes_assumption_reference_PC",
)
INTERVAL_REFERENCE = "https://www.cs.mcgill.ca/~colt2009/papers/012.pdf"
DEVELOPMENT_NOTICE = (
    "DEVELOPMENT ONLY. Descriptive exploration of existing simulations; "
    "these intervals are not confirmatory. No success gate, acceptance decision, "
    "or cumulative confirmation error spending is implemented."
)
NUMERICAL_KEYS = frozenset({
    "numerical_failure", "numerical_failures", "numerical_failure_count",
    "nonfinite_count", "nonfinite_values", "nan_count", "inf_count",
    "invalid_pvalue_count", "invalid_pvalues", "invalid_probability_count",
})


def _number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context}: expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context}: expected a finite number")
    return number


def _bounded(values: Iterable[float], lower: float, upper: float) -> list[float]:
    result = [_number(value, "bounded observation") for value in values]
    if not result:
        raise ValueError("At least one repetition is required")
    if any(value < lower or value > upper for value in result):
        raise ValueError(f"Observations must lie in [{lower}, {upper}]")
    return result


def empirical_bernstein_interval(
    values: Iterable[float], *, lower: float = 0.0, upper: float = 1.0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Two-sided bounded mean interval across independent repetitions.

    Maurer & Pontil (2009), Theorem 4 (Theorem 11 also allows independent,
    nonidentical observations). Apply each one-sided bound at alpha/2 and
    rescale [0, 1] to [lower, upper]. With unbiased sample variance s²:
    radius = sqrt(2*s²*log(4/alpha)/n)
             + 7*(upper-lower)*log(4/alpha)/(3*(n-1)).
    A single observation has only the full support interval. In particular,
    zero observed variance does NOT imply zero uncertainty.
    """
    lower, upper = _number(lower, "lower"), _number(upper, "upper")
    alpha = _number(alpha, "alpha")
    if lower >= upper or not math.isfinite(upper - lower):
        raise ValueError("A finite positive support width is required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    data = _bounded(values, lower, upper)
    n = len(data)
    mean = math.fsum(data) / n
    variance = math.fsum((value - mean) ** 2 for value in data) / (n - 1) if n > 1 else None
    log_term = math.log(4.0) - math.log(alpha)
    radius = (
        math.sqrt(2 * variance * log_term / n)
        + 7 * (upper - lower) * log_term / (3 * (n - 1))
        if variance is not None else None
    )
    return {
        "method": "two_sided_empirical_bernstein", "n": n, "mean": mean,
        "sample_variance": variance, "alpha": alpha, "radius": radius,
        "lower": max(lower, mean - radius) if radius is not None else lower,
        "upper": min(upper, mean + radius) if radius is not None else upper,
        "support": [lower, upper],
        "status": "computed" if n > 1 else "full_support_single_repetition",
    }


def paired_power_differences(
    candidate: Sequence[float], baseline: Sequence[float],
) -> list[float]:
    """Subtract matching repetition indices; never truncate or sort arrays."""
    a, b = _bounded(candidate, 0, 1), _bounded(baseline, 0, 1)
    if len(a) != len(b):
        raise ValueError("Paired power arrays must have identical repetition counts")
    return [x - y for x, y in zip(a, b, strict=True)]


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lo, hi = math.floor(index), math.ceil(index)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def _names(names: Sequence[str], context: str) -> list[str]:
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{context}: at least one nonempty method name is required")
    if len(set(names)) != len(names):
        raise ValueError(f"{context}: duplicate method name")
    return list(names)


def diagnostic_counts(diagnostics: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count observed diagnostic events, separately from affected repetitions.

    Diagnostics are shared across methods. Nulls (including serialized limma
    infinite-prior sentinels) are counted as nulls, not numerical failures.
    Explicit numerical counters may overlap each other and nonfinite leaves.
    """
    checks: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    numerical: Counter[str] = Counter()
    nonfinite: Counter[str] = Counter()
    nulls: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    convergence_reps = numerical_reps = fit_reps = 0

    def visit(value: Any, path: str, events: set[str]) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                if key == "converged":
                    if not isinstance(item, bool):
                        raise ValueError(f"diagnostics.{child}: converged must be boolean")
                    checks[child] += 1
                    if not item:
                        failures[child] += 1
                        events.add("convergence")
                        if path == "fit" or path.startswith("fit."):
                            events.add("fit")
                if key == "not_run" and item is True:
                    skipped[child] += 1
                if key in NUMERICAL_KEYS:
                    count = int(item) if isinstance(item, bool) else _number(item, child)
                    if count < 0 or int(count) != count:
                        raise ValueError(f"diagnostics.{child}: expected nonnegative integer count")
                    numerical[child] += int(count)
                    if count:
                        events.add("numerical")
                visit(item, child, events)
        elif isinstance(value, list):
            for item in value:
                visit(item, f"{path}[]", events)
        elif value is None:
            nulls[path] += 1
        elif isinstance(value, float) and not math.isfinite(value):
            nonfinite[path] += 1
            events.add("numerical")

    for diagnostic in diagnostics:
        events: set[str] = set()
        visit(diagnostic, "", events)
        convergence_reps += "convergence" in events
        numerical_reps += "numerical" in events
        fit_reps += "fit" in events
    return {
        "repetitions_with_diagnostics": len(diagnostics),
        "convergence_checks": sum(checks.values()),
        "nonconvergence_count": sum(failures.values()),
        "fit_nonconvergence_count": sum(v for k, v in failures.items() if k.startswith("fit.")),
        "repetitions_with_nonconvergence": convergence_reps,
        "repetitions_with_fit_nonconvergence": fit_reps,
        "numerical_counter_total": sum(numerical.values()),
        "nonfinite_numeric_values": sum(nonfinite.values()),
        "repetitions_with_numerical_diagnostics": numerical_reps,
        "convergence_checks_by_path": dict(sorted(checks.items())),
        "nonconvergence_by_path": dict(sorted(failures.items())),
        "numerical_counters_by_path": dict(sorted(numerical.items())),
        "nonfinite_values_by_path": dict(sorted(nonfinite.items())),
        "null_values_by_path": dict(sorted(nulls.items())),
        "not_run_by_path": dict(sorted(skipped.items())),
    }


def _merge_diagnostics(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = diagnostic_counts([])
    for item in items:
        for key, value in item.items():
            if isinstance(value, dict):
                for path, count in value.items():
                    total[key][path] = total[key].get(path, 0) + count
            else:
                total[key] += value
    return total


def _seconds(value: Any, context: str) -> float | None:
    if value is None:
        return None
    seconds = _number(value, context)
    if seconds < 0:
        raise ValueError(f"{context}: elapsed seconds must be nonnegative")
    return seconds


def _validate_artifact(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    settings = artifact.get("settings", {})
    if settings.get("phase") != "DEVELOPMENT_ONLY" or settings.get("independent_confirmation") is True:
        raise ValueError("Only DEVELOPMENT_ONLY artifacts can be analyzed")
    hashes = settings.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes or any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value)
        for value in hashes.values()
    ):
        raise ValueError("settings.source_sha256 must contain exact source hashes")
    # Settings are preserved verbatim; reject non-JSON metadata before output.
    json.dumps(settings, allow_nan=False)
    scenarios = artifact.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("A completed screen with nonempty scenarios is required; partial screens are unsupported")
    expected_cases = settings.get("cases")
    if expected_cases is not None and [s["case"] for s in scenarios] != expected_cases:
        raise ValueError("Scenario order/metadata must match settings.cases; incomplete or reordered screen")
    cases_seen: set[str] = set()
    parsed = []
    for index, scenario in enumerate(scenarios):
        case = scenario.get("case", {})
        name = case.get("name")
        if not isinstance(name, str) or not name or name in cases_seen:
            raise ValueError("Scenario names must be nonempty and unique within each input")
        cases_seen.add(name)
        if not isinstance(case.get("distribution"), str):
            raise TypeError(f"{name}: case.distribution is required")
        n = scenario.get("repetitions")
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"{name}: positive integer repetitions required")
        if settings.get("repetitions", n) != n:
            raise ValueError(f"{name}: repetitions differ from settings")
        if not isinstance(scenario.get("power_defined"), bool):
            raise TypeError(f"{name}: power_defined is required")
        if "replicated_signed_truths" in scenario:
            truths = _number(scenario["replicated_signed_truths"], f"{name}: truth count")
            if truths < 0 or int(truths) != truths or bool(truths) != scenario["power_defined"]:
                raise ValueError(f"{name}: inconsistent power_defined/truth count")
        rows = {}
        for row in scenario.get("rows", []):
            method = row.get("method")
            if not isinstance(method, str) or not method or method in rows:
                raise ValueError(f"{name}: method names must be nonempty and unique")
            metrics = {}
            for key in ("fdp_by_repetition", "power_by_repetition"):
                values = row.get(key)
                if not isinstance(values, list) or len(values) != n:
                    raise ValueError(f"{name}/{method}/{key}: repetition count mismatch")
                try:
                    metrics[key] = _bounded(values, 0, 1)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{name}/{method}/{key}: {exc}") from exc
            # This format has implicit IDs only; do not silently ignore newer IDs.
            if "repetition_ids" in row or "repetition_ids" in scenario:
                raise ValueError("Explicit repetition_ids are unsupported; this format pairs by position")
            rows[method] = metrics
        if not rows:
            raise ValueError(f"{name}: method rows are required")
        diagnostics = scenario.get("diagnostics")
        if diagnostics is None:
            diagnostics = []
        if (not isinstance(diagnostics, list) or len(diagnostics) not in (0, n)
                or any(not isinstance(d, dict) for d in diagnostics)):
            raise ValueError(f"{name}: diagnostics must be absent or have one object per repetition")
        parsed.append({
            "case_index": index, "case": case, "repetitions": n,
            "power_defined": scenario["power_defined"], "rows": rows,
            "replicated_signed_truths": scenario.get("replicated_signed_truths"),
            "elapsed_seconds": _seconds(scenario.get("elapsed_seconds"), name),
            "diagnostics": diagnostic_counts(diagnostics),
        })
    return parsed


def _point_interval(interval: dict[str, Any]) -> dict[str, Any]:
    return {**interval, "mean_difference": interval["mean"],
            "mean_difference_pp": 100 * interval["mean"]}


def _best_interval(comparisons: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """min of simultaneous paired bounds targets mu_candidate - max mu_baseline."""
    return {
        "method": "minimum_of_simultaneous_paired_bounds",
        "mean": min(v["mean"] for v in comparisons.values()),
        "lower": min(v["lower"] for v in comparisons.values()),
        "upper": min(v["upper"] for v in comparisons.values()),
        "target": "candidate_mean_minus_maximum_baseline_population_mean",
    }


def _average_intervals(intervals: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not intervals:
        return None
    return {
        "method": "equal_case_average_of_simultaneous_bounds",
        "case_count": len(intervals),
        **{key: math.fsum(row[key] for row in intervals) / len(intervals)
           for key in ("mean", "lower", "upper")},
    }


def _worst(cases: Sequence[dict[str, Any]], intervals: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not cases:
        return None
    index = min(range(len(cases)), key=lambda i: intervals[i]["mean"])
    value = intervals[index]["mean"]
    return {
        "case_index": cases[index]["case_index"], "case": cases[index]["case"]["name"],
        "mean_difference": value, "mean_difference_pp": 100 * value,
        "regression_pp": max(0.0, -100 * value),
        "interval_for_minimum_case_population_difference": {
            "lower": min(row["lower"] for row in intervals),
            "upper": min(row["upper"] for row in intervals),
        },
    }


def _macro(cases: Sequence[dict[str, Any]], candidate: str, baselines: Sequence[str]) -> dict[str, Any]:
    power_cases = [case for case in cases if case["power_defined"]]
    comparisons = {}
    for baseline in [*baselines, "best_baseline_mean"]:
        intervals = [case["candidates"][candidate]["vs_baselines"][baseline]
                     if baseline in baselines else case["candidates"][candidate]["vs_best_baseline_mean"]
                     for case in power_cases]
        average = _average_intervals(intervals)
        comparisons[baseline] = {
            "paired_power_difference": _point_interval(average) if average is not None else None,
            "worst_regression": _worst(power_cases, intervals),
        }
    return {
        "case_count": len(cases), "power_case_count": len(power_cases),
        "case_indices": [case["case_index"] for case in cases],
        "undefined_power_cases": [case["case"]["name"] for case in cases if not case["power_defined"]],
        "power": _average_intervals([case["methods"][candidate]["power"] for case in power_cases]),
        "mean_fdp": _average_intervals([case["methods"][candidate]["mean_fdp"] for case in cases]),
        "max_case_fdp_q95": max((case["methods"][candidate]["fdp_q95"] for case in cases), default=None),
        "max_case_fdp_q99": max((case["methods"][candidate]["fdp_q99"] for case in cases), default=None),
        "comparisons": comparisons,
    }


def analyze_artifact(
    artifact: dict[str, Any], *, candidates: Sequence[str] | None = None,
    baselines: Sequence[str] = DEFAULT_BASELINES, alpha: float = 0.05,
    interval: str = "empirical_bernstein",
) -> dict[str, Any]:
    """Analyze one run; multiple input runs remain separate estimands/families."""
    alpha = _number(alpha, "alpha")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    interval_function = empirical_bernstein_interval
    interval_reference = INTERVAL_REFERENCE
    interval_formula = "sqrt(2*s2*log(4/delta)/n) + 7*(upper-lower)*log(4/delta)/(3*(n-1))"
    if interval == "betting":
        from .robustness_betting import betting_interval, BETTING_REFERENCE
        def interval_function(values, *, alpha, lower=0., upper=1.):
            return betting_interval(values, alpha=alpha, support=(lower, upper))
        interval_reference = BETTING_REFERENCE
        interval_formula = "invert equal fixed-grid mixture mean_f product_i((1-f)+f*X_i/m) at 2/delta; reflect for upper bound"
    elif interval != "empirical_bernstein":
        raise ValueError("Unknown interval method")
    cases = _validate_artifact(artifact)
    baselines = _names(baselines, "baselines")
    if "best_baseline_mean" in baselines or any("oracle" in name.lower() for name in baselines):
        raise ValueError("Strong baselines must be non-oracle and cannot use reserved best_baseline_mean")
    if candidates is None:
        candidates = sorted({method for case in cases for method in case["rows"]
                             if method not in baselines and "oracle" not in method.lower()})
    candidates = _names(candidates, "candidates")
    if any("oracle" in name.lower() for name in candidates):
        raise ValueError("Oracle methods are diagnostic only; select non-oracle candidates")
    methods = list(dict.fromkeys([*candidates, *baselines]))
    for case in cases:
        missing = sorted(set(methods) - case["rows"].keys())
        if missing:
            raise ValueError(f"{case['case']['name']}: missing requested methods: {', '.join(missing)}")
    # One per-input union bound covers FDP, power, and every paired comparison.
    # Derived best-baseline, worst-case, and macro intervals reuse this event.
    family_size = sum(len(methods) * (1 + case["power_defined"])
                      + len(candidates) * len(baselines) * case["power_defined"] for case in cases)
    per_interval_alpha = alpha / family_size
    if per_interval_alpha == 0:
        raise ValueError("alpha is too small for the requested interval family")
    results = []
    for case in cases:
        rows = case["rows"]
        result = {key: value for key, value in case.items() if key != "rows"}
        result["methods"] = {}
        for method in methods:
            row = rows[method]
            fdp = row["fdp_by_repetition"]
            result["methods"][method] = {
                "mean_fdp": interval_function(fdp, alpha=per_interval_alpha),
                "fdp_q95": _quantile(fdp, .95), "fdp_q99": _quantile(fdp, .99),
                "power": interval_function(row["power_by_repetition"], alpha=per_interval_alpha)
                if case["power_defined"] else None,
            }
        result["best_baseline"] = None
        if case["power_defined"]:
            means = {name: math.fsum(rows[name]["power_by_repetition"]) / case["repetitions"]
                     for name in baselines}
            best = max(baselines, key=means.__getitem__)
            result["best_baseline"] = {
                "method": best, "mean_power": means[best],
                "tied_methods": [name for name in baselines if means[name] == means[best]],
                "selection": "largest observed scenario mean; argument order breaks exact ties",
            }
        result["candidates"] = {}
        for candidate in candidates:
            comparisons = {
                name: _point_interval(interval_function(
                    paired_power_differences(rows[candidate]["power_by_repetition"], rows[name]["power_by_repetition"]),
                    lower=-1, upper=1, alpha=per_interval_alpha,
                )) for name in baselines
            } if case["power_defined"] else {}
            result["candidates"][candidate] = {
                "vs_baselines": comparisons,
                "vs_best_baseline_mean": _point_interval(_best_interval(comparisons)) if comparisons else None,
                "vs_observed_best_baseline_paired": comparisons[best] if comparisons else None,
            }
        results.append(result)
    summed = [case["elapsed_seconds"] for case in results]
    return {
        "phase": "DEVELOPMENT_ONLY", "notice": DEVELOPMENT_NOTICE,
        "settings": artifact["settings"], "candidates": candidates, "baselines": baselines,
        "uncertainty": {
            "alpha_per_input_family": alpha, "elementary_interval_count": family_size,
            "alpha_per_elementary_interval": per_interval_alpha,
            "reference": interval_reference, "interval_method": interval,
            "formula": interval_formula,
            "pairing": "same input, scenario, and zero-based repetition array position",
            "unit": "independent common-mean simulation repetition within each case; arbitrary dependence within each repetition",
            "macro": "equal case weights, average simultaneous case bounds; no across-case independence assumed",
            "best_baseline": "max observed baseline mean, never mean of per-repetition maxima; uncertainty uses minimum across all simultaneous paired baseline bounds",
            "scope": "one input and the requested fixed method list; no adjustment across input runs, prior/adaptive development selections, or repeated analyses",
        },
        "resources": {
            "elapsed_wall_seconds": _seconds(artifact.get("elapsed_seconds"), "run"),
            "summed_case_elapsed_seconds": math.fsum(summed) if all(v is not None for v in summed) else None,
            "cases_missing_elapsed_seconds": sum(v is None for v in summed),
            "total_case_repetitions": sum(case["repetitions"] for case in results),
            "per_method_seconds": None,
            "note": "Recorded wall and summed worker-case durations; neither is measured CPU time. Per-method timings are unavailable.",
        },
        "diagnostics": _merge_diagnostics([case["diagnostics"] for case in results]),
        "diagnostic_limitations": (
            "Shared case diagnostics cannot be attributed to individual methods. Counts describe recorded "
            "converged flags, explicit numerical counters, and nonfinite numeric leaves. Numerical counters "
            "may overlap. Null values include intentional limma infinite-prior sentinels. Missing flags are "
            "not successes. Screens do not retain raw p-values or aborted-run failures; zero observed counts "
            "do not establish absence of numerical failures. Invalid FDP/power arrays cause an error, never row deletion."
        ),
        "scenarios": results,
        "macro": {candidate: {
            group: _macro([case for case in results if group == "all" or case["case"]["distribution"] == group],
                          candidate, baselines)
            for group in ("all", "normal", "t5")
        } for candidate in candidates},
    }


def _reference(path: Path, raw: bytes | None = None) -> dict[str, Any]:
    raw = path.read_bytes() if raw is None else raw
    return {"path": path.resolve().as_posix(), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def analyze_files(
    inputs: Sequence[str | Path], *, candidates: Sequence[str] | None = None,
    baselines: Sequence[str] = DEFAULT_BASELINES, alpha: float = 0.05,
    interval: str = "empirical_bernstein",
) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one input is required")
    runs, seen = [], set()
    for source in inputs:
        path = Path(source).resolve()
        raw = path.read_bytes()
        reference = _reference(path, raw)
        if reference["sha256"] in seen:
            raise ValueError("Duplicate input content; each screen must be supplied once")
        seen.add(reference["sha256"])
        artifact = json.loads(raw)
        if not isinstance(artifact, dict):
            raise TypeError(f"{path}: expected a JSON screen object")
        run = analyze_artifact(artifact, candidates=candidates, baselines=baselines, alpha=alpha, interval=interval)
        runs.append({"input_reference": reference, **run})
    module = Path(__file__).resolve()
    script = module.parents[2] / "scripts" / "robustness_analyze.py"
    return {
        "schema_version": 1, "phase": "DEVELOPMENT_ONLY", "notice": DEVELOPMENT_NOTICE,
        "analysis_sources": [_reference(module), *([_reference(script)] if script.is_file() else []),
            *([_reference(module.with_name("robustness_betting.py"))] if interval == "betting" else [])],
        "python_version": platform.python_version(), "runs": runs,
    }


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _interval_text(value: dict[str, Any] | None, scale: float = 1.0) -> str:
    if value is None:
        return "undefined"
    return f"{scale * value['mean']:.4f} [{scale * value['lower']:.4f}, {scale * value['upper']:.4f}]"


def render_markdown(report: dict[str, Any]) -> str:
    """Readable report; JSON carries full precision and the complete case metadata."""
    lines = ["# Robustness development analysis", "", DEVELOPMENT_NOTICE, "",
             ("Power differences use percentage points (pp). FDP uses [0, 1]. "
              "Bracketed ranges are conservative development intervals, not confirmatory CIs."), "",
             ("Interval method and primary reference are recorded per run. Each uses two-sided "
              "bounded-mean intervals with per-input Bonferroni allocation; the method is not selected per result."), "",
             ("Macro means give each case equal weight and average its simultaneous interval endpoints. "
             "Undefined-power cases remain in FDP/diagnostic summaries. normal/t5 groups use the recorded "
             "distribution label, including stress variants; all includes every recorded scenario."), "",
             ("The best baseline maximizes the observed case mean. Its uncertainty takes the minimum of "
             "all simultaneous paired bounds, accounting for baseline selection within this report. "
             "Cases and input runs are never pooled into independent repetitions."), ""]
    for run in report["runs"]:
        ref = run["input_reference"]
        lines.extend([f"## {_cell(run['settings'].get('run_id', 'unnamed run'))}", "",
                      f"Input: `{ref['path']}`", "", f"Input SHA-256: `{ref['sha256']}`", "",
                      f"Baselines: {', '.join(_cell(name) for name in run['baselines'])}.", "",
                      f"Intervals: [{run['uncertainty'].get('interval_method', 'empirical_bernstein')}]({run['uncertainty']['reference']}).", "",
                      (f"Elementary intervals: {run['uncertainty']['elementary_interval_count']}; "
                      f"family alpha: {run['uncertainty']['alpha_per_input_family']}; "
                      f"per-interval alpha: {run['uncertainty']['alpha_per_elementary_interval']:.8g}."), "",
                      run["uncertainty"]["scope"], "",
                      "### Resources and diagnostics", "",
                      (f"Wall seconds: {run['resources']['elapsed_wall_seconds']}; "
                      f"sum of case seconds: {run['resources']['summed_case_elapsed_seconds']}; "
                      f"case-repetitions: {run['resources']['total_case_repetitions']}."), "",
                      run["resources"]["note"], "", run["diagnostic_limitations"], "",
                      "| Case | Reps | Seconds | Diagnostic reps | Convergence checks | Nonconvergence (fit) | Numerical counters | Nonfinite values |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for case in [*run["scenarios"], {"case": {"name": "TOTAL"}, "repetitions": run["resources"]["total_case_repetitions"],
                                       "elapsed_seconds": run["resources"]["summed_case_elapsed_seconds"], "diagnostics": run["diagnostics"]}]:
            d = case["diagnostics"]
            lines.append(f"| {_cell(case['case']['name'])} | {case['repetitions']} | {case['elapsed_seconds']} | "
                         f"{d['repetitions_with_diagnostics']} | {d['convergence_checks']} | "
                         f"{d['nonconvergence_count']} ({d['fit_nonconvergence_count']}) | "
                         f"{d['numerical_counter_total']} | {d['nonfinite_numeric_values']} |")
        lines.extend(["", "Diagnostic paths (counts):", "", "```json",
                      json.dumps(run["diagnostics"], indent=2, sort_keys=True, allow_nan=False), "```", ""])
        for candidate in run["candidates"]:
            lines.extend([f"### Candidate: {_cell(candidate)}", "",
                          "| Macro group | Power cases / all | Comparator | Paired difference pp [interval] | Worst case | Worst difference pp |",
                          "| --- | ---: | --- | ---: | --- | ---: |"])
            for group, macro in run["macro"][candidate].items():
                for baseline, comparison in macro["comparisons"].items():
                    worst = comparison["worst_regression"]
                    lines.append(f"| {group} | {macro['power_case_count']} / {macro['case_count']} | {_cell(baseline)} | "
                                 f"{_interval_text(comparison['paired_power_difference'], 100)} | "
                                 f"{_cell(worst['case']) if worst else 'undefined'} | "
                                 f"{format(worst['mean_difference_pp'], '.4f') if worst else 'undefined'} |")
            lines.extend(["", "| Case | Comparator | Baseline mean power | Paired difference pp [interval] |",
                          "| --- | --- | ---: | ---: |"])
            for case in run["scenarios"]:
                entry = case["candidates"][candidate]
                for baseline in [*run["baselines"], "best_baseline_mean"]:
                    if baseline == "best_baseline_mean":
                        best = case["best_baseline"]
                        label = f"best: {best['method']}" if best else "best: undefined"
                        value, mean = entry["vs_best_baseline_mean"], best["mean_power"] if best else None
                    else:
                        label, value = baseline, entry["vs_baselines"].get(baseline)
                        power = case["methods"][baseline]["power"]
                        mean = power["mean"] if power else None
                    lines.append(f"| {_cell(case['case']['name'])} | {_cell(label)} | "
                                 f"{format(mean, '.6f') if mean is not None else 'undefined'} | {_interval_text(value, 100)} |")
            lines.append("")
        lines.extend(["### FDP and power by method and scenario", "",
                      "| Case | Method | Mean FDP [interval] | FDP q95 | FDP q99 | Mean power [interval] |",
                      "| --- | --- | ---: | ---: | ---: | ---: |"])
        for case in run["scenarios"]:
            for method, row in case["methods"].items():
                lines.append(f"| {_cell(case['case']['name'])} | {_cell(method)} | {_interval_text(row['mean_fdp'])} | "
                             f"{row['fdp_q95']:.6f} | {row['fdp_q99']:.6f} | {_interval_text(row['power'])} |")
        lines.extend(["", "### Recorded experiment source hashes", "",
                      "These are the exact hashes recorded in the input settings, not hashes of the current experiment checkout.", "",
                      "| Source | Recorded SHA-256 |", "| --- | --- |"])
        for source, digest in sorted(run["settings"]["source_sha256"].items()):
            lines.append(f"| {_cell(source)} | `{digest}` |")
        lines.append("")
    lines.extend(["## Analysis source hashes", "", "| Source | SHA-256 |", "| --- | --- |"])
    for ref in report["analysis_sources"]:
        lines.append(f"| {_cell(ref['path'])} | `{ref['sha256']}` |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="Completed DEVELOPMENT screen JSON; repeat to report runs separately")
    parser.add_argument("--output", type=Path, required=True,
                        help="New output prefix (or .json/.md path); writes both .json and .md")
    parser.add_argument("--candidate", action="append",
                        help="Repeat for each candidate; default: all recorded non-oracle, non-baseline methods")
    parser.add_argument("--baseline", action="append",
                        help="Repeat to replace the default strong baseline list: " + ", ".join(DEFAULT_BASELINES))
    parser.add_argument("--alpha", type=float, default=.05, help="Development family alpha per input (default .05)")
    parser.add_argument("--interval", choices=["empirical_bernstein", "betting"], default="empirical_bernstein")
    args = parser.parse_args(argv)
    prefix = args.output.with_suffix("") if args.output.suffix.lower() in (".json", ".md") else args.output
    outputs = [Path(str(prefix) + suffix) for suffix in (".json", ".md")]
    try:
        for path in outputs:
            if path.exists() or path.resolve() in {source.resolve() for source in args.input}:
                raise ValueError(f"Output already exists or aliases an input: {path}")
        report = analyze_files(args.input, candidates=args.candidate,
                               baselines=args.baseline if args.baseline is not None else DEFAULT_BASELINES,
                               alpha=args.alpha, interval=args.interval)
        contents = [json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
                    render_markdown(report)]
        for path, content in zip(outputs, contents, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print("DEVELOPMENT ONLY: " + ", ".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
