"""Bounded-uncertainty, pairing, provenance, and CLI regression tests."""
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from sca3_compass.robustness_analysis import (
    DEFAULT_BASELINES,
    analyze_artifact,
    analyze_files,
    diagnostic_counts,
    empirical_bernstein_interval,
    paired_power_differences,
    render_markdown,
)


def screen(power=None, *, name="normal_case", distribution="normal", defined=True):
    power = power or {"candidate": [.2, .8, .4, .6], "base_a": [.1, .7, .3, .5], "base_b": [.8, .0, .4, .0]}
    n = len(next(iter(power.values())))
    case = {"name": name, "distribution": distribution, "n": 16, "rho": .8, "effect": 3.5}
    return {
        "settings": {"phase": "DEVELOPMENT_ONLY", "independent_confirmation": False,
                     "run_id": "TEST", "seed": 123, "source_sha256": {"source.py": "a" * 64}},
        "elapsed_seconds": 2.5,
        "scenarios": [{"case": case, "repetitions": n, "power_defined": defined,
                       "replicated_signed_truths": 2 if defined else 0, "elapsed_seconds": 4.0,
                       "rows": [{"method": method, "power_by_repetition": values,
                                 "fdp_by_repetition": [i / n for i in range(n)],
                                 "power": {"mean": 999}, "fdp_q95": 999, "fdp_q99": 999}
                                for method, values in power.items()],
                       "diagnostics": [{"fit": {"converged": True},
                                        "shape": [{"converged": True}, {"converged": True}],
                                        "limma": [{"global_prior_df": None}],
                                        "radial_grid": {"not_run": True}} for _ in range(n)]}],
    }


def analyze(artifact, **kwargs):
    return analyze_artifact(artifact, candidates=["candidate"], baselines=["base_a", "base_b"], **kwargs)


def test_empirical_bernstein_formula_and_unbiased_variance():
    data = [-.2, .4, .1, -.1] * 250
    result = empirical_bernstein_interval(data, lower=-1, upper=1, alpha=.02)
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / (len(data) - 1)
    radius = math.sqrt(2 * variance * math.log(4 / .02) / len(data)) + 14 * math.log(4 / .02) / (3 * (len(data) - 1))
    assert result["sample_variance"] == pytest.approx(variance)
    assert result["radius"] == pytest.approx(radius)
    assert result["lower"] == pytest.approx(mean - radius)
    assert result["upper"] == pytest.approx(mean + radius)


def test_optional_betting_interval_preserves_pairing_and_family(tmp_path):
    from sca3_compass.robustness_betting import betting_interval
    artifact=screen()
    old=analyze(artifact)
    result=analyze(artifact,interval='betting')
    delta=result['uncertainty']['alpha_per_elementary_interval']
    assert delta==old['uncertainty']['alpha_per_elementary_interval']
    assert result['uncertainty']['interval_method']=='betting'
    paired=[a-b for a,b in zip(artifact['scenarios'][0]['rows'][0]['power_by_repetition'],
        artifact['scenarios'][0]['rows'][1]['power_by_repetition'])]
    expected=betting_interval(paired,delta,(-1,1))
    actual=result['scenarios'][0]['candidates']['candidate']['vs_baselines']['base_a']
    for key in ['mean','lower','upper']:
        assert actual[key]==expected[key]
    source=tmp_path/'screen.json';source.write_text(json.dumps(artifact),encoding='utf-8')
    report=analyze_files([source],candidates=['candidate'],baselines=['base_a'],interval='betting')
    assert any(Path(ref['path']).name=='robustness_betting.py' for ref in report['analysis_sources'])
    assert report['phase']=='DEVELOPMENT_ONLY'
    with pytest.raises(ValueError,match='Unknown interval'):
        analyze(artifact,interval='choose_best_after_results')


@pytest.mark.parametrize("value", [0.0, 1.0])
def test_constant_bounded_data_keeps_nonzero_uncertainty(value):
    interval = empirical_bernstein_interval([value] * 1000)
    assert 0 <= interval["lower"] <= value <= interval["upper"] <= 1
    assert interval["upper"] > interval["lower"]
    assert interval["sample_variance"] == 0
    assert empirical_bernstein_interval([value] * 1000, alpha=.001)["radius"] > interval["radius"]


def test_single_rep_uses_full_support_and_two_reps_are_clipped():
    one = empirical_bernstein_interval([.2], lower=-1, upper=1)
    assert (one["lower"], one["upper"], one["sample_variance"]) == (-1, 1, None)
    two = empirical_bernstein_interval([-1, 1], lower=-1, upper=1)
    assert (two["lower"], two["upper"]) == (-1, 1)


@pytest.mark.parametrize("data", [[], [float("nan")], [float("inf")], [-.001], [1.001], [None], [True]])
def test_invalid_bounded_data_is_rejected(data):
    with pytest.raises((TypeError, ValueError)):
        empirical_bernstein_interval(data)


@pytest.mark.parametrize("kwargs", [{"alpha": 0}, {"alpha": 1}, {"alpha": float("nan")},
                                   {"lower": 1, "upper": 1}, {"upper": float("inf")}])
def test_invalid_interval_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        empirical_bernstein_interval([.5], **kwargs)


def test_pairing_preserves_repetition_position_and_covariance():
    candidate, baseline = [.1, .7, .3, .9], [0, .6, .2, .8]
    paired = paired_power_differences(candidate, baseline)
    assert paired == pytest.approx([.1] * 4)
    assert empirical_bernstein_interval(paired, lower=-1, upper=1)["sample_variance"] < 1e-30
    wrong = paired_power_differences(candidate, list(reversed(baseline)))
    assert empirical_bernstein_interval(wrong, lower=-1, upper=1)["sample_variance"] > .1
    with pytest.raises(ValueError, match="identical repetition counts"):
        paired_power_differences(candidate, baseline[:-1])


def test_method_row_order_is_irrelevant_and_stored_summaries_are_not_trusted():
    artifact = screen()
    expected = analyze(artifact)
    artifact["scenarios"][0]["rows"].reverse()
    assert analyze(artifact) == expected
    case = expected["scenarios"][0]
    paired = case["candidates"]["candidate"]["vs_baselines"]["base_a"]
    assert paired["mean_difference"] == pytest.approx(.1)
    assert paired["sample_variance"] < 1e-30
    assert case["methods"]["candidate"]["fdp_q95"] == pytest.approx(.7125)
    assert case["methods"]["candidate"]["fdp_q99"] == pytest.approx(.7425)


def test_best_baseline_is_best_mean_not_repetitionwise_maximum():
    artifact = screen({"candidate": [.6] * 100, "base_a": [1, 0] * 50, "base_b": [0, 1] * 50})
    case = analyze(artifact)["scenarios"][0]
    assert case["best_baseline"]["mean_power"] == .5
    assert case["best_baseline"]["method"] == "base_a"
    assert case["best_baseline"]["tied_methods"] == ["base_a", "base_b"]
    entry = case["candidates"]["candidate"]
    assert entry["vs_best_baseline_mean"]["mean_difference"] == pytest.approx(.1)
    assert entry["vs_best_baseline_mean"]["lower"] == min(v["lower"] for v in entry["vs_baselines"].values())


def test_macro_uses_equal_case_weights_and_simultaneous_case_bounds():
    artifact = screen({"candidate": [.9] * 1000, "base_a": [.4] * 1000, "base_b": [.3] * 1000})
    second = screen({"candidate": [.1] * 10, "base_a": [.5] * 10, "base_b": [.6] * 10},
                    name="t5_case", distribution="t5")
    artifact["scenarios"] += second["scenarios"]
    report = analyze(artifact)
    # 2 cases * (3 methods * 2 metrics + 1 candidate * 2 baselines).
    assert report["uncertainty"]["elementary_interval_count"] == 16
    assert report["uncertainty"]["alpha_per_elementary_interval"] == .05 / 16
    macro = report["macro"]["candidate"]
    entry = macro["all"]["comparisons"]["base_a"]
    assert entry["paired_power_difference"]["mean_difference"] == pytest.approx(.05)
    intervals = [case["candidates"]["candidate"]["vs_baselines"]["base_a"] for case in report["scenarios"]]
    assert entry["paired_power_difference"]["lower"] == pytest.approx(sum(v["lower"] for v in intervals) / 2)
    assert entry["paired_power_difference"]["upper"] == pytest.approx(sum(v["upper"] for v in intervals) / 2)
    assert entry["worst_regression"]["case"] == "t5_case"
    assert entry["worst_regression"]["regression_pp"] == pytest.approx(40)
    assert macro["all"]["comparisons"]["best_baseline_mean"]["worst_regression"]["regression_pp"] == pytest.approx(50)
    assert macro["normal"]["power_case_count"] == macro["t5"]["power_case_count"] == 1
    assert report["resources"]["summed_case_elapsed_seconds"] == 8


def test_undefined_power_is_retained_for_fdp_and_diagnostics():
    report = analyze(screen(defined=False, name="null", distribution="t5"))
    case = report["scenarios"][0]
    assert case["best_baseline"] is None
    assert case["methods"]["candidate"]["power"] is None
    assert case["methods"]["candidate"]["mean_fdp"]["n"] == 4
    macro = report["macro"]["candidate"]["t5"]
    assert macro["undefined_power_cases"] == ["null"]
    assert macro["power_case_count"] == 0
    assert macro["comparisons"]["base_a"]["paired_power_difference"] is None
    assert report["macro"]["candidate"]["normal"]["case_count"] == 0
    assert report["diagnostics"]["repetitions_with_diagnostics"] == 4


@pytest.mark.parametrize("mutation,match", [
    (lambda a: a["scenarios"][0]["rows"].pop(), "missing requested methods"),
    (lambda a: a["scenarios"][0]["rows"].append(copy.deepcopy(a["scenarios"][0]["rows"][0])), "unique"),
    (lambda a: a["scenarios"][0]["rows"][0]["power_by_repetition"].pop(), "repetition count mismatch"),
    (lambda a: a["scenarios"][0]["rows"][0]["fdp_by_repetition"].__setitem__(0, float("nan")), "finite"),
    (lambda a: a["scenarios"][0].update(repetition_ids=[0, 1, 2, 3]), "repetition_ids"),
    (lambda a: a["settings"].update(phase="CONFIRMATORY"), "DEVELOPMENT_ONLY"),
    (lambda a: a["settings"].update(source_sha256={"x": "bad"}), "exact source hashes"),
    (lambda a: a["scenarios"][0]["diagnostics"].pop(), "diagnostics"),
    (lambda a: a["settings"].update(cases=[]), "Scenario order/metadata"),
])
def test_bad_artifacts_fail_without_silent_drops(mutation, match):
    artifact = screen()
    mutation(artifact)
    with pytest.raises(ValueError, match=match):
        analyze(artifact)


def test_defaults_include_all_five_strong_baselines_and_exclude_oracles():
    power = {"candidate": [.5, .6], **{name: [.4, .5] for name in DEFAULT_BASELINES},
             "oracle_marginal_mean": [.9, .9]}
    report = analyze_artifact(screen(power))
    assert report["baselines"] == list(DEFAULT_BASELINES)
    assert report["candidates"] == ["candidate"]
    with pytest.raises(ValueError, match="non-oracle"):
        analyze_artifact(screen(power), candidates=["candidate"], baselines=["oracle_marginal_mean"])


def test_nested_diagnostic_counts_and_nulls_are_not_failures():
    result = diagnostic_counts([
        {"fit": {"converged": False}, "shape": [{"converged": False}, {"converged": True}],
         "numerical_failure_count": 2, "matrix": [[float("nan"), float("inf")]],
         "limma": [{"global_prior_df": None}], "grid": {"not_run": True}},
        {"fit": {"converged": True}, "nested": {"solver": {"converged": False}},
         "numerical_failure_count": 0},
    ])
    assert result["nonconvergence_count"] == 3
    assert result["fit_nonconvergence_count"] == 1
    assert result["repetitions_with_nonconvergence"] == 2
    assert result["numerical_counter_total"] == 2
    assert result["nonfinite_numeric_values"] == 2
    assert result["repetitions_with_numerical_diagnostics"] == 1
    assert result["null_values_by_path"] == {"limma[].global_prior_df": 1}
    assert result["not_run_by_path"] == {"grid.not_run": 1}
    clean = diagnostic_counts([{"limma": {"global_prior_df": None}}])
    assert clean["numerical_counter_total"] == clean["nonfinite_numeric_values"] == 0
    assert clean["convergence_checks"] == 0


def test_exact_input_and_source_hashes_and_deterministic_reports(tmp_path):
    source = tmp_path / "screen.json"
    raw = json.dumps(screen()).encode()
    source.write_bytes(raw)
    report = analyze_files([source], candidates=["candidate"], baselines=["base_a", "base_b"])
    assert report["runs"][0]["input_reference"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["runs"][0]["settings"]["source_sha256"] == {"source.py": "a" * 64}
    for reference in report["analysis_sources"]:
        assert reference["sha256"] == hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest()
    assert analyze_files([source], candidates=["candidate"], baselines=["base_a", "base_b"]) == report
    markdown = render_markdown(report)
    assert "DEVELOPMENT ONLY" in markdown
    assert report["runs"][0]["input_reference"]["sha256"] in markdown
    assert "a" * 64 in markdown
    with pytest.raises(ValueError, match="Duplicate input"):
        analyze_files([source, source], candidates=["candidate"], baselines=["base_a", "base_b"])


def test_cli_repeated_selection_json_markdown_and_no_overwrite(tmp_path):
    source, prefix = tmp_path / "input.json", tmp_path / "report"
    artifact = screen()
    source.write_text(json.dumps(artifact), encoding="utf-8")
    second = tmp_path / "input2.json"
    artifact["settings"]["run_id"] = "SECOND"
    second.write_text(json.dumps(artifact), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "robustness_analyze.py"
    command = [sys.executable, "-B", str(script), "--input", str(source), "--input", str(second),
               "--output", str(prefix) + ".json", "--candidate", "candidate", "--candidate", "base_a",
               "--baseline", "base_a", "--baseline", "base_b"]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    raw = prefix.with_suffix(".json").read_bytes()
    report = json.loads(raw)
    assert len(report["runs"]) == 2
    assert report["runs"][0]["candidates"] == ["candidate", "base_a"]
    assert report["runs"][0]["baselines"] == ["base_a", "base_b"]
    assert "not confirmatory" in prefix.with_suffix(".md").read_text(encoding="utf-8")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["input.json", "input2.json", "report.json", "report.md"]
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0
    assert "already exists" in again.stderr
    assert prefix.with_suffix(".json").read_bytes() == raw
