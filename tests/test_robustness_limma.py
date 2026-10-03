# SPDX-License-Identifier: GPL-2.0-or-later
"""Numerical regression against retained output from the authentic author R.

These tests do not require R. Regenerate the independently computed reference
batch with scripts/robustness_limma_parity.py; its report records provenance.
"""

import csv
import hashlib
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t

from sca3_compass.robustness_limma import squeeze_var

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "external" / "limma-reference"
REPORT = json.loads((REFERENCE / "parity" / "report.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _reference_data():
    inputs, outputs = defaultdict(list), defaultdict(list)
    with (REFERENCE / "parity" / "inputs.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            inputs[row["case"]].append(row)
    with (REFERENCE / "parity" / "author-r.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            outputs[row["case"], int(row["robust"])].append(row)
    return inputs, outputs


@pytest.mark.parametrize("name,robust", [(entry["case"], entry["robust"]) for entry in REPORT["results"]])
def test_independent_author_r_reference(name, robust):
    inputs, outputs = _reference_data()
    rows = inputs[name]
    x = np.array([float(row["variance"]) for row in rows])
    df = float(rows[0]["df"])
    tails = float(rows[0]["lower"]), float(rows[0]["upper"])
    posterior, prior_df, diagnostics = squeeze_var(x, df=df, robust=robust, winsor_tail_p=tails)
    reference = outputs[name, int(robust)]
    assert len(reference) == len(x)
    assert [int(row["index"]) for row in reference] == list(range(len(x)))
    total = np.minimum(df + prior_df, len(x) * df)
    statistic = np.array([float(row["mean"]) for row in rows]) / np.sqrt(0.8 * posterior)
    values = {"posterior": posterior, "prior_df": prior_df,
              "prior_variance": diagnostics["prior_variance"],
              "global_prior_df": diagnostics["global_prior_df"], "total_df": total,
              "t_one_sided": t.sf(statistic, total), "t_two_sided": 2*t.cdf(-np.abs(statistic), total)}
    tolerances = REPORT["tolerances"]["true" if robust else "false"]
    for key, actual in values.items():
        expected = np.array([float(row[key]) for row in reference])
        np.testing.assert_allclose(actual, expected, **tolerances, err_msg=f"{name}: {key}")
    assert isinstance(prior_df, np.ndarray) if robust else np.isscalar(prior_df)
    assert diagnostics["pooled_residual_df"] == len(x)*df


def test_reference_provenance_and_integrity():
    assert REPORT["author_R_executed"] is True
    assert REPORT["passed"] is True
    assert "statmod: 1.5.2" in REPORT["R_stdout"]
    assert REPORT["comparisons"] == len(REPORT["results"])
    manifest = json.loads((REFERENCE / "source-manifest.json").read_text(encoding="utf-8"))
    assert manifest["limma_release"] == "RELEASE_3_22"
    for entry in manifest["sources"]:
        content = (REFERENCE / entry["path"]).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(content).hexdigest() == entry["sha256_lf"]
    for filename, key in (("inputs.csv", "input_sha256"), ("author-r.csv", "author_output_sha256")):
        assert hashlib.sha256((REFERENCE / "parity" / filename).read_bytes()).hexdigest() == REPORT[key]


def test_author_cases_exercise_every_robust_fit_branch():
    branches = {entry["branch"] for entry in REPORT["results"] if entry["robust"]}
    assert branches == {"finite_prior", "infinite_prior", "nonrobust_lower_bound",
                        "nonrobust_infinite_fallback", "no_winsorization"}


@pytest.mark.parametrize("robust", [False, True])
def test_scale_and_permutation_equivariance_and_no_input_mutation(robust):
    rng = np.random.default_rng(28)
    x = rng.f(20, 5, 300)
    x[0] = 1000
    original = x.copy()
    x.flags.writeable = False
    post, prior, info = squeeze_var(x, robust=robust)
    permutation = rng.permutation(len(x))
    permuted_post, permuted_prior, _ = squeeze_var(x[permutation], robust=robust)
    scaled_post, scaled_prior, scaled_info = squeeze_var(x * 17, robust=robust)
    np.testing.assert_array_equal(x, original)
    np.testing.assert_allclose(permuted_post, post[permutation], rtol=1e-11)
    np.testing.assert_allclose(permuted_prior, prior[permutation] if robust else prior, rtol=1e-11)
    np.testing.assert_allclose(scaled_post, post * 17, rtol=1e-11)
    np.testing.assert_allclose(scaled_prior, prior, rtol=1e-11)
    assert scaled_info["prior_variance"] == pytest.approx(17 * info["prior_variance"], rel=1e-11)


def test_standard_infinite_df_uses_pooled_arithmetic_variance():
    x = np.array([0.8, 0.9, 1.1, 1.25, 1.0, 0.95])
    posterior, prior, info = squeeze_var(x)
    assert np.isinf(prior)
    assert info["prior_variance"] == pytest.approx(np.mean(x), rel=1e-14)
    np.testing.assert_allclose(posterior, np.mean(x), rtol=1e-14)


def test_robust_infinite_global_df_can_have_gene_specific_finite_df():
    x = np.r_[np.ones(999), 1e9]
    posterior, prior, info = squeeze_var(x, robust=True)
    assert np.isinf(info["global_prior_df"])
    assert np.isinf(prior).any()
    assert np.isfinite(prior[-1])
    assert prior[-1] == 0
    assert posterior[-1] == x[-1]
    assert info["outlier_adjusted_count"] > 0
    assert np.isfinite(posterior).all()


def test_robust_prior_df_monotonicity_for_hypervariable_genes():
    x = np.random.default_rng(924).f(20, 5, 1000)
    x[:5] *= 1000
    _, prior, info = squeeze_var(x, robust=True)
    assert np.all(np.diff(prior[np.argsort(x)]) <= 1e-10)
    assert prior[np.argmax(x)] < 0.1 * info["global_prior_df"]


def test_zero_tails_matches_standard_without_changing_api_shape():
    x = np.random.default_rng(4791).f(20, 5, 200)
    regular_post, regular_prior, _ = squeeze_var(x)
    robust_post, robust_prior, info = squeeze_var(x, robust=True, winsor_tail_p=(0, 0))
    np.testing.assert_array_equal(regular_post, robust_post)
    np.testing.assert_array_equal(robust_prior, np.full(len(x), regular_prior))
    assert info["branch"] == "no_winsorization"


def test_conditional_student_is_moderated_t_under_the_parameter_mapping():
    # This identity concerns a matched prior; it does not assert that fitting
    # the prior from calibration and fitting it from all genes give equal estimates.
    d = 20
    q = np.random.default_rng(618).f(d, 5, 600) * d
    posterior, nu, info = squeeze_var(q/d, df=d)
    scatter = info["prior_variance"]
    means = np.linspace(-3, 3, len(q))
    conditional_scale = 0.8 * (nu * scatter + q) / (nu + d)
    np.testing.assert_allclose(0.8 * posterior, conditional_scale, rtol=1e-14)
    np.testing.assert_allclose(t.sf(means/np.sqrt(0.8*posterior), nu+d),
                               t.sf(means/np.sqrt(conditional_scale), nu+d), rtol=1e-14)


@pytest.mark.parametrize("x", [[], [1, 2, 3], [1, 2, 3, 0], [1, 2, 3, -1],
                               [1, 2, 3, np.nan], [1, 2, 3, np.inf], [[1, 2], [3, 4]]])
@pytest.mark.parametrize("robust", [False, True])
def test_reject_incomplete_or_unsupported_variances(x, robust):
    with pytest.raises(ValueError, match="full vector"):
        squeeze_var(x, robust=robust)


@pytest.mark.parametrize("df", [0, -1, np.inf, np.nan, 1e-8, [20], [20, 20, 20, 20]])
def test_reject_non_scalar_or_invalid_df(df):
    with pytest.raises(ValueError, match="scalar residual df"):
        squeeze_var(np.ones(4), df=df)


@pytest.mark.parametrize("tails", [(0, 0.1), (0.05, 0), (-0.1, 0.1), (0.1, 0.5),
                                   (0.1, np.nan), (0.1,), (0.1, 0.2, 0.3)])
def test_reject_unsupported_winsor_tails(tails):
    with pytest.raises(ValueError, match="Winsor tail"):
        squeeze_var(np.ones(10), robust=True, winsor_tail_p=tails)


def test_robust_requires_boolean():
    with pytest.raises(TypeError, match="boolean"):
        squeeze_var(np.ones(10), robust="False")
