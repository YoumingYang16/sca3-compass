import itertools

import numpy as np
import pytest
from scipy.stats import false_discovery_control, norm

from sca3_compass.molecular_data import (
    assert_expression_access,
    geo_base,
    parse_matrix_metadata,
)
from sca3_compass.molecular_methods import (
    GaussianMaxT,
    covariance_root,
    ebh,
    equicorrelation,
    evaluate_methods,
    fdr_adjust,
    partial_conjunction,
)


def test_by_and_bh_agree_with_independent_scipy_reference():
    p = np.random.default_rng(4).uniform(size=70)
    for method in ("BY", "BH"):
        np.testing.assert_allclose(fdr_adjust(p, method), false_discovery_control(p, method=method.lower()))


def test_pc_matches_exhaustive_intersection_union():
    p = np.random.default_rng(2).uniform(size=(50, 5))
    for r in range(1, 6):
        subsets = list(itertools.combinations(range(5), 5 - r + 1))
        reference = np.max([np.minimum(1, (6 - r) * p[:, subset].min(axis=1)) for subset in subsets], axis=0)
        np.testing.assert_allclose(partial_conjunction(p, r), reference)


def test_one_strong_study_is_not_replication():
    assert partial_conjunction(np.array([1e-30, 0.4, 0.5]), 2) == 0.8


def test_pc_requires_valid_r_and_values():
    for r in (0, 4, 1.5):
        with pytest.raises(ValueError):
            partial_conjunction(np.array([0.1, 0.2, 0.3]), r)
    for p in (np.array([np.nan]), np.array([-0.01]), np.array([1.1])):
        with pytest.raises(ValueError):
            fdr_adjust(p)


def test_maxt_monotonic_nonzero_and_duplicate_invariance():
    calibration = GaussianMaxT(np.ones((3, 3)), 100_000, 22)
    points = np.repeat(np.array([-1, 0, 1, 2, 20])[:, None], 3, axis=1)
    p = calibration.pvalues(points)
    assert np.all(np.diff(p) <= 0) and p[-1] == calibration.minimum_p
    np.testing.assert_allclose(p[:-1], norm.sf(points[:-1, 0]), atol=0.006)


def test_mc_gaussian_null_calibration():
    rho = 0.7
    rng = np.random.default_rng(37)
    cov = equicorrelation(4, rho)
    calibration = GaussianMaxT(cov, 100_000, 91)
    z = rng.normal(size=(30_000, 4)) @ covariance_root(cov).T
    assert np.mean(calibration.pvalues(z) <= 0.05) <= 0.055


def test_covariance_validation():
    with pytest.raises(ValueError):
        covariance_root(np.array([[1, 2], [2, 1]]))


def test_ebh_step_up():
    np.testing.assert_array_equal(ebh(np.array([1000, 30, 0, 1]), 0.1), [True, True, False, False])


def test_directional_hypothesis_shape():
    z = np.random.default_rng(18).normal(size=(20, 3, 4))
    results = evaluate_methods(z, 2, 0.05, GaussianMaxT(np.eye(4), 1000, 2))
    assert all(r.shape == (20, 2) and r.dtype == bool for r in results.values())


def test_holdout_is_not_accessible_to_expression_parser():
    with pytest.raises(PermissionError):
        assert_expression_access({"role": "sealed_candidate"})


def test_geo_url_rejects_traversal():
    with pytest.raises(ValueError):
        geo_base("../../x")
    assert geo_base("GSE93713").endswith("GSE93nnn/GSE93713")


def test_matrix_parser_never_consumes_expression(tmp_path):
    import gzip
    path = tmp_path / "metadata.gz"
    with gzip.open(path, "wt") as handle:
        handle.write('!Sample_geo_accession\t"GSM1"\t"GSM2"\n!Sample_title\t"a"\t"b"\n!series_matrix_table_begin\ninvalid expression intentionally unread\n')
    assert parse_matrix_metadata(path)["sample_count"] == 2


def test_mc_uncertainty_is_not_zero_after_no_errors():
    from sca3_compass.molecular_benchmark import summarize
    report = summarize([0.0] * 400)
    assert report["ci95"][0] == 0
    assert 0 < report["ci95"][1] < 0.05
    assert summarize([1.0] * 400)["ci95"][1] == 1
