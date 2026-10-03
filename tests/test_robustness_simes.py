"""Bounded deterministic audit checks; no research simulation or acceptance rule."""
import json
from copy import deepcopy
from itertools import combinations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm, t
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_loading as loading
from sca3_compass import robustness_methods as methods
from sca3_compass import robustness_prior as prior
from sca3_compass.robustness_projection import projection_pc
from sca3_compass.robustness_simes import (
    SIMES_POLICY,
    simes_eligibility,
    simes_intersection,
    simes_pc,
)


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def _negative_shape(studies=4):
    shape = np.eye(studies)
    shape[1, 2] = shape[2, 1] = -.5
    return shape


def _legacy_pc(p, shape=None, *, diagnostics=None):
    value = (np.sort(p, axis=-1)[..., 1:] * [3., 1.5, 1.]).min(axis=-1)
    return np.where(value <= .5, value, 1.)


def _explicit_pc(p, shape, r=2):
    """Independent intersection enumeration; keep original study identities."""
    result = np.zeros(p.shape[:-1])
    count = p.shape[-1] - r + 1
    for indices in combinations(range(p.shape[-1]), count):
        sub = shape[np.ix_(indices, indices)]
        values = p[..., list(indices)]
        if np.any(sub[~np.eye(count, dtype=bool)] < 0):
            value = np.minimum(1., count * values.min(axis=-1))
        else:
            value = (np.sort(values, axis=-1) * count / np.arange(1, count + 1)).min(axis=-1)
            value = np.where(value <= .5, value, 1.)
        result = np.maximum(result, value)
    return result


@pytest.mark.parametrize("alpha,expected", [
    (.05, .05002561933876686), (.25, .25652759145643766), (.5, .5463915593447809),
])
def test_known_negative_spd_counterexample_and_bonferroni_bound(alpha, expected):
    rho = -.5

    def copula(a, b):
        return quad(lambda x: norm.pdf(x) * norm.cdf((norm.ppf(b) - rho*x)/np.sqrt(1-rho*rho)),
                    -np.inf, norm.ppf(a), epsabs=1e-13, epsrel=1e-12)[0]

    rejection = alpha + copula(alpha, alpha) - 2*copula(alpha/2, alpha)
    assert rejection == pytest.approx(expected, abs=2e-13)
    assert rejection > alpha
    # Bonferroni's union of two marginal events has this exact probability.
    assert alpha - copula(alpha/2, alpha/2) <= alpha
    info = {}
    shape = np.array([[1., rho], [rho, 1.]])
    p = np.array([[.2, .2], [.3, .4], [.01, .9]])
    np.testing.assert_array_equal(simes_intersection(p, shape, diagnostics=info),
                                  np.minimum(1, 2*p.min(axis=-1)))
    assert info["method"] == "bonferroni" and not info["halflevel_extension"]
    assert info["minimum_retained_correlation"] == -.5


@pytest.mark.parametrize("studies", range(1, 7))
def test_arbitrary_subsets_pc_is_max_and_preserves_study_identity(studies):
    p = np.mod(np.arange(7*2*studies).reshape(7, 2, studies)*.137, 1.)
    shape = np.eye(studies)
    if studies >= 2:
        shape[0, 1] = shape[1, 0] = -.5
    if studies >= 4:
        shape[-2, -1] = shape[-1, -2] = .3
    for r in range(1, studies+1):
        info = {}
        result = simes_pc(p, shape, r, diagnostics=info)
        np.testing.assert_allclose(result, _explicit_pc(p, shape, r), rtol=2e-15)
        permutation = np.arange(studies)[::-1]
        np.testing.assert_allclose(result, simes_pc(p[..., permutation], shape[np.ix_(permutation, permutation)], r),
                                   rtol=2e-15)
        assert info["intersection_count"] == len(list(combinations(range(studies), studies-r+1)))
        assert not info["uses_pvalues_for_selection"]
        json.dumps(info, allow_nan=False)
    # A singleton retained intersection is eligible even when other pairs are negative.
    assert info["fallback_intersection_count"] == 0


def test_halflevel_extension_precedes_weighting_without_weakening_bonferroni():
    p = np.array([[.25, .25], [.3, .3], [.5, 1.], [np.nextafter(.5, 1), 1.]])
    value = simes_intersection(p, np.eye(2))
    np.testing.assert_array_equal(value, [.25, .3, 1., 1.])
    # An ordinary Simes value above .5 must become 1 before division by W.
    pc = simes_pc(np.array([[.1, .2, .4, .9]]), np.eye(4))
    np.testing.assert_array_equal(pc / 2., [.5])
    bonf = simes_intersection(np.array([[.3, .4]]), np.array([[1., -.5], [-.5, 1.]]))
    np.testing.assert_array_equal(bonf, [.6])
    assert simes_intersection(np.array([.5]), np.eye(1)) == .5
    assert simes_intersection(np.array([np.nextafter(.5, 1)]), np.eye(1)) == 1.


@pytest.mark.parametrize("negative", [False, True])
def test_normalization_rescaling_and_shape_only_decisions(negative):
    shape = _negative_shape() if negative else .7*np.eye(4)+.3
    scale = np.array([.25, 2., 3., .5])
    scaled = shape * scale[:, None] * scale[None, :]
    saved = scaled.copy()
    scaled.flags.writeable = False
    p = np.array([[.02, .1, .3, .4], [.3, .21, .29, .6]])
    a, b = {}, {}
    np.testing.assert_array_equal(simes_pc(p, shape, diagnostics=a), simes_pc(p, scaled, diagnostics=b))
    assert a == b
    changed = {}
    simes_pc(1-p, scaled, diagnostics=changed)
    assert changed == b
    np.testing.assert_array_equal(scaled, saved)
    tiny = np.eye(2)
    tiny[0, 1] = tiny[1, 0] = -np.nextafter(0., 1.)
    assert not simes_eligibility(tiny)["simes_eligible"]
    tiny *= 4
    assert not simes_eligibility(tiny)["simes_eligible"]


@pytest.mark.parametrize("shape", [np.zeros((2, 2)), [[1, 2], [2, 1]], [[1, 1], [1, 1]],
                                        [[1, .2], [0, 1]], [[1, np.nan], [np.nan, 1]],
                                        [[np.inf, 0], [0, 1]], [[-1, 0], [0, -1]]])
def test_invalid_shape_never_silently_runs_simes(shape):
    with pytest.raises(ValueError, match="shape"):
        simes_intersection([.1, .2], shape)


@pytest.mark.parametrize("p", [[], [np.nan], [-.1], [1.1], [.1, .2]])
def test_invalid_or_mismatched_probabilities_raise(p):
    with pytest.raises(ValueError):
        simes_intersection(p, np.eye(1))


def test_support_simes_negative_pc_embedding_and_retained_only_diagnostics():
    means = np.array([[40., norm.isf(.2), norm.isf(.2), 0.]])
    signed = np.stack((means, -means), axis=1)
    shape = _negative_shape()
    profiles = np.array([[1., 1., 1., 0.], [1., 0., 0., 1.]])
    info = {}
    result = projection_pc(signed, 1., shape, np.inf, profiles, support_simes=True, diagnostics=info)
    assert result[0, 0] == pytest.approx(.4)  # Old retained-pair Simes was .2.
    assert info["fallback_intersection_count"] == 2
    assert all(item["simes_eligible"] for item in info["intersections"] if item["sign"] == 1)
    assert [item["retained_studies"] for item in info["intersections"]
            if not item["simes_eligible"]] == [[0, 1, 2], [1, 2]]
    changed = {}
    projection_pc(2*signed, 1., shape, np.inf, profiles, support_simes=True, diagnostics=changed)
    assert changed == info
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("df", [np.inf, 23.])
def test_support_positive_unitdiag_legacy_exact_including_empty_support(df):
    m = np.sin(np.arange(40).reshape(10, 4)*.51)*4
    signed = np.stack((m, -m), axis=1)
    variance = np.linspace(.3, 2., len(m))
    shape = .7*np.eye(4)+.3
    profiles = np.array([[0., 0., 0., 0.], [3., 2., 0., 1.]])
    expected = np.zeros((len(m), 2))
    for sign in range(2):
        for indices in combinations(range(4), 3):
            profile = profiles[sign, list(indices)]
            if not np.any(profile > 0):
                profile = np.ones(3)
            keep = profile > 0
            subshape = shape[np.ix_(indices, indices)]
            statistic = signed[:, sign, list(indices)][:, keep]/np.sqrt(variance[:, None]*np.diag(subshape)[keep])
            p = norm.sf(statistic) if np.isinf(df) else t.sf(statistic, df)
            p = np.where(statistic > 0, p, 1.)
            count = int(keep.sum())
            value = np.min(np.sort(p, axis=1)*count/np.arange(1, count+1), axis=1)
            expected[:, sign] = np.maximum(expected[:, sign], np.where(value <= .5, value, 1.))
    np.testing.assert_array_equal(projection_pc(signed, variance, shape, df, profiles, support_simes=True), expected)


def _fixture(monkeypatch, gaussian, shapes):
    """Small fixed arrays and known noise; no fitted optimization or simulation."""
    g, k = 32, 6
    index = np.arange(g*4*k).reshape(g, 4, k)
    z = np.sin(index*.37)+.7*np.cos(index*.13)
    z += (3*np.sin(np.arange(g*4).reshape(g, 4)*.41))[..., None]
    x = np.cos(np.arange(96*k).reshape(96, k)*.57)+np.sin(np.arange(96*k).reshape(96, k)*.31)
    fit = methods.RadialFit(.3, .8, 7., 0., True, 0, gaussian)
    monkeypatch.setattr(methods, "fit_calibration", lambda *args: fit)
    known_prior = {"df": np.inf if gaussian else 7., "scatter": .8,
                   "gaussian_bic_selected": gaussian, "converged": True, "iterations": 0, "objective": 0.}
    monkeypatch.setattr(prior, "fit_energy_prior", lambda *args: known_prior.copy())

    def residual_energy(z, rho):
        y = z @ methods.contrasts(k)
        q = np.empty(len(z))
        for fold, shape in enumerate(shapes):
            held = np.arange(len(z)) % 2 == fold
            q[held] = np.einsum("gsk,st,gtk->g", y[held], np.linalg.inv(shape), y[held])/(1-rho)
        return q, [{"matrix": shape.tolist()} for shape in shapes]

    monkeypatch.setattr(methods, "crossfit_residual_energy", residual_energy)
    monkeypatch.setattr(loading, "crossfit_residual_energy", residual_energy)
    diagnostic = {"fit": fit.__dict__, "shape": [{"matrix": shape.tolist()} for shape in shapes]}
    return z, x, diagnostic


@pytest.mark.parametrize("gaussian", [False, True])
def test_fixed_and_energy_positive_unitdiag_all_output_arrays_legacy_exact(monkeypatch, gaussian):
    shapes = [.7*np.eye(4)+.3, .4*np.eye(4)+.6]
    z, x, diagnostic = _fixture(monkeypatch, gaussian, shapes)
    current, info = methods.evaluate_candidates(z, x, "frontier")
    current_prior = prior.energy_prior_candidates(z, x, deepcopy(diagnostic))
    monkeypatch.setattr(methods, "simes_pc", _legacy_pc)
    monkeypatch.setattr(prior, "simes_pc", _legacy_pc)
    legacy, _ = methods.evaluate_candidates(z, x, "frontier")
    legacy_prior = prior.energy_prior_candidates(z, x, deepcopy(diagnostic))
    assert current.keys() == legacy.keys()
    for name in current:
        np.testing.assert_array_equal(current[name], legacy[name], err_msg=name)
    for name in current_prior:
        np.testing.assert_array_equal(current_prior[name], legacy_prior[name], err_msg=name)
    # Independently reproduce the OLD energy-prior marginal denominator too,
    # rather than testing only the replacement Simes combiner against itself.
    means = z.mean(-1)
    signed = np.stack((means, -means), axis=1)
    residual = (z-means[..., None]) @ methods.contrasts(6)
    for fold, shape in enumerate(shapes):
        held = np.arange(len(z)) % 2 == fold
        q = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual)/.7
        variance = np.full(held.sum(), (2.5/6)*.8) if gaussian else (2.5/6)*(7*.8+q[held])/27
        statistic = signed[held]/np.sqrt(variance[:, None, None])
        marginal = norm.sf(statistic) if gaussian else t.sf(statistic, 27)
        marginal = np.where(statistic > 0, marginal, 1.)
        expected = _legacy_pc(marginal)
        weight = np.ones(held.sum()) if gaussian else prior.power_radial_weight(q[held], 20, 7., .8, 2.5/6)[0]
        for source in ["target_only", "pooled_prior"]:
            np.testing.assert_array_equal(current_prior[f"{source}_simes_PC"][held], expected)
            np.testing.assert_array_equal(current_prior[f"{source}_bonf_PC"][held],
                                          np.minimum(1., 3*np.sort(marginal, axis=-1)[..., 1]))
            np.testing.assert_array_equal(current_prior[f"{source}_weighted_simes_PC"][held],
                                          np.minimum(1., expected/weight[:, None]))
    for fold in info["simes"]:
        assert fold["policy"] == SIMES_POLICY and fold["fallback_intersection_count"] == 0


@pytest.mark.parametrize("gaussian", [False, True])
def test_fixed_loading_and_prior_share_negative_shape_fallback(monkeypatch, gaussian):
    shapes = [_negative_shape(), .7*np.eye(4)+.3]
    z, x, diagnostic = _fixture(monkeypatch, gaussian, shapes)
    values, info = methods.evaluate_candidates(z, x, "frontier")
    for fold, shape in enumerate(shapes):
        held = np.arange(len(z)) % 2 == fold
        expected = _explicit_pc(values["conditional_t_crossfit"][held], shape)
        np.testing.assert_allclose(values["conditional_simes_assumption_reference_PC"][held], expected, rtol=2e-15)
        assert info["simes"][fold]["fallback_intersection_count"] == (2 if fold == 0 else 0)
    supplied, supplied_info = loading._fixed_baselines(z, info, values)
    recomputed, recomputed_info = loading._fixed_baselines(z, info, None)
    for name in supplied:
        np.testing.assert_array_equal(supplied[name], recomputed[name], err_msg=name)
    assert supplied_info["simes"] == recomputed_info["simes"] == info["simes"]
    energy = prior.energy_prior_candidates(z, x, diagnostic)
    for fold in diagnostic["energy_prior"]["folds"]:
        assert fold["simes"]["fallback_intersection_count"] == (2 if fold["fold"] == 0 else 0)
    assert all(np.isfinite(value).all() for value in energy.values())
    stale = deepcopy(values)
    stale["conditional_simes_assumption_reference_PC"][0, 0] = .123456
    with pytest.raises(ValueError, match="rerun evaluator"):
        loading._fixed_baselines(z, info, stale)


@pytest.mark.parametrize("gaussian", [False, True])
@pytest.mark.parametrize("negative", [False, True])
def test_nonunitdiag_energy_parity_with_cone_projection_and_rescaling(monkeypatch, gaussian, negative):
    correlation = _negative_shape() if negative else .7*np.eye(4)+.3
    z, x, info = _fixture(monkeypatch, gaussian, [correlation, correlation])
    normalized = prior.energy_prior_candidates(z, x, deepcopy(info))
    scales = np.array([2., .5, 3., 1.25])
    shape = correlation*scales[:, None]*scales[None, :]
    info["shape"] = [{"matrix": shape.tolist()} for _ in range(2)]
    scaled_z = z*scales[None, :, None]
    result = prior.energy_prior_candidates(scaled_z, x, info)
    # The prior and cone already receive the same known Q and full shape.
    for name in result:
        np.testing.assert_allclose(result[name], normalized[name], rtol=3e-12, atol=1e-14, err_msg=name)
    means = scaled_z.mean(-1)
    signed = np.stack((means, -means), axis=1)
    residual = (scaled_z-means[..., None]) @ methods.contrasts(6)
    q = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual)/.7
    v = 2.5/6
    df = np.inf if gaussian else 27.
    variance = np.full(len(z), v*.8) if gaussian else v*(7*.8+q)/df
    for baseline, options in [("bonf", {"bonferroni": True}), ("simes", {"support_simes": True})]:
        expected = projection_pc(signed, variance, shape, df, np.ones((2, 4)), **options)
        for source in ["target_only", "pooled_prior"]:
            np.testing.assert_allclose(result[f"{source}_{baseline}_PC"], expected, rtol=3e-12, atol=1e-14)
    # Contrast with the audited bug: omitted diagonal is observably different.
    wrong = norm.sf(signed/np.sqrt(variance[:, None, None])) if gaussian else t.sf(signed/np.sqrt(variance[:, None, None]), df)
    wrong_bonf = np.minimum(1., 3*np.sort(wrong, axis=-1)[..., 1])
    assert np.max(np.abs(result["target_only_bonf_PC"]-wrong_bonf)) > .01


@pytest.mark.parametrize("gaussian", [False, True])
def test_adopted_loading_negative_correlation_matches_known_noise(monkeypatch, gaussian):
    shape = _negative_shape()
    z, _, info = _fixture(monkeypatch, gaussian, [shape, shape])
    monkeypatch.setattr(loading, "loading_gate", lambda *args: (True, {}))
    monkeypatch.setattr(loading, "learned_loading", lambda *args: (np.ones(6), {}))
    monkeypatch.setattr(loading, "tyler_shape", lambda *args: (shape.copy(), {}))
    sentinel = np.ones((len(z), 2))
    result, diagnostics = loading.loading_candidates(z, info, sentinel, sentinel)
    covariance, root, inverse = loading.pipeline_roots(.3, 6)
    weights, basis, v = loading.positive_geometry(np.ones(6), covariance, root, inverse)
    y = (z@inverse)@basis
    q = np.einsum("gsk,st,gtk->g", y, np.linalg.inv(shape), y)
    d = 4*basis.shape[1]
    df = np.inf if gaussian else 7+d
    posterior = np.full(len(z), .8) if gaussian else (7*.8+q)/df
    means = z@weights
    signed = np.stack((means, -means), axis=1)
    statistic = signed/np.sqrt(v*posterior[:, None, None]*np.diag(shape))
    marginal = norm.sf(statistic) if gaussian else t.sf(statistic, df)
    marginal = np.where(statistic > 0, marginal, 1.)
    np.testing.assert_allclose(result["loading_simes_assumption_reference_PC"], _explicit_pc(marginal, shape), rtol=3e-12)
    np.testing.assert_allclose(result["loading_bonf_PC"], np.minimum(1, 3*np.sort(marginal, axis=-1)[..., 1]), rtol=3e-12)
    for fold in diagnostics["simple_baselines"]["folds"]:
        assert fold["simes"]["fallback_intersection_count"] == 2
