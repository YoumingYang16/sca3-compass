import json
from copy import deepcopy

import numpy as np
import pytest
from scipy.special import ndtr
from scipy.stats import t
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_loading as module
from sca3_compass.molecular_methods import partial_conjunction
from sca3_compass.robustness_cone import cone_partial_conjunction
from sca3_compass.robustness_limma import squeeze_var
from sca3_compass.robustness_loading import (
    learned_loading,
    loading_candidates,
    loading_gate,
    pipeline_roots,
    positive_geometry,
)
from sca3_compass.robustness_methods import evaluate_candidates, tyler_shape
from sca3_compass.robustness_weighting import power_radial_weight


@pytest.fixture(autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture(scope="module", params=[False, True], ids=["student", "gaussian"])
def case(request):
    """Actual original evaluator output, without a guessed fallback formula."""
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(812 + int(request.param))
        g, s, k = 65, 4, 6
        _, root, _ = pipeline_roots(.8, k)
        study_root = np.linalg.cholesky(.65*np.ones((s, s)) + .35*np.eye(s))
        z = np.einsum("st,gtk->gsk", study_root, rng.normal(size=(g, s, k))) @ root
        calibration = rng.normal(size=(4, 128, k)) @ root
        if not request.param:
            z *= np.sqrt(3/rng.chisquare(5, g))[:, None, None]
            calibration *= np.sqrt(3/rng.chisquare(5, (4, 128)))[:, :, None]
        z[:16, :2] += 4
        z[16:32] -= 4
        p, diagnostic = evaluate_candidates(z, calibration, profile="frontier")
        assert diagnostic["fit"]["gaussian_bic_selected"] == request.param
        return z, p, diagnostic


BASELINE_SOURCES = {
    "loading_bonf_PC": "conditional_t_crossfit",
    "loading_simes_assumption_reference_PC": "conditional_simes_assumption_reference_PC",
    "loading_weighted_simes_assumption_reference_PC": "power_weighted_simes_assumption_reference_PC",
    "loading_limma_standard_PC": "limma_standard",
    "loading_limma_robust_PC": "limma_robust",
}


def call_candidate(case, **kwargs):
    z, p, diagnostic = case
    return loading_candidates(z, diagnostic, p["conditional_cone_PC"], p["power_weighted_cone_PC"], **kwargs)


def fixed_pc(p, source):
    return p[source] if source.endswith("_PC") else partial_conjunction(p[source], 2)


def set_gates(monkeypatch, flags):
    choices = iter(flags)
    monkeypatch.setattr(module, "loading_gate", lambda *args: (next(choices), {"fixture_gate": True}))


def legacy_candidate_arrays(z, diagnostics, base_pc, base_weighted_pc, force=False):
    """Pre-sidecar candidate calculation, retained independently as regression.

    No simple-baseline helper participates in these expected arrays. Helpers
    for the original loading/geometry/cone are unchanged by this sidecar.
    """
    g, s, k = z.shape
    fit = diagnostics["fit"]
    covariance, root, inverse_root = pipeline_roots(fit["rho"], k)
    output, weighted = np.array(base_pc, copy=True), np.array(base_weighted_pc, copy=True)
    for fold in range(2):
        held = np.arange(g) % 2 == fold
        training = z[~held]
        adopt, _ = module.loading_gate(training, root, inverse_root)
        if adopt or force:
            loading, _ = learned_loading(training @ inverse_root, root)
            weights, basis, v = positive_geometry(loading, covariance, root, inverse_root)
            train_residual = (training @ inverse_root) @ basis
            shape, _ = tyler_shape(train_residual.transpose(0, 2, 1).reshape(-1, s))
            y = (z[held] @ inverse_root) @ basis
            q = np.einsum("gsk,st,gtk->g", y, np.linalg.inv(shape), y)
            d = s*basis.shape[1]
            df = np.inf if fit["gaussian_bic_selected"] else fit["df"] + d
            posterior = (np.full(len(q), fit["scatter"]) if np.isinf(df) else
                         (fit["df"]*fit["scatter"]+q)/df)
            means = z[held] @ weights
            means = np.stack((means, -means), 1)
            value = cone_partial_conjunction(means, v*posterior, [shape, shape], df)
            output[held] = value
            weight = (np.ones(len(q)) if np.isinf(df) else
                      power_radial_weight(q, d, fit["df"], fit["scatter"], v)[0])
            weighted[held] = np.minimum(1, value/weight[:, None])
    return output, weighted


def test_positive_geometry_annihilates_signal_and_is_noise_independent():
    for rho in [.1,.8,.95]:
        covariance,root,inverse=pipeline_roots(rho,6)
        for loading in [np.ones(6),np.array([.2,.5,.8,1.2,1.5,1.8])]:
            w,h,v=positive_geometry(loading,covariance,root,inverse)
            assert np.all(w>=0) and v>0
            np.testing.assert_allclose(w@loading,1,atol=1e-12)
            np.testing.assert_allclose(h.T@h,np.eye(h.shape[1]),atol=1e-12)
            np.testing.assert_allclose(loading@inverse@h,0,atol=1e-12)
            np.testing.assert_allclose(w@root@h,0,atol=1e-12)


def test_loading_gate_uses_predictive_not_in_sample_fit():
    with threadpool_limits(limits=1):
        rng=np.random.default_rng(130987)
        _,root,inverse=pipeline_roots(.8,6)
        noise=rng.normal(size=(240,4,6))@root
        effect=np.zeros((240,4,1))
        effect[:120]=rng.choice([-4.,4.],size=(120,4,1))
        common=noise+effect
        unequal=noise+effect*np.array([.2,.5,.8,1.2,1.5,1.8])
        assert not loading_gate(common,root,inverse)[0]
        assert loading_gate(unequal,root,inverse)[0]
        estimate,diagnostic=learned_loading(unequal@inverse,root)
        assert diagnostic['converged'] and np.all(estimate>0)
        assert np.corrcoef(estimate,[.2,.5,.8,1.2,1.5,1.8])[0,1]>.95


@pytest.mark.parametrize("flags,force", [((False, False), False), ((True, True), False),
                                        ((True, False), False), ((False, True), False),
                                        ((False, False), True)])
def test_original_candidate_outputs_bitwise_unchanged(case, monkeypatch, flags, force):
    z, p, diagnostic = case
    set_gates(monkeypatch, flags)
    expected, expected_weighted = legacy_candidate_arrays(
        z, diagnostic, p["conditional_cone_PC"], p["power_weighted_cone_PC"], force)
    set_gates(monkeypatch, flags)
    result, info = call_candidate(case, force=force, base_baselines=p)
    np.testing.assert_array_equal(result["loading_adaptive_PC"], expected)
    np.testing.assert_array_equal(result["loading_weighted_PC"], expected_weighted)
    assert set(result) == {"loading_adaptive_PC", "loading_weighted_PC", *BASELINE_SOURCES}
    for value in result.values():
        assert value.shape == (len(z), 2)
        assert np.isfinite(value).all() and np.all((value >= 0) & (value <= 1))
    expected_fallback = sum(np.sum(np.arange(len(z)) % 2 == fold)
                            for fold in range(2) if not flags[fold] and not force)/len(z)
    assert info["fallback_gene_fraction"] == expected_fallback


@pytest.mark.parametrize("supplied", [False, True])
def test_all_fallback_is_bitwise_original_baselines(case, monkeypatch, supplied):
    _, p, _ = case
    set_gates(monkeypatch, [False, False])
    result, info = call_candidate(case, base_baselines=p if supplied else None)
    for name, source in BASELINE_SOURCES.items():
        np.testing.assert_array_equal(result[name], fixed_pc(p, source))
    assert info["simple_baselines"]["fallback"]["source"] == (
        "supplied_evaluate_candidates" if supplied else "recomputed_fixed_geometry")


def test_fallback_without_shape_diagnostics_reconstructs_original(case, monkeypatch):
    z, p, diagnostic = case
    minimal = {"fit": diagnostic["fit"]}
    set_gates(monkeypatch, [False, False])
    result, _ = call_candidate((z, p, minimal))
    for name, source in BASELINE_SOURCES.items():
        np.testing.assert_array_equal(result[name], fixed_pc(p, source))


def test_supplied_fallback_never_refits_or_mutates_inputs(case, monkeypatch):
    z, p, diagnostic = case
    snapshot = deepcopy(diagnostic)
    z = z.copy()
    z.flags.writeable = False
    frozen = {key: value.copy() for key, value in p.items()}
    for value in frozen.values():
        value.flags.writeable = False
    set_gates(monkeypatch, [False, False])

    def forbidden(*args, **kwargs):
        raise AssertionError("Complete supplied fallback must not refit geometry/prior")

    monkeypatch.setattr(module, "squeeze_var", forbidden)
    monkeypatch.setattr(module, "crossfit_residual_energy", forbidden)
    result, _ = call_candidate((z, frozen, diagnostic), base_baselines=frozen)
    assert diagnostic == snapshot
    for name, source in BASELINE_SOURCES.items():
        expected = fixed_pc(p, source)
        np.testing.assert_array_equal(result[name], expected)
        result[name][:] = .123
        np.testing.assert_array_equal(frozen[source], p[source])


@pytest.mark.parametrize("constant", [False, True], ids=["df16", "df20"])
def test_adopted_simple_baselines_use_identical_projection_energy_and_full_family(case, monkeypatch, constant):
    z, _, diagnostic = case
    g, s, k = z.shape
    loading = np.ones(k) if constant else np.array([.2, .5, .8, 1.2, 1.5, 1.8])
    shape = .4*np.ones((s, s)) + .6*np.eye(s)
    set_gates(monkeypatch, [True, True])
    monkeypatch.setattr(module, "learned_loading", lambda *args: (loading.copy(), {"fixture": True}))
    monkeypatch.setattr(module, "tyler_shape", lambda *args: (shape.copy(), {"fixture": True}))
    calls = []

    def recording_squeeze(variance, df, robust):
        calls.append((variance.copy(), df, robust))
        return squeeze_var(variance, df=df, robust=robust)

    monkeypatch.setattr(module, "squeeze_var", recording_squeeze)
    result, info = call_candidate(case)
    assert len(calls) == 4  # two variants, each with G observations, never G/2
    fit = diagnostic["fit"]
    covariance, root, inverse = pipeline_roots(fit["rho"], k)
    w, basis, v = positive_geometry(loading, covariance, root, inverse)
    d = s*basis.shape[1]
    assert d == (20 if constant else 16)
    y = (z @ inverse) @ basis
    q = np.einsum("gsk,st,gtk->g", y, np.linalg.inv(shape), y)
    means = z @ w
    means = np.stack((means, -means), axis=1)
    df = np.inf if fit["gaussian_bic_selected"] else fit["df"] + d
    posterior = np.full(g, fit["scatter"]) if np.isinf(df) else (fit["df"]*fit["scatter"] + q)/df
    statistic = means/np.sqrt(v*posterior[:, None, None])
    study_p = np.where(means > 0, ndtr(-statistic) if np.isinf(df) else t.sf(statistic, df), 1.)
    expected_bonf = np.minimum(1, 3*np.sort(study_p, axis=-1)[..., 1])
    raw_simes = np.min(np.sort(study_p, axis=-1)[..., 1:]*[3, 1.5, 1], axis=-1)
    expected_simes = np.where(raw_simes <= .5, raw_simes, 1.)
    own_weight = (np.ones(g) if np.isinf(df) else
                  power_radial_weight(q, d, fit["df"], fit["scatter"], v)[0])
    np.testing.assert_allclose(result["loading_bonf_PC"], expected_bonf, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(result["loading_simes_assumption_reference_PC"], expected_simes, rtol=2e-12, atol=1e-15)
    np.testing.assert_allclose(result["loading_weighted_simes_assumption_reference_PC"],
                               np.minimum(1, expected_simes/own_weight[:, None]), rtol=2e-12, atol=1e-15)
    for robust in [False, True]:
        post, prior, _ = squeeze_var(q/d, df=d, robust=robust)
        total_df = np.minimum(d + np.broadcast_to(prior, (g,)), g*d)
        expected = partial_conjunction(t.sf(means/np.sqrt(v*post[:, None, None]), total_df[:, None, None]), 2)
        name = "loading_limma_robust_PC" if robust else "loading_limma_standard_PC"
        np.testing.assert_allclose(result[name], expected, rtol=2e-10, atol=1e-14)
    for index, (variances, dimension, robust) in enumerate(calls):
        assert len(variances) == g and dimension == d and robust == bool(index % 2)
        np.testing.assert_allclose(variances, q/d, rtol=2e-14)
    simple = info["simple_baselines"]
    assert not simple["limma_prior_is_crossfit"] and simple["limma_uses_own_gene_residual_in_prior"]
    assert simple["geometry_and_gate_shared_with_candidate"]
    assert simple["fallback"]["source"] == "not_used_all_folds_adopted"
    for fold in simple["folds"]:
        assert fold["n_prior_genes"] == g and fold["residual_dimension"] == d
        assert all(entry["n_variances"] == g for entry in fold["limma"])


@pytest.mark.parametrize("adopted_fold", [0, 1])
def test_mixed_gate_keeps_fallback_exact_and_does_not_pool_incompatible_df(case, monkeypatch, adopted_fold):
    z, p, _ = case
    set_gates(monkeypatch, [fold == adopted_fold for fold in range(2)])
    monkeypatch.setattr(module, "learned_loading", lambda *args: (
        np.array([.2, .5, .8, 1.2, 1.5, 1.8]), {"fixture": True}))
    calls = []

    def recording_squeeze(variance, df, robust):
        calls.append((len(variance), df, robust))
        return squeeze_var(variance, df=df, robust=robust)

    monkeypatch.setattr(module, "squeeze_var", recording_squeeze)
    result, info = call_candidate(case, base_baselines=p)
    fallback = np.arange(len(z)) % 2 != adopted_fold
    for name, source in BASELINE_SOURCES.items():
        np.testing.assert_array_equal(result[name][fallback], fixed_pc(p, source)[fallback])
    assert calls == [(len(z), 16, False), (len(z), 16, True)]
    assert [fold["fold"] for fold in info["simple_baselines"]["folds"]] == [adopted_fold]


@pytest.mark.parametrize("invalid", ["missing", "shape", "nan", "negative", "above_one"])
def test_rejects_invalid_supplied_fallback_mapping(case, monkeypatch, invalid):
    _, p, _ = case
    supplied = {key: p[key].copy() for key in BASELINE_SOURCES.values()}
    key = "conditional_t_crossfit"
    if invalid == "missing":
        supplied.pop(key)
    elif invalid == "shape":
        supplied[key] = supplied[key][..., 0]
    else:
        supplied[key][0, 0, 0] = {"nan": np.nan, "negative": -.1, "above_one": 1.1}[invalid]
    set_gates(monkeypatch, [False, False])
    with pytest.raises(ValueError, match="fixed baseline mapping"):
        call_candidate(case, base_baselines=supplied)


def test_simes_extension_precedes_weighting():
    values = np.array([[[.1, .2, .3, .4], [.2, .4, .8, .9], [.01, .01, .8, .9]]])
    np.testing.assert_allclose(module._simes_pc(values, np.eye(4)), [[.4, 1., .03]])


@pytest.mark.parametrize("invalid", ["one_fold", "shape", "nan", "asymmetric", "indefinite"])
def test_rejects_incomplete_or_invalid_recorded_fallback_shapes(case, monkeypatch, invalid):
    z, p, diagnostic = case
    diagnostic = deepcopy(diagnostic)
    if invalid == "one_fold":
        diagnostic["shape"] = diagnostic["shape"][:1]
    else:
        shape = np.eye(4)
        if invalid == "shape":
            shape = np.eye(3)
        elif invalid == "nan":
            shape[0, 0] = np.nan
        elif invalid == "asymmetric":
            shape[0, 1] = .2
        else:
            shape[0, 0] = -1
        diagnostic["shape"][0]["matrix"] = shape.tolist()
    set_gates(monkeypatch, [False, False])
    with pytest.raises(ValueError, match="shape"):
        call_candidate((z, p, diagnostic))


@pytest.mark.parametrize("df", [1.5, 3., 5., 20.])
@pytest.mark.parametrize("rho", [.1, .8, .95])
def test_heavy_tail_loading_probabilities_and_diagnostics_are_finite(df, rho):
    """Numerical smoke only: known fixture parameters are NOT FDR evidence."""
    rng = np.random.default_rng(939)
    g, s, k = 96, 4, 6
    _, root, _ = pipeline_roots(rho, k)
    study_root = np.linalg.cholesky(.65*np.ones((s, s)) + .35*np.eye(s))
    z = np.einsum("st,gtk->gsk", study_root, rng.normal(size=(g, s, k))) @ root
    z *= np.sqrt(df/rng.chisquare(df, g))[:, None, None]
    loading = np.array([.2, .5, .8, 1.2, 1.5, 1.8])
    z[:24] += 4*loading
    z[24:48] -= 4*loading
    diagnostic = {"fit": {"rho": rho, "df": df, "scatter": 1., "gaussian_bic_selected": False}}
    sentinel = np.full((g, 2), .123)
    values, info = loading_candidates(z, diagnostic, sentinel, sentinel, force=True)
    for p in values.values():
        assert p.shape == (g, 2)
        assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    assert info["fallback_gene_fraction"] == 0
    json.dumps(info, allow_nan=False)
