"""Numerical contracts for geometry extraction, not calibration experiments."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from sca3_compass import robustness_geometry as module
from sca3_compass import robustness_loading as loader
from sca3_compass.robustness_methods import tyler_shape


@pytest.fixture(autouse=True)
def single_thread_blas():
    with threadpool_limits(limits=1):
        yield


def target_fixture(g=65, k=6, rho=.8):
    rng = np.random.default_rng(4312)
    _, root, _ = loader.pipeline_roots(rho, k)
    study_root = np.linalg.cholesky(.4 * np.ones((4, 4)) + .6 * np.eye(4))
    z = np.einsum("st,gtk->gsk", study_root, rng.normal(size=(g, 4, k))) @ root
    z *= np.sqrt(5 / rng.chisquare(5, g))[:, None, None]
    ell = np.linspace(.2, 1.8, k)
    z += rng.choice([-4., 0., 4.], size=(g, 4, 1)) * ell
    return z, {"fit": {"rho": rho}}


def set_gates(monkeypatch, flags):
    decisions = iter(flags)

    def gate(*args):
        adopt = next(decisions)
        return adopt, {"adopt": adopt, "fixture_gate": True}

    monkeypatch.setattr(module, "loading_gate", gate)


def loader_formula(z, diagnostics, fold):
    """Independent adopted branch of loading_candidates, before any prior.

    Use the original helpers directly, including held-only Q as in the
    loader. Do not call the extractor or any of its private helpers.
    """
    g, s, k = z.shape
    held = np.arange(g) % 2 == fold
    training = z[~held]
    covariance, root, inverse_root = loader.pipeline_roots(diagnostics["fit"]["rho"], k)
    loading, _ = loader.learned_loading(training @ inverse_root, root)
    weights, basis, variance = loader.positive_geometry(loading, covariance, root, inverse_root)
    train_residual = (training @ inverse_root) @ basis
    shape, _ = tyler_shape(train_residual.transpose(0, 2, 1).reshape(-1, s))
    residual = (z[held] @ inverse_root) @ basis
    q = np.empty(g)
    q[held] = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual)
    q[~held] = np.einsum("gsk,st,gtk->g", train_residual, np.linalg.inv(shape), train_residual)
    return {
        "means": z @ weights, "q": q, "dimension": s * basis.shape[1],
        "projection_variance": variance, "study_shape": shape, "fold": fold,
    }, loading, weights


def assert_same_geometry(actual, expected):
    assert set(actual) == set(expected)
    for key in ("means", "q", "study_shape"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-14, atol=1e-14)
    for key in ("dimension", "projection_variance", "fold"):
        assert actual[key] == expected[key]


@pytest.mark.parametrize("g,k,rho", [(24, 3, -.2), (65, 6, .8), (66, 6, .95)])
def test_adopted_all_gene_geometry_matches_independent_loader_formula(g, k, rho):
    z, diagnostic = target_fixture(g, k, rho)
    geometries, receipt = module.fit_fold_geometries(z, diagnostic, force=True)
    _, root, inverse = loader.pipeline_roots(rho, k)
    for fold, geometry in enumerate(geometries):
        expected, loading, weights = loader_formula(z, diagnostic, fold)
        assert_same_geometry(geometry, expected)
        held = np.arange(g) % 2 == fold
        np.testing.assert_array_equal(geometry["means"][held], z[held] @ weights)
        np.testing.assert_array_equal(geometry["q"][held], expected["q"][held])
        assert geometry["means"].shape == (g, 4) and geometry["q"].shape == (g,)
        assert geometry["dimension"] in (4 * (k - 2), 4 * (k - 1))
        info = receipt["folds"][fold]
        np.testing.assert_array_equal(info["loading"], loading)
        np.testing.assert_array_equal(info["projection_weights"], weights)
        assert info["gate"] == loader.loading_gate(z[~held], root, inverse)[1]
        assert info["study_shape_condition_number"] >= 1
    assert receipt["fallback_gene_fraction"] == 0
    assert receipt["frozen_fit_rho"] == rho
    assert receipt["pipeline_condition_number"] >= 1
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize("adopted_fold", [0, 1])
def test_mixed_gate_preserves_fold_slots_and_odd_family_fraction(monkeypatch, adopted_fold):
    z, diagnostic = target_fixture()
    set_gates(monkeypatch, [fold == adopted_fold for fold in range(2)])
    geometries, receipt = module.fit_fold_geometries(z, diagnostic)
    assert len(geometries) == 2
    assert geometries[1 - adopted_fold] is None
    expected, _, _ = loader_formula(z, diagnostic, adopted_fold)
    assert_same_geometry(geometries[adopted_fold], expected)
    fallback = np.arange(len(z)) % 2 != adopted_fold
    assert receipt["fallback_gene_fraction"] == fallback.sum() / len(z)
    assert [entry["fold"] for entry in receipt["folds"]] == [0, 1]
    assert receipt["folds"][1 - adopted_fold]["fallback_to_fixed_geometry"]


def test_no_adoption_returns_none_without_fitting_fallback_or_prior(monkeypatch):
    z, diagnostic = target_fixture()
    set_gates(monkeypatch, [False, False])

    def forbidden(*args, **kwargs):
        raise AssertionError("No post-gate geometry, fallback, prior or p-value work permitted")

    for name in ("learned_loading", "positive_geometry", "tyler_shape"):
        monkeypatch.setattr(module, name, forbidden)
    for name in ("_fixed_baselines", "_loading_baselines", "squeeze_var",
                 "student_tail", "cone_partial_conjunction", "power_radial_weight"):
        monkeypatch.setattr(loader, name, forbidden)
    geometries, receipt = module.fit_fold_geometries(z, diagnostic)
    assert geometries == [None, None]
    assert receipt["fallback_gene_fraction"] == 1
    assert all(entry["fallback_to_fixed_geometry"] for entry in receipt["folds"])
    assert all("loading" not in entry for entry in receipt["folds"])


def test_force_records_failed_gates_and_adopts_both(monkeypatch):
    z, diagnostic = target_fixture()
    set_gates(monkeypatch, [False, False])
    geometries, receipt = module.fit_fold_geometries(z, diagnostic, force=np.bool_(True))
    assert all(geometry is not None for geometry in geometries)
    assert receipt["fallback_gene_fraction"] == 0
    for info in receipt["folds"]:
        assert not info["gate"]["adopt"]
        assert info["forced_ablation"] and not info["fallback_to_fixed_geometry"]


def test_every_learning_call_receives_only_that_folds_training_rows(monkeypatch):
    z, diagnostic = target_fixture()
    _, root, inverse = loader.pipeline_roots(diagnostic["fit"]["rho"], z.shape[-1])
    calls = []

    def gate(training, actual_root, actual_inverse):
        fold = len(calls) // 4
        held = np.arange(len(z)) % 2 == fold
        np.testing.assert_array_equal(training, z[~held])
        np.testing.assert_array_equal(actual_root, root)
        np.testing.assert_array_equal(actual_inverse, inverse)
        calls.append("gate")
        return True, {"adopt": True}

    def learn(whitened, actual_root):
        fold = len(calls) // 4
        held = np.arange(len(z)) % 2 == fold
        np.testing.assert_array_equal(whitened, z[~held] @ inverse)
        calls.append("loading")
        return loader.learned_loading(whitened, actual_root)

    bases = []

    def geometry(*args):
        result = loader.positive_geometry(*args)
        bases.append(result[1])
        calls.append("projection")
        return result

    def shape(directions):
        fold = len(calls) // 4
        held = np.arange(len(z)) % 2 == fold
        residual = (z[~held] @ inverse) @ bases[fold]
        expected = residual.transpose(0, 2, 1).reshape(-1, 4)
        np.testing.assert_array_equal(directions, expected)
        calls.append("shape")
        return tyler_shape(directions)

    monkeypatch.setattr(module, "loading_gate", gate)
    monkeypatch.setattr(module, "learned_loading", learn)
    monkeypatch.setattr(module, "positive_geometry", geometry)
    monkeypatch.setattr(module, "tyler_shape", shape)
    module.fit_fold_geometries(z, diagnostic)
    assert calls == ["gate", "loading", "projection", "shape"] * 2


@pytest.mark.parametrize("fold", [0, 1])
def test_held_perturbation_cannot_change_own_geometry_or_training_q(fold):
    z, diagnostic = target_fixture()
    held = np.arange(len(z)) % 2 == fold
    changed = z.copy()
    changed[held] += np.linspace(11., 29., z.shape[-1])
    first, first_receipt = module.fit_fold_geometries(z, diagnostic, force=True)
    second, second_receipt = module.fit_fold_geometries(changed, diagnostic, force=True)
    assert first_receipt["folds"][fold] == second_receipt["folds"][fold]
    for key in ("dimension", "projection_variance"):
        assert first[fold][key] == second[fold][key]
    np.testing.assert_array_equal(first[fold]["study_shape"], second[fold]["study_shape"])
    for key in ("means", "q"):
        np.testing.assert_array_equal(first[fold][key][~held], second[fold][key][~held])
    assert not np.allclose(first[fold]["means"][held], second[fold]["means"][held])


@pytest.mark.parametrize("constant", [False, True], ids=["two_constraints", "one_constraint"])
def test_known_geometry_noise_orthogonality_and_shared_positive_null(monkeypatch, constant):
    k, s, rho = 6, 4, .8
    ell = np.ones(k) if constant else np.array([.2, .5, .8, 1.2, 1.5, 1.8])
    covariance, root, inverse = loader.pipeline_roots(rho, k)
    weights, basis, variance = loader.positive_geometry(ell, covariance, root, inverse)
    shape = .4 * np.ones((s, s)) + .6 * np.eye(s)
    # This finite design has exactly zero mean and identity second moment;
    # no Monte Carlo thresholds or inference-performance claims are involved.
    identity = np.eye(s * k)
    whitened = (np.concatenate((identity, -identity)) * np.sqrt(s * k)).reshape(-1, s, k)
    z = np.einsum("st,gtk->gsk", np.linalg.cholesky(shape), whitened) @ root
    monkeypatch.setattr(module, "loading_gate", lambda *args: (True, {"adopt": True}))
    monkeypatch.setattr(module, "learned_loading", lambda *args: (ell.copy(), {}))
    monkeypatch.setattr(module, "tyler_shape", lambda *args: (shape.copy(), {}))
    geometries, receipt = module.fit_fold_geometries(z, {"fit": {"rho": rho}})
    residual = (z @ inverse) @ basis
    assert np.all(weights >= 0) and variance > 0
    np.testing.assert_allclose(weights @ ell, 1, atol=1e-14)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-14)
    np.testing.assert_allclose(ell @ inverse @ basis, 0, atol=1e-14)
    np.testing.assert_allclose(weights @ root @ basis, 0, atol=1e-14)
    for geometry in geometries:
        assert geometry["dimension"] == (20 if constant else 16)
        np.testing.assert_array_equal(geometry["means"], z @ weights)
        np.testing.assert_allclose(
            geometry["q"], np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual),
            rtol=1e-14,
        )
        np.testing.assert_allclose(geometry["means"].T @ geometry["means"] / len(z),
                                   variance * shape, atol=1e-14)
        np.testing.assert_allclose(geometry["means"].T @ residual.reshape(len(z), -1) / len(z),
                                   0, atol=1e-14)
        np.testing.assert_allclose(geometry["q"].mean(), geometry["dimension"], atol=1e-13)
    assert receipt["folds"][0]["projection_variance"] == variance
    beta = np.resize(np.array([0., -3., 2., 0.]), (len(z), s))
    shifted, _ = module.fit_fold_geometries(z + beta[..., None] * ell, {"fit": {"rho": rho}})
    signal, _ = module.fit_fold_geometries(beta[..., None] * ell, {"fit": {"rho": rho}})
    for fold in range(2):
        np.testing.assert_allclose(shifted[fold]["q"], geometries[fold]["q"], atol=1e-12)
        np.testing.assert_allclose(shifted[fold]["means"], geometries[fold]["means"] + beta, atol=1e-14)
        np.testing.assert_allclose(signal[fold]["means"], beta, atol=1e-14)
        np.testing.assert_allclose(signal[fold]["q"], 0, atol=1e-25)
        assert np.all(signal[fold]["means"][beta == 0] == 0)
        assert np.all(signal[fold]["means"][beta < 0] <= 0)


def test_reuses_original_helpers_and_leaves_inputs_fit_and_loading_source_untouched():
    for name in ("pipeline_roots", "loading_gate", "learned_loading", "positive_geometry"):
        assert getattr(module, name) is getattr(loader, name)
    assert module.tyler_shape is tyler_shape
    z, diagnostic = target_fixture()
    snapshot = z.copy()
    z.flags.writeable = False
    diagnostic["fit"].update(df=object(), scatter=object(), gaussian_bic_selected=object())
    diagnostic["shape"] = object()
    before = dict(diagnostic["fit"])
    frozen = MappingProxyType({"fit": MappingProxyType(diagnostic["fit"]), "shape": diagnostic["shape"]})
    source = Path(loader.__file__)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    geometries, receipt = module.fit_fold_geometries(z, frozen, force=True)
    assert frozen["fit"] == before
    np.testing.assert_array_equal(z, snapshot)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert not receipt["uses_truth_labels"] and not receipt["refits_calibration_rho"]
    assert not receipt["uses_own_gene_to_learn_loading"]
    assert not receipt["uses_own_gene_to_learn_study_shape"]
    geometries[0]["means"][:] = 0
    np.testing.assert_array_equal(z, snapshot)


def test_receipt_is_strict_json_and_does_not_store_gene_arrays():
    z, diagnostic = target_fixture()
    _, receipt = module.fit_fold_geometries(z, diagnostic, force=True)
    encoded = json.dumps(receipt, allow_nan=False)
    assert json.loads(encoded) == receipt
    assert len(encoded) < 8000

    def check(value):
        if isinstance(value, dict):
            assert not {"means", "q", "held", "training", "residual", "basis"} & value.keys()
            for item in value.values():
                check(item)
        elif isinstance(value, list):
            assert len(value) <= z.shape[-1]
            for item in value:
                check(item)

    check(receipt)
    assert "shared positive loading" in receipt["conditions"]
    assert "gene-specific loading heterogeneity" in receipt["limitations"]


@pytest.mark.parametrize("z", [
    np.ones((23, 4, 3)), np.ones((24, 3, 3)), np.ones((24, 4, 2)),
    np.ones((24, 4)), np.ones((24, 4, 3, 1)), np.array(1.),
    np.full((24, 4, 3), np.nan), np.full((24, 4, 3), np.inf),
    np.ones((24, 4, 3), dtype=complex), np.full((24, 4, 3), "1"),
])
def test_rejects_invalid_targets_before_learning(monkeypatch, z):
    monkeypatch.setattr(module, "loading_gate", lambda *args: pytest.fail("Invalid input reached gate"))
    with pytest.raises(ValueError, match="targets"):
        module.fit_fold_geometries(z, {"fit": {"rho": .2}})


@pytest.mark.parametrize("rho", [np.nan, np.inf, -np.inf, -.5, -.6, 1., 1.1,
                                  [.2], "0.2", True, .2 + .1j, None])
def test_rejects_invalid_frozen_rho_before_learning(monkeypatch, rho):
    monkeypatch.setattr(module, "loading_gate", lambda *args: pytest.fail("Invalid rho reached gate"))
    with pytest.raises(ValueError, match="rho"):
        module.fit_fold_geometries(np.ones((24, 4, 3)), {"fit": {"rho": rho}})


@pytest.mark.parametrize("diagnostic", [None, [], {}, {"fit": None}, {"fit": {}}, {"fit": 1}])
def test_requires_frozen_fit_rho(diagnostic):
    with pytest.raises(ValueError, match="rho"):
        module.fit_fold_geometries(np.ones((24, 4, 3)), diagnostic)


@pytest.mark.parametrize("force", [1, "false", None, [False]])
def test_rejects_nonboolean_force(force):
    with pytest.raises(TypeError, match="force"):
        module.fit_fold_geometries(np.ones((24, 4, 3)), {"fit": {"rho": .2}}, force=force)


def test_changing_unused_prior_fields_cannot_change_geometry():
    z, diagnostic = target_fixture()
    changed = deepcopy(diagnostic)
    changed["fit"].update(df=1.1, scatter=1e9, gaussian_bic_selected=False)
    first, first_receipt = module.fit_fold_geometries(z, diagnostic, force=True)
    second, second_receipt = module.fit_fold_geometries(z, changed, force=True)
    for actual, expected in zip(first, second):
        assert_same_geometry(actual, expected)
    assert first_receipt == second_receipt
