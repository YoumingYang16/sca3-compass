from copy import deepcopy
from inspect import signature

import numpy as np
import pytest
from scipy.stats import chi2, f

from sca3_compass import robustness_prior as prior_module
from sca3_compass.robustness_cone import cone_partial_conjunction
from sca3_compass.robustness_methods import compound_quadratic, contrasts
from sca3_compass.robustness_prior import (
    energy_logpdf,
    energy_prior_candidates,
    fit_energy_prior,
)
from sca3_compass.robustness_weighting import power_radial_weight

LABELS = {
    "target_only_cone_PC", "target_only_weighted_cone_PC",
    "pooled_prior_cone_PC", "pooled_prior_weighted_cone_PC",
}
LABELS.update(f'{source}_{baseline}_PC' for source in ('target_only', 'pooled_prior')
              for baseline in ('bonf', 'simes', 'weighted_simes'))


def _sample_energies(rng, d, df, scatter):
    q = scatter * rng.chisquare(d)
    if np.isfinite(df):
        q *= df / rng.chisquare(df, size=len(d))
    return q


def _data(seed=793, genes=81, df=5.):
    rng = np.random.default_rng(seed)
    rho = .45
    pipeline = (1 - rho) * np.eye(6) + rho
    root = np.linalg.cholesky(pipeline).T
    z = rng.normal(size=(genes, 4, 6)) @ root
    if np.isfinite(df):
        z *= np.sqrt(df / rng.chisquare(df, size=(genes, 1, 1)))
    calibration = rng.normal(size=(4, 80, 6)) @ root
    shapes = [.75 * np.eye(4) + .25, .4 * np.eye(4) + .6]
    diagnostics = {"fit": {"rho": rho}, "shape": [{"matrix": x.tolist()} for x in shapes]}
    return z, calibration, diagnostics


@pytest.mark.parametrize("df", [1.05, 1.5, 5., 30., 200., np.inf])
@pytest.mark.parametrize("scatter", [np.exp(-9), .6, np.exp(9)])
def test_energy_density_matches_scaled_scipy_reference(df, scatter):
    d = np.tile([1., 6., 20., 37.], 15)
    q = scatter * np.geomspace(1e-7, 1e7, len(d))
    reference = (
        chi2.logpdf(q / scatter, d) - np.log(scatter) if np.isinf(df)
        else f.logpdf(q / (d * scatter), d, df) - np.log(d * scatter)
    )
    np.testing.assert_allclose(energy_logpdf(q, d, df, scatter), reference, rtol=2e-13, atol=2e-12)
    np.testing.assert_allclose(
        energy_logpdf(q, 20, df, scatter), energy_logpdf(q, np.full(len(q), 20), df, scatter)
    )


@pytest.mark.parametrize("df", [np.inf, 1.5, 5.])
@pytest.mark.parametrize("mixed", [False, True])
def test_prior_recovers_gaussian_and_student_with_mixed_dimensions(df, mixed):
    rng = np.random.default_rng(712803)
    d = np.resize([20, 6] if mixed else [20], 24000)
    q = _sample_energies(rng, d, df, .7)
    fit = fit_energy_prior(q, d if mixed else 20)
    assert set(fit) == {"df", "scatter", "gaussian_bic_selected", "converged", "iterations", "objective",
                        "optimizer_success", "selected_projected_score", "score_tolerance", "numerical_recovery"}
    assert fit["converged"]
    assert fit["scatter"] == pytest.approx(.7, rel=.06)
    assert fit["gaussian_bic_selected"] == np.isinf(df)
    if np.isinf(df):
        assert np.isinf(fit["df"]) and fit["iterations"] == 0
        assert fit["scatter"] == pytest.approx(q.sum() / d.sum(), rel=1e-14)
    else:
        assert fit["df"] == pytest.approx(df, rel=.09)
        assert fit["iterations"] > 0
    assert fit["objective"] == pytest.approx(
        -energy_logpdf(q, d, fit["df"], fit["scatter"]).mean(), abs=1e-12
    )


def test_bic_uses_energy_observation_count_and_one_versus_two_parameters(monkeypatch):
    # With ten d=20 observations, this improvement beats log(10)/2,
    # but not log(200)/2. The latter would incorrectly count coordinates.
    q = chi2.ppf((np.arange(10) + .5) / 10, 20)
    gaussian_nll = -energy_logpdf(q, 20, np.inf, q.mean() / 20).mean()
    from scipy.optimize import OptimizeResult

    for gain, gaussian in [(0.05, True), (.18, False)]:
        def minimize(*args, gain=gain, **kwargs):
            return OptimizeResult(
                x=np.log([1., 5.]), fun=gaussian_nll - gain, success=True, nit=4
            )
        monkeypatch.setattr(prior_module, "minimize", minimize)
        assert fit_energy_prior(q, 20)["gaussian_bic_selected"] is gaussian


def test_analytic_gradient_and_bounded_multistart(monkeypatch):
    from scipy.optimize import OptimizeResult, check_grad

    calls = []

    def minimize(objective, start, **kwargs):
        calls.append((start, kwargs))
        for params in [np.log([.7, 1.5]), np.log([2., 5.]), np.log([.3, 150.])]:
            error = check_grad(lambda p: objective(p)[0], lambda p: objective(p)[1], params)
            assert error < 2e-6
        return OptimizeResult(x=np.array(start), fun=objective(start)[0], success=False, nit=200)

    monkeypatch.setattr(prior_module, "minimize", minimize)
    # Force recovery to fail too: retaining the old failure-path assertion
    # must not accidentally exercise a now-successful second solver.
    monkeypatch.setattr(prior_module, "least_squares", lambda fun, x, **kwargs:
        OptimizeResult(x=np.array(x), success=False, nfev=100, message="injected recovery failure"))
    rng = np.random.default_rng(623)
    d = np.resize([6, 20], 400)
    fit = fit_energy_prior(_sample_energies(rng, d, 1.5, .7), d)
    assert not fit["gaussian_bic_selected"] and not fit["converged"]
    np.testing.assert_allclose([np.exp(call[0][1]) for call in calls], [5., 30.])
    for _, kwargs in calls:
        assert kwargs["method"] == "L-BFGS-B" and kwargs["jac"] is True
        np.testing.assert_allclose(kwargs["bounds"], [(-9, 9), (np.log(1.05), np.log(200))])


@pytest.mark.parametrize("q,d", [
    ([], 20), (1., 20), ([[1., 2.]], 20), ([0., 1.], 20),
    ([-1., 2.], 20), ([np.nan, 1.], 20), ([np.inf, 1.], 20),
    ([1., 2.], 0), ([1., 2.], -1), ([1., 2.], np.inf),
    ([1., 2.], [6]), ([1., 2.], [[6, 20]]), ([1., 2.], [6, np.nan]),
])
def test_invalid_raw_energies_and_dimensions_are_rejected(q, d):
    with pytest.raises(ValueError):
        fit_energy_prior(q, d)
    with pytest.raises(ValueError):
        energy_logpdf(q, d, 5., 1.)


def test_each_outer_fold_recomputes_train_and_held_energy_with_its_frozen_shape(monkeypatch):
    z, calibration, diagnostics = _data()
    original = deepcopy(diagnostics)
    saved_z, saved_calibration = z.copy(), calibration.copy()
    seen = []
    fit_impl = fit_energy_prior

    def record_fit(q, dimensions):
        seen.append((q.copy(), np.broadcast_to(dimensions, q.shape).copy()))
        return fit_impl(q, dimensions)

    def no_tyler(*args, **kwargs):
        pytest.fail("Supplied outer-fold shapes must not be refitted")

    monkeypatch.setattr(prior_module, "fit_energy_prior", record_fit)
    monkeypatch.setattr("sca3_compass.robustness_methods.tyler_shape", no_tyler)
    result = energy_prior_candidates(z, calibration, diagnostics)
    assert set(result) == LABELS
    assert set(diagnostics["energy_prior"]["model_labels"]) == LABELS
    assert diagnostics["fit"] == original["fit"] and diagnostics["shape"] == original["shape"]
    np.testing.assert_array_equal(z, saved_z)
    np.testing.assert_array_equal(calibration, saved_calibration)
    residual = z @ contrasts(6)
    rho = diagnostics["fit"]["rho"]
    v = (1 + 5 * rho) / 6
    cal_q = compound_quadratic(calibration.reshape(-1, 6), rho)
    for fold, info in enumerate(diagnostics["energy_prior"]["folds"]):
        held = np.arange(len(z)) % 2 == fold
        shape = np.asarray(diagnostics["shape"][fold]["matrix"])
        q = np.einsum("gsk,st,gtk->g", residual, np.linalg.inv(shape), residual) / (1 - rho)
        np.testing.assert_allclose(seen[2 * fold][0], q[~held], atol=1e-12)
        np.testing.assert_array_equal(seen[2 * fold][1], 20)
        np.testing.assert_allclose(seen[2 * fold + 1][0], np.r_[q[~held], cal_q], atol=1e-12)
        np.testing.assert_array_equal(seen[2 * fold + 1][1], np.r_[np.full((~held).sum(), 20), np.full(len(cal_q), 6)])
        means = z[held].mean(-1)
        for source in ["target_only", "pooled_prior"]:
            fit = info[source]
            if fit["gaussian_bic_selected"]:
                df, posterior, weight = np.inf, fit["scatter"], np.ones(held.sum())
            else:
                df = fit["df"] + 20
                posterior = (fit["df"] * fit["scatter"] + q[held]) / df
                weight = power_radial_weight(q[held], 20, fit["df"], fit["scatter"], v)[0]
            expected = cone_partial_conjunction(np.stack((means, -means), 1), v * posterior, [shape, shape], df)
            np.testing.assert_allclose(result[f"{source}_cone_PC"][held], expected, atol=1e-13)
            np.testing.assert_allclose(result[f"{source}_weighted_cone_PC"][held], np.minimum(1, expected / weight[:, None]), atol=1e-13)
    for values in result.values():
        assert values.shape == (len(z), 2) and np.isfinite(values).all()
        assert np.all((values >= 0) & (values <= 1))


def test_held_genes_cannot_change_their_own_fitted_prior():
    z, calibration, diagnostics = _data(genes=120)
    changed_diagnostics = deepcopy(diagnostics)
    energy_prior_candidates(z, calibration, diagnostics)
    changed = z.copy()
    changed[::2] *= 12
    energy_prior_candidates(changed, calibration, changed_diagnostics)
    before, after = diagnostics["energy_prior"]["folds"], changed_diagnostics["energy_prior"]["folds"]
    for source in ["target_only", "pooled_prior"]:
        assert before[0][source] == after[0][source]
        assert before[1][source]["scatter"] != after[1][source]["scatter"]


def test_common_signal_annihilation_sign_symmetry_and_no_oracle_inputs():
    z, calibration, diagnostics = _data()
    base_diagnostics = deepcopy(diagnostics)
    base = energy_prior_candidates(z, calibration, base_diagnostics)
    diagnostics["fit"].update(df=1e8, scatter=500., gaussian_bic_selected=True)
    diagnostics["truth_labels"] = np.ones(len(z))
    altered = energy_prior_candidates(z, calibration, diagnostics)
    for label in LABELS:
        np.testing.assert_array_equal(base[label], altered[label])
    negative = energy_prior_candidates(-z, calibration, deepcopy(diagnostics))
    for label in LABELS:
        np.testing.assert_allclose(negative[label], base[label][:, ::-1], atol=1e-13)
    shifted_diagnostics = deepcopy(diagnostics)
    shift = np.random.default_rng(90).normal(size=(len(z), 4, 1)) * 8
    energy_prior_candidates(z + shift, calibration, shifted_diagnostics)
    for fold in range(2):
        for source in ["target_only", "pooled_prior"]:
            old = base_diagnostics["energy_prior"]["folds"][fold][source]
            new = shifted_diagnostics["energy_prior"]["folds"][fold][source]
            assert old['gaussian_bic_selected'] == new['gaussian_bic_selected']
            np.testing.assert_allclose(old["scatter"], new["scatter"], rtol=1e-7)
            if not old['gaussian_bic_selected']:
                np.testing.assert_allclose(old['df'], new['df'], rtol=1e-7)
    assert list(signature(fit_energy_prior).parameters) == ["q", "dimensions"]
    assert list(signature(energy_prior_candidates).parameters) == ["z", "calibration", "diagnostics", "scale_mode", "geometries"]
    parameter=signature(energy_prior_candidates).parameters['geometries']
    assert parameter.kind.name=='KEYWORD_ONLY' and parameter.default is None
    assert not diagnostics["energy_prior"]["uses_truth_labels"]
    assert not diagnostics["energy_prior"]["automatic_prior_source_selection"]


def test_normal_calibration_does_not_force_t5_targets_into_gaussian_prior():
    z, calibration, diagnostics = _data(seed=3871, genes=6000)
    diagnostics["shape"] = [{"matrix": np.eye(4).tolist()} for _ in range(2)]
    diagnostics["fit"].update(df=1e8, scatter=1., gaussian_bic_selected=True)
    energy_prior_candidates(z, calibration, diagnostics)
    for fold in diagnostics["energy_prior"]["folds"]:
        fit = fold["target_only"]
        assert fit["converged"] and not fit["gaussian_bic_selected"]
        assert fit["df"] == pytest.approx(5., rel=.15)


def test_gaussian_branch_uses_fixed_scatter_and_exactly_unit_weights(monkeypatch):
    z, calibration, diagnostics = _data()
    gaussian = {
        "df": np.inf, "scatter": 1.7, "gaussian_bic_selected": True,
        "converged": True, "iterations": 0, "objective": 0.,
    }
    monkeypatch.setattr(prior_module, "fit_energy_prior", lambda *args: gaussian.copy())

    def no_weights(*args, **kwargs):
        pytest.fail("Gaussian branch must use unit weights")

    monkeypatch.setattr(prior_module, "power_radial_weight", no_weights)
    base = energy_prior_candidates(z, calibration, deepcopy(diagnostics))
    # Change only held-gene residuals, preserving the observed means.
    changed = z.copy()
    mean = z[::2].mean(-1, keepdims=True)
    changed[::2] = mean + 10 * (z[::2] - mean)
    new = energy_prior_candidates(changed, calibration, deepcopy(diagnostics))
    for source in ["target_only", "pooled_prior"]:
        np.testing.assert_array_equal(base[f"{source}_cone_PC"], base[f"{source}_weighted_cone_PC"])
        np.testing.assert_allclose(base[f"{source}_cone_PC"], new[f"{source}_cone_PC"], atol=1e-13)


@pytest.mark.parametrize("bad", ["zero_target", "zero_calibration", "nan_target", "nan_calibration", "rho", "shape", "asymmetric_shape"])
def test_wrapper_rejects_invalid_raw_energies_and_geometry(bad):
    z, calibration, diagnostics = _data()
    if bad == "zero_target":
        z[0] = 0
    elif bad == "zero_calibration":
        calibration[0, 0] = 0
    elif bad == "nan_target":
        z[0, 0, 0] = np.nan
    elif bad == "nan_calibration":
        calibration[0, 0, 0] = np.nan
    elif bad == "rho":
        diagnostics["fit"]["rho"] = 1.
    elif bad == "shape":
        diagnostics["shape"][0]["matrix"] = np.zeros((4, 4)).tolist()
    else:
        diagnostics["shape"][0]["matrix"][0][1] = .5
    with pytest.raises(ValueError):
        energy_prior_candidates(z, calibration, diagnostics)
    assert "energy_prior" not in diagnostics
