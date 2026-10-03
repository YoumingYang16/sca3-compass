import inspect
from dataclasses import replace
from itertools import combinations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import f, norm, t

from sca3_compass.robustness_cone import cone_partial_conjunction
from sca3_compass.robustness_pc_weight import (
    METHODS,
    _allocate,
    _model_u,
    _PCScorer,
    _student_bank,
    build_pc_weight_tables,
)
from sca3_compass.robustness_projection import projection_pc
from sca3_compass.robustness_simes import simes_pc


def build(**kwargs):
    args = {"dimension": 20, "prior_df": 5., "prior_scale": .6, "projection_variance": .8,
            "shape": .65 * np.ones((4, 4)) + .35 * np.eye(4),
            "profiles": np.array([[1., 1., 0., 0.], [1., .7, 1., .3]]),
            "bins": 8, "noise_samples": 48, "batch_size": 24}
    args.update(kwargs)
    return build_pc_weight_tables(**args)


@pytest.fixture(scope="module")
def small():
    return build()


@pytest.mark.parametrize("df", [25., np.inf])
@pytest.mark.parametrize("rho", [.65, -.2])
def test_cached_kernels_match_existing_direct_pc(df, rho):
    shape = (1 - rho) * np.eye(4) + rho * np.ones((4, 4))
    diagonal = np.array([.7, 1.2, 2., .9])
    shape = diagonal[:, None] * shape * diagonal[None, :]
    profiles = np.array([[2., 2., 0., 0.], [1., 0., 0., 0.]])
    # Includes empty retained triples, negative shapes and non-unit diagonals.
    rng = np.random.default_rng(567)
    means = rng.normal(size=(73, 2, 4)) * 2 + 1
    variance = np.linspace(.3, 2., len(means))
    scorer = _PCScorer(shape, profiles, df)
    fast = scorer.pvalues(means / np.sqrt(variance[:, None, None]))
    for method, option in (("projection", {}), ("support_simes", {"support_simes": True}),
                           ("profile_bonferroni", {"bonferroni": True})):
        expected = projection_pc(means, variance, shape, df, profiles, **option)
        np.testing.assert_allclose(fast[method], expected, rtol=2e-12, atol=2e-14)
    expected_cone = cone_partial_conjunction(means, variance, [shape, shape], df)
    np.testing.assert_allclose(fast["cone"], expected_cone, rtol=2e-12, atol=2e-14)
    z = means / np.sqrt(variance[:, None, None] * np.diag(shape))
    marginal = norm.sf(z) if np.isinf(df) else t.sf(z, df)
    marginal[z <= 0] = 1.
    np.testing.assert_allclose(fast["simes"], simes_pc(marginal, shape), atol=2e-14)
    bonf = np.max([np.minimum(1., 3 * marginal[..., list(j)].min(axis=-1))
                   for j in combinations(range(4), 3)], axis=0)
    np.testing.assert_allclose(fast["bonferroni"], bonf, atol=2e-14)


def test_projection_independent_closed_form_bottleneck():
    rng = np.random.default_rng(98)
    means = rng.normal(size=(123, 2, 4)) + 2
    profile = np.ones((2, 4))
    score = _PCScorer(np.eye(4), profile, 23, ("projection",))
    minimum = np.min([means[..., list(j)].sum(axis=-1) / np.sqrt(3)
                      for j in combinations(range(4), 3)], axis=0)
    expected = np.where(minimum > 0, t.sf(minimum, 23), 1.)
    np.testing.assert_allclose(score.pvalues(means)["projection"], expected, atol=2e-14)
    # One highly positive study cannot erase the triple excluding it.
    means[:] = 0
    means[..., 0] = 100
    np.testing.assert_array_equal(score.pvalues(means)["projection"], np.ones((123, 2)))


def test_builder_power_and_clustered_se_against_direct_pc():
    result = build(bins=4, noise_samples=32, batch_size=11,
                   methods=("projection", "support_simes"))
    bank = _student_bank(32, 25, result.shape, result.diagnostics["seed"])
    grid = np.array(result.diagnostics["weight_grid"])
    effects = result.diagnostics["effects"]
    for b in range(4):
        q = 20 * .6 * f.ppf((b + .5) / 4, 20, 5)
        variance = .8 * (5 * .6 + q) / 25
        for method in result.tables:
            successes = []
            for amplitude in effects:
                means = np.sqrt(variance) * bank[:, None, :] + amplitude * result.profiles[None]
                p = projection_pc(means, variance, result.shape, 25, result.profiles,
                                  support_simes=method == "support_simes")
                successes.append(p[..., None] <= .0003 * grid)
            per_row = np.mean(successes, axis=0)
            np.testing.assert_allclose(result.predictions[method][:, b], per_row.mean(axis=0), atol=1e-15)
            np.testing.assert_allclose(result.prediction_se[method][:, b],
                                       per_row.std(axis=0, ddof=1) / np.sqrt(32), atol=1e-15)


def test_exact_model_integrals_each_method_sign_and_allocation(small):
    for method in METHODS:
        for mode in ("height", "subbin"):
            np.testing.assert_allclose(small.model_integral(method, allocation=mode), 1., atol=1e-14, rtol=0)
            # Integrate actual lookup on every constant interval, not observed Q.
            boundaries = np.arange(9) / 8
            if mode == "subbin":
                boundaries = np.unique(np.r_[boundaries,
                    ((np.arange(8)[:, None] + small.fractions[method]) / 8).ravel()])
            mids = (boundaries[:-1] + boundaries[1:]) / 2
            weights = small.lookup_u(mids, method, allocation=mode)
            integrated = np.diff(boundaries) @ weights
            np.testing.assert_allclose(integrated, 1., atol=2e-14, rtol=0)
            assert np.all((weights >= .05 - 1e-14) & (weights <= 20 + 1e-14))
    assert any(not np.array_equal(small.tables[m][0], small.tables[m][1]) for m in METHODS)
    assert not np.array_equal(small.tables["projection"], small.tables["simes"])


def test_midpoint_q_lookup_and_extreme_finite_q(small):
    u = (np.arange(8) + .5) / 8
    q = 20 * .6 * f.ppf(u, 20, 5)
    np.testing.assert_allclose(_model_u(q, 20, 5, .6), u, atol=2e-14)
    np.testing.assert_array_equal(small.lookup(q), small.tables["projection"].T)
    finite = np.array([0., np.nextafter(0., 1.), 1., np.finfo(float).max])
    weights = small.lookup(finite)
    np.testing.assert_array_equal(weights[0], small.tables["projection"][:, 0])
    np.testing.assert_array_equal(weights[-1], small.tables["projection"][:, -1])
    assert np.isfinite(weights).all()
    assert small.lookup(2.).shape == (2,)
    assert small.lookup(np.ones((2, 3))).shape == (2, 3, 2)
    assert small.lookup(np.empty(0)).shape == (0, 2)


def test_subbin_exact_endpoints_and_ties(small):
    # Binary-representable split fractions make the tie convention test exact.
    toy = replace(small, tables={"projection": np.arange(8).reshape(2, 4) + 1.},
                  overspend={"projection": np.full((2, 4), 2.)},
                  underspend={"projection": np.full((2, 4), .5)},
                  fractions={"projection": np.array([.25, .75])})
    for b in range(4):
        left = b / 4
        np.testing.assert_array_equal(toy.lookup_u(left, allocation="subbin"), [2., 2.])
        first = (b + .25) / 4
        np.testing.assert_array_equal(toy.lookup_u(np.nextafter(first, 0), allocation="subbin"), [2., 2.])
        np.testing.assert_array_equal(toy.lookup_u(first, allocation="subbin"), [.5, 2.])
        np.testing.assert_array_equal(toy.lookup_u((b + .75) / 4, allocation="subbin"), [.5, .5])
    np.testing.assert_array_equal(toy.lookup_u(1., allocation="subbin"), [.5, .5])
    np.testing.assert_array_equal(toy.lookup_u(1.), toy.tables["projection"][:, -1])
    full = replace(toy, fractions={"projection": np.array([0., 1.])})
    np.testing.assert_array_equal(full.lookup_u([0., 1.], allocation="subbin"), [[.5, 2.], [.5, 2.]])


def test_nonconcavity_ablation_realizes_midpoint_mixture():
    grid = np.array([.5, 1., 2.])
    prediction = np.tile([0., 0., 1.], (4, 1))
    height, over, under, fraction, proxy = _allocate(prediction, grid)
    np.testing.assert_allclose(height, 1., atol=1e-14)
    assert fraction == pytest.approx(1 / 3)
    assert proxy == pytest.approx(1 / 3)
    # A step at weight 2 has zero power at the interpolated height 1.
    assert np.mean(height >= 2.) == 0
    assert fraction * np.mean(over >= 2.) + (1 - fraction) * np.mean(under >= 2.) == pytest.approx(proxy)
    assert quad(lambda u: 2. if u < fraction else .5, 0, 1, points=[fraction])[0] == pytest.approx(1.)
    flat = _allocate(np.zeros((4, 3)), grid)
    np.testing.assert_allclose(flat[0], 1., atol=1e-14)


def test_gaussian_exact_ones_no_bank(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Gaussian path must not draw a bank")
    monkeypatch.setattr("sca3_compass.robustness_pc_weight._student_bank", forbidden)
    result = build(prior_df=np.inf)
    for method in METHODS:
        np.testing.assert_array_equal(result.tables[method], np.ones((2, 8)))
        for mode in ("height", "subbin"):
            np.testing.assert_array_equal(result.lookup([0., 1., np.finfo(float).max], method, allocation=mode), np.ones((3, 2)))
    assert result.diagnostics["bank_sha256"] is None
    assert not result.predictions


def test_bank_deterministic_batch_invariant_and_no_global_rng_or_input_mutation():
    shape = .4 * np.ones((4, 4)) + .6 * np.eye(4)
    profile = np.array([[2., 1., 0., 0.], [0., 3., 3., 1.]])
    shape_before, profile_before = shape.copy(), profile.copy()
    global_before = np.random.get_state()
    one = build(shape=shape, profiles=profile, batch_size=7, methods=("projection", "support_simes"))
    two = build(shape=shape, profiles=profile * 4, batch_size=24, methods=("projection", "support_simes"))
    for a, b in zip(global_before, np.random.get_state()):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(shape, shape_before)
    np.testing.assert_array_equal(profile, profile_before)
    for method in one.tables:
        np.testing.assert_array_equal(one.tables[method], two.tables[method])
        np.testing.assert_array_equal(one.predictions[method], two.predictions[method])
        np.testing.assert_array_equal(one.prediction_se[method], two.prediction_se[method])
    assert one.diagnostics["bank_sha256"] == two.diagnostics["bank_sha256"]
    a = _student_bank(48, 25, shape, 7)
    np.testing.assert_array_equal(a, _student_bank(48, 25, shape, 7))
    assert not np.array_equal(a, _student_bank(48, 25, shape, 8))
    q = np.array([0., 2., 99.])
    before = q.copy()
    one.lookup(q, allocation="subbin")
    np.testing.assert_array_equal(q, before)
    with pytest.raises(ValueError):
        one.tables["projection"][0, 0] = 0
    with pytest.raises(ValueError):
        one.tables["projection"].setflags(write=True)
    with pytest.raises(TypeError):
        one.tables["projection"] = np.ones((2, 8))
    assert not one.diagnostics["uses_observed_means"]
    assert not one.diagnostics["uses_truth_labels"]
    assert not {"means", "labels", "q"}.intersection(inspect.signature(build_pc_weight_tables).parameters)


@pytest.mark.parametrize("change", [
    {"shape": np.eye(3)}, {"shape": np.ones((4, 4))}, {"shape": np.eye(4) * np.nan},
    {"shape": np.eye(4) + np.triu(np.ones((4, 4)), 1)},
    {"profiles": np.zeros((2, 4))}, {"profiles": -np.ones((2, 4))},
    {"profiles": np.full((2, 4), np.inf)}, {"profiles": np.ones((4,))},
    {"prior_df": 0}, {"prior_df": np.nan}, {"prior_df": -np.inf},
    {"prior_scale": 0}, {"prior_scale": np.inf}, {"projection_variance": -1},
    {"dimension": 0}, {"dimension": True}, {"dimension": 3.5}, {"bins": 1},
    {"noise_samples": 1}, {"batch_size": 0}, {"seed": -1},
    {"reference_level": 0}, {"reference_level": .05}, {"reference_level": np.nan},
    {"effects": ()}, {"effects": (2., np.inf)}, {"effects": (0.,)},
    {"weight_grid": (.5, 2.)}, {"weight_grid": (0., 1., 2.)},
    {"weight_grid": (.5, 1., 1., 2.)}, {"methods": ("unknown",)},
    {"methods": ()}, {"methods": ("projection", "projection")},
])
def test_invalid_builder_inputs_fail_closed(change):
    with pytest.raises((ValueError, FloatingPointError)):
        build(**change)


@pytest.mark.parametrize("q", [-1., np.nan, np.inf, -np.inf, [0., np.nan]])
def test_invalid_q_fails_closed_even_gaussian(small, q):
    with pytest.raises(ValueError):
        small.lookup(q)
    gaussian = replace(small, prior_df=np.inf)
    with pytest.raises(ValueError):
        gaussian.lookup(q)


@pytest.mark.parametrize("u", [-1e-20, 1.00000000001, np.inf, np.nan])
def test_invalid_intervals_fail_closed(small, u):
    with pytest.raises(ValueError):
        small.lookup_u(u)


def test_overflowing_model_fails_closed_and_no_optimistic_fallback():
    with pytest.raises(FloatingPointError):
        build(prior_scale=np.finfo(float).max)
    with pytest.raises(FloatingPointError):
        build(prior_df=np.inf, prior_scale=np.finfo(float).max, projection_variance=10.)
    scorer = _PCScorer(np.eye(4), np.ones((2, 4)), 25)
    with pytest.raises(ValueError):
        scorer.pvalues(np.full((1, 2, 4), np.inf))


def test_no_invalid_lookup_modes(small):
    with pytest.raises(ValueError):
        small.lookup(1., allocation="random")
    with pytest.raises(ValueError):
        small.lookup(1., "unknown")
