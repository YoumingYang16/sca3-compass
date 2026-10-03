import numpy as np
from scipy.linalg import eigh
from finite_calibration import shape_fit, kappa_upper, contrasts, projection_p
from finite_calibration import matrix_kappa_upper, matrix_order, evaluate


def test_affine_and_block_radial_equivariance():
    rng = np.random.default_rng(91)
    y = rng.normal(size=(40, 4, 5))
    a = np.array([[2, 1, .2, 0], [0, 1, .3, 0], [.1, 0, 3, .4], [0, 0, 0, .5]])
    radial = np.exp(rng.normal(size=(40, 1, 1))*3)
    old = shape_fit(y)
    changed = shape_fit((a @ y)*radial)
    expected = a @ old @ a.T
    expected *= 4/np.trace(expected)
    np.testing.assert_allclose(changed, expected, rtol=1e-10, atol=1e-10)


def test_vectorized_reference_fit():
    y = np.random.default_rng(12).normal(size=(3, 16, 4, 5))
    np.testing.assert_allclose(shape_fit(y), np.stack([shape_fit(x) for x in y]))


def test_kappa_block_count_and_radial_invariance():
    c = np.random.default_rng(15).normal(size=(12, 4, 6))
    u, receipt = kappa_upper(c)
    v, _ = kappa_upper(c*np.arange(1, 13)[:, None, None])
    assert receipt['blocks'] == 12
    np.testing.assert_allclose(u, v)


def test_protection_encloses_known_shape_variance():
    rng = np.random.default_rng(16)
    r = .65*np.ones((4, 4))+.35*np.eye(4)
    h = shape_fit(rng.normal(size=(128, 4, 5)))
    eig = eigh(h, r, eigvals_only=True)
    factor = eig[-1]/eig[0]
    z = rng.normal(size=(32, 4, 6))+3
    a = np.array([1, 0, 1, .5])
    safe = projection_p(z, a, h, 2, factor)
    oracle = projection_p(z, a, r, 2, 1)
    assert np.all(safe >= oracle-1e-14)


def test_shape_initialization_is_not_regularized_old_tyler():
    y = np.random.default_rng(42).normal(size=(8, 4, 5))
    for steps in [0, 1, 12]:
        a = np.diag([1, 2, .5, 4])
        h = shape_fit(y, steps)
        expected = a @ h @ a.T
        expected *= 4/np.trace(expected)
        np.testing.assert_allclose(shape_fit(a@y, steps), expected, atol=1e-9)


def test_matrix_kappa_affine_radial_invariance():
    rng = np.random.default_rng(161)
    c = rng.normal(size=(12, 4, 6))
    a = np.array([[1, .6, 0, 0], [0, 2, 0, .1], [0, 0, 1, .3], [0, 0, 0, .5]])
    u, _, receipt = matrix_kappa_upper(c)
    v, _, _ = matrix_kappa_upper((a@c)*np.exp(rng.normal(size=(12, 1, 1))*2))
    np.testing.assert_allclose(u, v, rtol=1e-10)
    assert receipt['blocks'] == 12


def test_matrix_order_coverage_polynomial():
    from scipy.stats import binom, f
    for n in [4, 12, 32, 128]:
        k, c = matrix_order(n, .005)
        assert binom.sf(k-1, n, f.cdf(c, 4, 2)) <= .005+1e-15


def test_held_changes_do_not_change_its_training_receipt():
    rng = np.random.default_rng(17)
    z = rng.normal(size=(32, 4, 6)); cal = rng.normal(size=(12, 4, 6))
    a = evaluate(z, cal, seed=19)
    z[0] += 50
    b = evaluate(z, cal, seed=19)
    for key in ['shape', 'profiles', 'gamma']:
        np.testing.assert_array_equal(a['folds'][0][key], b['folds'][0][key])
    assert a['folds'][0]['pilots'] == b['folds'][0]['pilots']
    assert a['q_internal'] == .05-.005-2/200


def test_classical_bb_by_floor():
    from sca3_compass.molecular_methods import fdr_adjust
    assert not np.any(fdr_adjust(np.full((256, 2), .01), 'BY') <= .05)
