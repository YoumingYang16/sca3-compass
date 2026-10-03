import numpy as np
from sca3_compass.robustness_adaptive_cone import projection_face
from sca3_compass.robustness_universal import orthant_distance


def test_active_face_identity_case():
    x=np.array([[1.,2.,-3.],[1.,2.,3.],[-1.,-2.,-3.],[0.,1.,-1.]])
    q,k=projection_face(x,np.eye(3))
    np.testing.assert_allclose(q,np.sum(np.maximum(x,0)**2,axis=1))
    np.testing.assert_array_equal(k,[2,3,0,1])


def test_active_face_agrees_with_original_projection():
    rng=np.random.default_rng(918913)
    for rho in [0.,.65,.99]:
        shape=(1-rho)*np.eye(3)+rho
        x=rng.normal(size=(3000,3))@np.linalg.cholesky(shape).T
        q,k=projection_face(x,shape)
        np.testing.assert_allclose(q,orthant_distance(x,shape),atol=1e-10)
        assert np.all((k>=0)&(k<=3))
