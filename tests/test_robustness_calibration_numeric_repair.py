import numpy as np
import pytest
from scipy.optimize import OptimizeResult
from sca3_compass.robustness_calibration_numeric_repair import (
    gaussian_value_gradient, gaussian_interior_solution, select_student_start,
    fit_calibration_with_diagnostics)


def test_exact_gaussian_eigenvariance_optimum_and_gradient():
    result = gaussian_interior_solution(2.5, 8., 6)
    assert result.success
    scale, rho = np.exp(result.x[1]), result.x[0]
    assert scale*(1-rho) == pytest.approx(.5)
    assert scale*(1+5*rho) == pytest.approx(8)
    assert np.max(np.abs(result.jac)) < 1e-12
    center = np.array([.7, -.2])
    value, grad = gaussian_value_gradient(center, 2.5, 8, 6)
    for i in range(2):
        delta=np.zeros(2);delta[i]=1e-6
        finite=(gaussian_value_gradient(center+delta,2.5,8,6)[0]-
                gaussian_value_gradient(center-delta,2.5,8,6)[0])/2e-6
        assert finite == pytest.approx(grad[i], abs=1e-7)
    assert value >= result.fun


@pytest.mark.parametrize('o,c,k', [(0,1,6),(1,0,6),(1,1,1),(np.nan,1,6),
    (1e-8,10,6),(10,1e-8,6),(1e-10,1e-10,6),(1e8,1e8,6)])
def test_invalid_or_outside_box_solution_is_not_clipped(o,c,k):
    assert gaussian_interior_solution(o,c,k) is None


def test_roundoff_better_failed_start_cannot_displace_success():
    bad=OptimizeResult(success=False,fun=-1.0000000000004,x=np.ones(3))
    good=OptimizeResult(success=True,fun=-1.,x=np.ones(3))
    fit,ok=select_student_start([bad,good])
    assert fit is good and ok
    fit,ok=select_student_start([bad])
    assert fit is bad and not ok
    with pytest.raises(ArithmeticError):
        select_student_start([OptimizeResult(success=False,fun=np.nan,x=np.ones(3))])


def test_degenerate_data_are_explicit_failure():
    with pytest.raises(ArithmeticError):fit_calibration_with_diagnostics(np.zeros((32,6)))
    with pytest.raises(ValueError):fit_calibration_with_diagnostics(np.array([[1,np.nan],[2,3]]))


def test_fixed_gaussian_fixture_reproducible_without_labels():
    x=np.random.default_rng(16421).normal(size=(96,6))
    a,da=fit_calibration_with_diagnostics(x)
    b,db=fit_calibration_with_diagnostics(x.copy())
    assert a==b and da==db and a.converged
    assert da['gaussian_analytic_interior']
    assert not da['independent_confirmation']


def test_failed_student_model_does_not_silently_select_gaussian(monkeypatch):
    import sca3_compass.robustness_calibration_numeric_repair as repair
    def failed(*args,**kwargs):
        return OptimizeResult(success=False,fun=100.,x=np.array([.1,0.,np.log(5)]),nit=1,message='test failure')
    monkeypatch.setattr(repair,'minimize',failed)
    fit,diag=repair.fit_calibration_with_diagnostics(np.random.default_rng(77).normal(size=(32,6)))
    assert fit.gaussian_bic_selected and not fit.converged
    assert not diag['student_any_success']


def test_outside_gaussian_solution_uses_original_bounded_solver(monkeypatch):
    import sca3_compass.robustness_calibration_numeric_repair as repair
    calls=[]
    def numerical(fun,x0,**kwargs):
        calls.append(kwargs)
        gaussian=len(x0)==2
        return OptimizeResult(success=True,fun=2. if gaussian else 3.,
            x=np.array([.995,0.]) if gaussian else np.array([.99,0.,np.log(5)]),nit=1,message='fixture')
    monkeypatch.setattr(repair,'minimize',numerical)
    # Strong common direction makes the unconstrained rho exceed .995.
    rng=np.random.default_rng(57)
    x=rng.normal(size=(32,1))+1e-4*rng.normal(size=(32,6))
    fit,diag=repair.fit_calibration_with_diagnostics(x)
    assert not diag['gaussian_analytic_interior'] and fit.converged
    assert len(calls)==3
    assert calls[-1]['bounds']==[(-.95/5,.995),(-7,5)]
    assert calls[-1]['options']=={'maxiter':80,'ftol':1e-10}


def test_failed_gaussian_comparison_is_not_hidden(monkeypatch):
    import sca3_compass.robustness_calibration_numeric_repair as repair
    monkeypatch.setattr(repair,'gaussian_interior_solution',lambda *args:None)
    def numerical(fun,x0,**kwargs):
        gaussian=len(x0)==2
        return OptimizeResult(success=not gaussian,fun=100. if gaussian else 1.,
            x=np.array([.1,0.]) if gaussian else np.array([.1,0.,np.log(5)]),nit=1,message='fixture')
    monkeypatch.setattr(repair,'minimize',numerical)
    fit,diag=repair.fit_calibration_with_diagnostics(np.random.default_rng(6).normal(size=(32,6)))
    assert not fit.gaussian_bic_selected and not fit.converged and not diag['gaussian_success']
