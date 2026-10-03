"""Regression for R0043's real near-stationary line-search failure."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_prior import fit_energy_prior
from sca3_compass.robustness_methods import evaluate_candidates,contrasts


def test_r43_known_failed_prior_is_certified_without_likelihood_change():
    script=Path(__file__).resolve().parents[1]/'scripts/robustness_screen.py'
    spec=importlib.util.spec_from_file_location('prior_recovery_generator',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rng=np.random.default_rng(np.random.SeedSequence([4303011,20]))
    case={'distribution':'normal','rho':.1,'n':256,'effect':4.5}
    with threadpool_limits(limits=1):
        for _ in range(59):
            z,x,_=module.data(rng,case)
        _,diag=evaluate_candidates(z,x,'frontier')
        shape=np.asarray(diag['shape'][1]['matrix'])
        residual=(z-z.mean(-1)[...,None])@contrasts(6)
        q=np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(shape),residual)/(1-diag['fit']['rho'])
        fit=fit_energy_prior(q[::2],20)
    assert not fit['gaussian_bic_selected']
    assert fit['converged'] and fit['selected_projected_score']<=1e-7
    assert abs(fit['objective']-3.332811250594152)<1e-10
    assert abs(fit['df']-57.33073615369306)<.001
    if fit['numerical_recovery'] is not None:
        assert fit['numerical_recovery']['accepted']
    json.dumps(fit,allow_nan=False)
