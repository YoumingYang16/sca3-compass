from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.stats import false_discovery_control
from sca3_compass import molecular_v1 as v1
from sca3_compass import robustness_methods as rm, robustness_bootstrap_guard as bg, robustness_calibration_efficiency as ce
from sca3_compass.robustness_calibration_numeric_repair import fit_calibration
from sca3_compass.molecular_methods import fdr_adjust,partial_conjunction,ebh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_r2_numeric_regression import repaired_aliases
from robustness_r1_window import read
ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('ack',[False,None,1])
def test_scope_requires_literal_ack(ack):
    with pytest.raises(ValueError):v1.check_inputs(np.ones((256,4,6)),np.ones((4,4,6)),ack)


@pytest.mark.parametrize('shape',[(255,4,6),(256,3,6),(256,4,5)])
def test_unvalidated_dimensions_refused(shape):
    with pytest.raises(ValueError):v1.check_inputs(np.ones(shape),np.ones((4,4,6)),True)


def test_no_truth_api_and_no_degenerate_input():
    import inspect
    assert 'truth' not in inspect.signature(v1.evaluate).parameters
    with pytest.raises(ValueError):v1.check_inputs(np.ones((256,4,6)),np.ones((4,4,6)),True)
    z=np.ones((256,4,6));z[0,0,0]=np.inf
    with pytest.raises(ValueError):v1.check_inputs(z,np.ones((4,4,6)),True)


@pytest.mark.parametrize('case,rep',[(0,0),(27,0),(77,0),(80,465),(81,500)])
def test_exact_repaired_existing_K_parity(case,rep):
    ip=ROOT/f'artifacts/robustness/R0076-repetitions/case-{case:04}/rep-{rep:06}-input.npz'
    with np.load(ip,allow_pickle=False) as data:z,cal=data['z'],data['calibration']
    seed=int(np.random.SeedSequence([7605229,case,rep,913]).generate_state(1)[0])
    result=v1.evaluate(z,cal,seed=seed,acknowledge_scope=True)
    with repaired_aliases([rm,bg,ce],fit_calibration):
        original,diag=ce.evaluate(z,cal,seed=seed,draws=16,mode='both')
    assert np.array_equal(result['e_values']['K_NR'],original['R2K_pilotc0.5_projection_gate_eBH'])
    assert np.array_equal(result['e_values']['B_strong'],original['R2P_pilotc0.5_support_simes_by_eBH'])
    assert np.array_equal(result['e_values']['B_fair_conditional_eBH'],original['R2K_pilotc0.5_ordinary_bonf_eBH'])
    assert np.array_equal(result['p_values']['B_fair_conditional_BY'],diag['held_pvalues']['K_ordinary_bonf'])
    for mode in ['projection','ordinary_bonf']:
        assert np.array_equal(result['p_values']['K_'+mode],diag['held_pvalues']['K_'+mode])
    gaussian=np.minimum(1,6*__import__('scipy').special.ndtr(-np.stack((z,-z),1)).min(-1))
    assert np.array_equal(result['p_values']['B_original'],partial_conjunction(gaussian,2))
    for m in ['B_original','B_fair_plugin','B_fair_guard','B_common']:
        p=result['p_values'][m]
        assert np.allclose(fdr_adjust(p).ravel(),false_discovery_control(p.ravel(),method='by'),atol=2e-15,rtol=2e-15)
    assert np.all(result['p_values']['B_fair_guard'] >= result['p_values']['B_fair_plugin'])
    assert result['diagnostics']['untestable_assumptions_verified'] is False
    assert result['evidence_level']=='EMPIRICAL_ONLY'


def test_failed_resample_not_silently_dropped(monkeypatch):
    ip=ROOT/'artifacts/robustness/R0076-repetitions/case-0027/rep-000000-input.npz'
    with np.load(ip,allow_pickle=False) as data:z,cal=data['z'],data['calibration']
    real=v1._fit;calls=[]
    def fitter(x):
        calls.append(1)
        if len(calls)>1:raise ArithmeticError('explicit test fixture failure')
        return real(x)
    monkeypatch.setattr(v1,'_fit',fitter)
    answer=v1.evaluate(z,cal,acknowledge_scope=True)
    assert answer['status']=='COMPUTED_WITH_DECLARED_FOLD_FALLBACK'
    assert not answer['discoveries']['K_NR'].any()
    assert not answer['discoveries']['B_fair_guard'].any()
    assert not answer['discoveries']['B_common'].any()
    assert all(not f['guard_success'] for f in answer['diagnostics']['folds'])


def test_singleton_PC_null_not_global_only():
    p=np.array([[1e-100,.4,.5,.9]])
    assert partial_conjunction(p,2)[0]==1.
