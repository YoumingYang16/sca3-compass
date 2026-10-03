"""Deterministic/theorem/formal-plan tests; no fresh scientific C2 samples."""
import copy
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from sca3_compass.robustness_paired_bounds import paired_interval
from robustness_r2_confirmation import joint_evaluate,validate,one,initialize
from robustness_r2_confirmation_plan import build
from robustness_r2_development import read
from sca3_compass.robustness_calibration_efficiency import evaluate
from threadpoolctl import threadpool_limits


def test_empirical_bernstein_affine_constants_and_non_iid_mean():
    a=np.array([-.2,.3,.1,.4]);b=np.array([.4,.1,-.3,.2]);delta=.01
    x=np.r_[a,b];y=(x+1)/2;n=len(y);log=np.log(2/delta)
    radius=2*(np.sqrt(2*y.var(ddof=1)*log/n)+7*log/(3*(n-1)))
    r=paired_interval([a,b],delta)
    assert r['mean_difference']==pytest.approx((a.mean()+b.mean())/2)
    assert r['radius']==pytest.approx(radius)
    assert r['n_independent_families']==8


@pytest.mark.parametrize('arrays',[[np.arange(3)], [np.array([0,np.nan])],
    [np.ones(2),np.ones(3)], [np.ones((2,2))], [np.zeros(1)], []])
def test_invalid_or_unequal_family_arrays_rejected(arrays):
    with pytest.raises(ValueError):paired_interval(arrays,.01)


def test_pair_interval_symmetry_and_large_n_precision():
    x=np.tile([0.,.02],10000)
    a=paired_interval([x],.001);b=paired_interval([-x],.001)
    assert a['lower']>0
    assert a['lower']==pytest.approx(-b['upper'])
    assert a['radius']<.003


def test_pre_C2_plan_preserves_H_all_frontends_error_counts_and_scope():
    # Once C2 is frozen inspect the saved plan, NEVER redesign it.
    path=ROOT/'configs/robustness_C2_analysis.json'
    p=read(path) if path.exists() else build()
    validate(p)
    old=read(ROOT/'configs/robustness_C1_analysis.json')
    assert p['families']['H']==old['families']['H']
    assert len(p['families']['F_all_same_information'])==72
    assert p['original_validity_scope_indices']==old['validity_scope_indices']
    assert len(p['expanded_matched_indices'])==8
    assert p['seed'] not in [7205107,7305119,7505173]
    invalid=copy.deepcopy(p);invalid['error_allocation']['I_strata']+=.001
    with pytest.raises(ValueError):validate(invalid)


@pytest.mark.parametrize('case,rep',[(0,0),(27,0),(59,0),(63,0),(75,0)])
def test_same_family_memoized_K_equals_fresh_and_original_R1_arrays(case,rep):
    path=ROOT/f'artifacts/robustness/R0072-repetitions/case-{case:04}/rep-{rep:06}.json.gz'
    if not path.exists():pytest.skip('Existing local DEV fixture unavailable')
    record=read(path)
    with np.load(record['input_path']) as z:z,cal=z['z'],z['calibration']
    seed=int(np.random.SeedSequence([7205107,case,rep,913]).generate_state(1)[0])
    with threadpool_limits(1):
        joint,d1,d2,reused=joint_evaluate(z,cal,seed)
        fresh,df=evaluate(z,cal,seed=seed,mode='K')
    assert len(joint)==124 and reused
    for name,values in fresh.items():np.testing.assert_array_equal(values,joint[name])
    for name,values in df['held_pvalues'].items():np.testing.assert_array_equal(values,d2['held_pvalues'][name])
    with np.load(record['evidence_path']) as old:
        for name in joint:
            if not name.startswith('R2K_'):np.testing.assert_array_equal(joint[name],old[name])


def test_incomplete_R1_bank_calls_fresh_K_not_cherry_picked_partial_bank(monkeypatch):
    import robustness_r2_confirmation as m
    monkeypatch.setattr(m,'_r0',lambda *a: ({},{}))
    monkeypatch.setattr(m,'reference',lambda *a,**kw: ({},{'folds':[{'guard_success':False,'bank':[1]}]}))
    called=[]
    def candidate(*a,**kw):called.append(kw);return {},{}
    monkeypatch.setattr(m,'candidate',candidate)
    _,_,_,reused=m.joint_evaluate(None,None,91)
    assert not reused and called[0]['cached_folds'] is None and called[0]['mode']=='K'


def test_runner_preserves_orphans_and_records_failure(tmp_path):
    p={'run_id':'RFIX','method_version':'unit','seed':1}
    initialize(tmp_path,p)
    folder=tmp_path/'artifacts/robustness/RFIX-repetitions/case-0000';folder.mkdir(parents=True)
    path=folder/'rep-000000-input.npz';path.write_bytes(b'preserve')
    out=one(0,0)
    assert out['status']=='failed' and path.read_bytes()==b'preserve'
    assert 'Orphan' in read(out['path'])['error']


def test_replay_selection_keeps_all_guard_failures_and_first_K_per_case():
    from robustness_r2_confirm_replay import selection
    p={'replay_plan':{'cases':[0,27],'reps':[0,1]}}
    r={'R1_failed_folds':[{'case':1,'rep':2},{'case':1,'rep':3}],
       'K_failed_folds':[{'case':7,'rep':9},{'case':7,'rep':3},{'case':8,'rep':4}]}
    assert selection(p,r)==[(0,0),(0,1),(1,2),(1,3),(7,3),(8,4),(27,0),(27,1)]
