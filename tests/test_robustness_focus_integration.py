"""Integration contracts, not acceptance or inferential validation."""
import json
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from sca3_compass import robustness_geometry as geometry
from sca3_compass import robustness_loading_focus as focus
from sca3_compass import robustness_loading_audit_gate as bulk
from test_robustness_loading_focus import mechanism_fixture


@pytest.mark.parametrize('kind',['sparse','downcov'])
@threadpool_limits.wrap(limits=1)
def test_focus_veto_has_expected_mechanism_without_truth(kind):
    x,_,_=mechanism_fixture(kind)
    diag={'fit':{'rho':.95 if kind=='downcov' else .8}}
    result,receipt=geometry.fit_fold_geometries(x,diag,audit_mode='focused_veto')
    assert all((value is not None)==(kind=='sparse') for value in result)
    for item in receipt['folds']:
        audit=item['raw_loading_audit']
        assert audit['focused_audit']['audit_valid']
        assert audit['veto_triggered']==(kind=='downcov')
        assert audit['final_unforced_adoption']==(kind=='sparse')
    json.dumps(receipt,allow_nan=False)


@threadpool_limits.wrap(limits=1)
def test_focus_geometry_excludes_own_outer_holdout():
    x,_,_=mechanism_fixture('sparse')
    diag={'fit':{'rho':.8}}
    _,before=geometry.fit_fold_geometries(x,diag,audit_mode='focused_veto')
    changed=x.copy();changed[::2]+=np.arange(6)*100
    _,after=geometry.fit_fold_geometries(changed,diag,audit_mode='focused_veto')
    assert before['folds'][0]==after['folds'][0]


@pytest.mark.parametrize('old_adopt',[False,True])
@pytest.mark.parametrize('bulk_negative',[False,True])
@pytest.mark.parametrize('focus_negative',[False,True])
@pytest.mark.parametrize('valid',[False,True])
@threadpool_limits.wrap(limits=1)
def test_gate_truth_table_without_real_labels(monkeypatch,old_adopt,bulk_negative,focus_negative,valid):
    x,_,_=mechanism_fixture('sparse')
    def raw(*args,**kwargs):
        return False,{'old_gate':{'adopt':old_adopt},'all_scores_finite':True,
            'all_inner_fits_converged':True,'mean_log_residual_gain':-.5 if bulk_negative else .1,
            'gene_level_standard_error':.01}
    monkeypatch.setattr(bulk,'raw_loading_gate',raw)
    monkeypatch.setattr(focus,'focused_loading_audit',lambda *args,**kwargs:
        {'audit_valid':valid,'negative_evidence':focus_negative,'positive_evidence':False})
    _,receipt=geometry.fit_fold_geometries(x,{'fit':{'rho':.8}},audit_mode='focused_veto')
    expected=old_adopt and valid and not(bulk_negative and focus_negative)
    assert all(item['fallback_to_fixed_geometry']==(not expected) for item in receipt['folds'])


@threadpool_limits.wrap(limits=1)
def test_raised_focus_error_is_preserved_and_fails_closed(monkeypatch):
    x,_,_=mechanism_fixture('sparse')
    def fail(*args,**kwargs):
        raise ValueError('intentional malformed audit')
    monkeypatch.setattr(focus,'focused_loading_audit',fail)
    result,receipt=geometry.fit_fold_geometries(x,{'fit':{'rho':.8}},audit_mode='focused_veto')
    assert result==[None,None]
    for item in receipt['folds']:
        failure=item['raw_loading_audit']['focused_audit']
        assert not failure['audit_valid'] and failure['numerical_failure_count']==1
        assert failure['error_message']=='intentional malformed audit'
