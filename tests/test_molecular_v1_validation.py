from pathlib import Path
import copy
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from molecular_v1_validation import validate_plan,score,METHODS,VERSION,OUTSIDE


def plan():
    return {'counts':[512]*54+[1024]*30,'error_allocation':{'paired':.02,'fdr':.02,'local':.01},
            'report_budget':.05,'candidate_version':VERSION,'methods':list(METHODS),
            'outside_indices':OUTSIDE,'confirmation_attempts':1,'seed':78091711}


def test_valid_fixed_plan():
    validate_plan(plan())


@pytest.mark.parametrize('bad',[True,1,2.5,-1])
def test_reject_bad_count(bad):
    p=plan();p['counts'][0]=bad
    with pytest.raises(ValueError):validate_plan(p)


def test_reject_unequal_core():
    p=plan();p['counts'][0]+=1
    with pytest.raises(ValueError):validate_plan(p)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-.1,0.,.04,True])
def test_budget_must_be_positive_bounded_finite(bad):
    p=plan();p['error_allocation']['paired']=bad
    with pytest.raises(ValueError):validate_plan(p)


def test_no_extra_attempt_or_scope_change():
    for key,value in [('confirmation_attempts',2),('outside_indices',[]),('candidate_version','fake'),('methods',['K_NR'])]:
        p=plan();p[key]=value
        with pytest.raises(ValueError):validate_plan(p)


def test_signed_null_power_undefined_and_fdp_not_pooled():
    truth=np.zeros((4,2),bool);rej=np.zeros_like(truth);rej[0,0]=True
    assert score(rej,truth)=={'fdp':1.,'power':None,'tp':0,'fp':1,'discoveries':1}
    truth[1,1]=True;rej[1,1]=True
    assert score(rej,truth)['power']==1.
    assert score(rej,truth)['fdp']==.5


def test_release_receipt_cannot_promote_unaccepted_version(tmp_path):
    import json
    from molecular_v1 import verify_release,EvidenceError
    path=tmp_path/'receipt.json'
    path.write_text(json.dumps({'method_version':VERSION,'decision':'V0'}),encoding='utf-8')
    with pytest.raises(EvidenceError):verify_release(path)
