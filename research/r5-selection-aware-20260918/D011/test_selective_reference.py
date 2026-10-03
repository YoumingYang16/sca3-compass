import numpy as np
import pytest
from selective_reference import selected_errors,reference

class NormalLaw:
    def __init__(self,sd):self.sd=sd
    def draw(self,rng,n):return rng.normal(scale=self.sd,size=n),{'fixture':'GAUSSIAN_DIAGNOSTIC_ONLY'}

@pytest.mark.parametrize('borrow',[True,False])
def test_selected_pairs_exact_membership_and_gaussian_bridge_independence(borrow):
    vs=.1;vt=.7;a=vt/(vs+vt);c=np.sqrt(vs+vt)
    x,y,rec=selected_errors(NormalLaw(np.sqrt(vs)),NormalLaw(np.sqrt(vt)),0,0,c,borrow,
        [np.random.default_rng(214),np.random.default_rng(512)],50000)
    assert np.all((x-y<=c)==borrow)
    bridge=a*x+(1-a)*y;variance=vs*vt/(vs+vt)
    assert abs(bridge.mean())<6*np.sqrt(variance/len(x))
    assert abs(bridge.var()/variance-1)<.04
    assert rec['retained']==50000

def test_cap_does_not_substitute_unconditional_pairs():
    with pytest.raises(RuntimeError,match='selected-pair cap') as caught:
        selected_errors(NormalLaw(1),NormalLaw(1),0,0,1,True,
            [np.random.default_rng(1),np.random.default_rng(2)],100,cap=1)
    assert caught.value.receipt['attempted_pairs']==1
    assert caught.value.receipt['required']==100

def test_actual_conditional_bank_reference_and_shape_pairing():
    r=np.random.default_rng(978);cs=r.normal(size=(8,4,6));ct=r.normal(size=(4,4,6))
    p,rec=reference(37,32,cs,ct,3,outer_draws=127,meta_draws=255,audit=True)
    assert np.all((rec['es']-rec['et']<=p.c)==p.borrow_observed)
    assert not len(rec['inner_target']) and not len(rec['inner_bridge'])
    for x in rec['outer_shape_tuples']:assert np.allclose(np.linalg.eigvalsh(x['H']),x['eigen'])
    assert np.isfinite(p.pvalues(np.array([0.,1.,100.]),p.borrow_observed)).all()
