import numpy as np
import pytest
from sca3_compass.robustness_bootstrap_guard import components,variance,bootstrap_bank
from sca3_compass.robustness_methods import contrasts

def model(rho=.8,scatter=1.,df=5.):
    shape=.65*np.ones((4,4))+.35*np.eye(4)
    return {'rho':rho,'shape':shape,'dimension':20,'prior':{'df':df,'scatter':scatter,'gaussian_bic_selected':np.isinf(df)}}

@pytest.mark.parametrize('df',[5.,np.inf])
def test_guard_dominates_plugin_and_duplicate_bank(df):
    rng=np.random.default_rng(817);z=rng.normal(size=(20,4,6));m=z.mean(-1);y=(z-m[...,None])@contrasts(6)
    theta=model(df=df);profiles=np.array([[2.,2.,0.,0.],[0.,0.,2.,2.]])
    p=components(m,y,profiles,theta,[theta,model(rho=.9,df=df)])
    same=components(m,y,profiles,theta,[theta,theta])
    for mode in p['plugin']:
        assert np.all(p['guard'][mode]>=p['plugin'][mode])
        np.testing.assert_array_equal(same['guard'][mode],same['plugin'][mode])

def test_composite_null_shift_monotonic_known_nuisance():
    rng=np.random.default_rng(818);z=rng.normal(size=(20,4,6));m=z.mean(-1);y=(z-m[...,None])@contrasts(6)
    theta=model();p=components(m,y,np.ones((2,4)),theta,[theta]);q=components(m-2,y,np.ones((2,4)),theta,[theta])
    for mode in p['plugin']:assert np.all(q['plugin'][mode][:,0]>=p['plugin'][mode][:,0])

def test_conditional_variance_definition():
    y=np.ones((7,4,5));theta=model();v,df=variance(y,theta)
    q=np.einsum('gsk,st,gtk->g',y,np.linalg.inv(theta['shape']),y)/.2
    np.testing.assert_allclose(v,(5+q)/25*(5/6));assert df==25


@pytest.mark.parametrize('df',[1.5,5.,np.inf])
def test_pipeline_scale_factorization_without_refitting(df):
    # Algebraic fixture only: isolates the calibration rho factor while the
    # contrast-scale eta^2, study shape, radial df and observations stay fixed.
    y=(np.arange(140,dtype=float).reshape(7,4,5)-69.5)/80
    eta_squared=.2
    for rho in [.1,.8,.95]:
        theta=model(rho=rho,scatter=eta_squared/(1-rho),df=df)
        actual,conditional_df=variance(y,theta)
        energy=np.einsum('gsk,st,gtk->g',y,np.linalg.inv(theta['shape']),y)
        kappa=(1+5*rho)/(6*(1-rho))
        expected=(np.full(len(y),kappa*eta_squared) if np.isinf(df)
                  else kappa*(df*eta_squared+energy)/(df+20))
        np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
        assert conditional_df==(np.inf if np.isinf(df) else df+20)

def test_fixed_directions_no_bank_refit(monkeypatch):
    import sca3_compass.robustness_bootstrap_guard as mod
    original=mod.positive_direction;calls=[]
    def spy(profile,shape):calls.append(shape.copy());return original(profile,shape)
    monkeypatch.setattr(mod,'positive_direction',spy)
    theta=model();other=model();other['shape']=np.eye(4)
    rng=np.random.default_rng(8);y=rng.normal(size=(10,4,5));components(rng.normal(size=(10,4)),y,np.ones((2,4)),theta,[theta,other])
    assert len(calls)==16
    assert all(np.allclose(c,.65*np.ones((3,3))+.35*np.eye(3)) for c in calls)

def test_failed_bootstrap_disables_only_affected_fold(monkeypatch):
    import sca3_compass.robustness_bootstrap_guard as mod
    calls=[];theta=model(df=np.inf)
    def bank(*args):
        calls.append(1);ok=len(calls)>1
        return theta,[theta],[{'status':'completed' if ok else 'failed'}],ok
    monkeypatch.setattr(mod,'bootstrap_bank',bank)
    monkeypatch.setattr(mod,'learn_profiles',lambda *a:(np.ones((2,4)),np.ones(2),{'fallback':False}))
    rng=np.random.default_rng(83);z=50+rng.normal(size=(16,4,6));cal=rng.normal(size=(4,16,6))
    out,info=mod.evaluate(z,cal,draws=1)
    assert info['folds'][0]['fallback'];assert not info['folds'][1]['fallback']
    assert np.all(info['held_pvalues']['guard_projection'][::2]==1)
    assert np.all(out['R1B_guard_pilotc0.5_projection_gate_eBH'][::2]==0)
    assert np.any(out['R1B_guard_pilotc0.5_projection_gate_eBH'][1::2]>0)

def test_held_gene_change_cannot_change_its_training(monkeypatch):
    import json
    import sca3_compass.robustness_bootstrap_guard as mod
    from sca3_compass.robustness_pattern_test import _json_value
    rng=np.random.default_rng(81743);z=rng.normal(size=(32,4,6));cal=rng.normal(size=(4,32,6))
    _,a=mod.evaluate(z,cal,draws=1,seed=72)
    changed=z.copy();changed[::2]*=3.;changed[::2,:,0]+=5.
    _,b=mod.evaluate(changed,cal,draws=1,seed=72)
    assert json.dumps(_json_value(a['folds'][0]),sort_keys=True)==json.dumps(_json_value(b['folds'][0]),sort_keys=True)
