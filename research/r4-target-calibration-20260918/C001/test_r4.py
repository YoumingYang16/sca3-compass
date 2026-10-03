import json,inspect
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from r4_common import model,fc,pb,DEPENDENCIES,ROOT,verify_runtime
from r4_model import evaluate,load_observed
from r4_kernel import evaluate_kernel
from two_bank_reference import reference
from target_calibration import scale

@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(1): yield

def fixture(seed=731):
    streams=np.random.SeedSequence(seed).spawn(3)
    z,cs,ct=[np.random.default_rng(s).normal(size=(n,4,6)) for s,n in zip(streams,[32,32,4])]
    z[:8,:3]+=3.5
    return z,cs,ct

@pytest.mark.parametrize('weight,bound',[(0,1.),(0,5.),(1,1.)])
def test_endpoint_exact(weight,bound):
    z,cs,ct=fixture();used=ct if weight else cs
    a=model.evaluate(z,used,seed=911,reference_draws=4095,mismatch_bound=bound)
    b=evaluate_kernel(z,cs,ct,seed=911,weight=weight,bound=bound)
    assert a['status']==b['status']=='completed'
    for key,old in [('p',a['p']['R3_main']),('e',a['evidence']['R3_main']),
            ('reference',a['references']['R3_main']),('decision',a['decisions']['R3_main']),
            ('ordinary_e',a['evidence']['R3_ordinary'])]:
        assert np.array_equal(b[key],old),key
    for x,y in zip(a['folds'],b['folds']):
        assert np.array_equal(x['inference_shape'],y['inference_shape'])
        for xx,yy in zip(x['directions'],y['directions']):
            assert np.array_equal(xx['direction'],yy['direction'])

def test_observed_statistic_endpoints():
    z,cs,ct=fixture()
    for w in [0.,1.]:
        used=ct if w else cs;k,_,_=scale(cs,ct,w,1.)
        assert k==model.geometric_kappa(used)
        h=fc.shape_fit(z[:8]@fc.contrasts(6),2);a=np.array([.2,.3,.5,0.])
        y=z@fc.contrasts(6)
        energy=np.einsum('gik,ij,gjk->g',y,np.linalg.inv(h),y)
        s=z.mean(-1)@a/np.sqrt(k*energy*(a@h@a)/20)
        s_old=z.mean(-1)@a/np.sqrt(model.geometric_kappa(used)*energy*(a@h@a)/20)
        assert np.array_equal(s,s_old)

def test_same_H_joint_reference_and_pooled_error():
    ref,tuples=reference(131,32,32,4,4/36,255,audit=True)
    raw=[]
    for t in tuples:
        assert np.array_equal(np.linalg.eigvalsh(t['H']),t['eigen'])
        d=t['B']*t['H'][:,0,0]*np.sum(t['U']/t['eigen'],axis=1)/20
        assert np.array_equal(d,t['denominator2'])
        raw.extend(t['normal']/np.sqrt(d))
    assert np.array_equal(np.sort(raw),ref)
    assert np.array_equal(ref,model.references(131,32,36,255)['R3_main'])

def test_delta1_pool_identity():
    _,cs,ct=fixture();w=4/36;k,_,_=scale(cs,ct,w,1.)
    assert np.isclose(k,model.geometric_kappa(np.concatenate([cs,ct])),rtol=2e-15)

def test_PILOT_does_not_use_external_scale():
    z,cs,ct=fixture()
    a=evaluate_kernel(z,cs,ct,seed=72,reference_draws=255,weight=4/36,bound=1.)
    b=evaluate_kernel(z,cs,ct,seed=72,reference_draws=255,weight=4/36,bound=5.)
    for x,y in zip(a['folds'],b['folds']):
        assert np.array_equal(x['gamma'],y['gamma']) and x['calibrators']==y['calibrators']
        assert np.array_equal(x['direction_profiles'],y['direction_profiles'])
    assert np.array_equal(a['reference'],b['reference'])
    assert np.all(b['e']<=a['e'])

def test_grid_and_no_truth_inputs(tmp_path):
    z,cs,ct=fixture();b=evaluate(z,cs,ct,seed=51,reference_draws=255,mode='bridge_count',external_drift_bound=3.)
    ranks=b['p']*256
    assert np.all(ranks==np.floor(ranks))
    assert np.all((ranks[:,:,0]==256)|(ranks[:,:,0]%3==0))
    for key in ['truth','shape','kappa','delta','radii','mystery']:
        p=tmp_path/(key+'.npz');np.savez(p,z=z,calibration_source=cs,**{key:1})
        with pytest.raises(ValueError): load_observed(p)
    assert not {'truth','shape','kappa','delta','radii'} & set(inspect.signature(evaluate).parameters)

def test_missing_target_and_nt1_rejected():
    z,cs,ct=fixture()
    with pytest.raises(ValueError): evaluate(z,cs,seed=1,mode='target_only')
    with pytest.raises(ValueError): evaluate(z,cs,ct[:1],seed=1,mode='target_only')
    with pytest.raises(ValueError): evaluate(z,cs,ct,seed=1,mode='bridge_count')

def test_source_binding():
    assert verify_runtime()==DEPENDENCIES
    assert '/C001/' in str(model.__file__).replace('\\','/')

def test_fail_closed():
    z,cs,ct=fixture();ct[:]=0
    out=evaluate_kernel(z,cs,ct,seed=1,reference_draws=255,weight=4/36,bound=3.)
    assert out['status']=='conservative_numerical_failure'
    assert not out['decision'].any() and not out['e'].any()

def test_strong_infinite_df_diagnostic_serialization():
    from r4_experiment import diagnostic_json
    value={'df':float('inf'),'finite':np.float64(2.),'array':np.array([1.,np.inf])}
    converted=diagnostic_json(value)
    assert converted['df']=={'nonfinite_float':'positive_infinity'}
    assert converted['finite']==2. and value['df']==float('inf')
    json.dumps(converted,allow_nan=False)

def test_independent_generator_streams():
    from r4_simulation import generate
    from r4_common import read,PHASE
    p=read(PHASE/'protocol-dev.json');a,_,s=generate(p,0,0)
    p['cases'][0]['nt']=12;b,_,t=generate(p,0,0)
    assert np.array_equal(a['z'],b['z'])
    assert np.array_equal(a['calibration_source'],b['calibration_source']) and s==t
