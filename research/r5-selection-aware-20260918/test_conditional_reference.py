import numpy as np
from conditional_reference import reference
from r5_kernel import evaluate
from r5_common import fc

def test_conditional_reference_depends_on_residuals_not_kappa():
    r=np.random.default_rng(889);cs=r.normal(size=(8,4,6));ct=r.normal(size=(4,4,6))
    # Rescaling common means modifies kappa but leaves centered logA residuals.
    cs2=cs+2*cs.mean(-1,keepdims=True);ct2=ct+4*ct.mean(-1,keepdims=True)
    a,ra=reference(14,32,cs,ct,127,127);b,rb=reference(14,32,cs2,ct2,127,127)
    assert np.allclose(ra['ancillary_source'],rb['ancillary_source'])
    assert np.allclose(ra['ancillary_target'],rb['ancillary_target'])
    # Rejection sequence and samples coincide up to floating arithmetic.
    assert np.allclose(a.es,b.es) and np.allclose(a.et,b.et)

def test_conditional_endpoints_run_same_information():
    r=np.random.default_rng(8);z=r.normal(size=(32,4,6));cs=r.normal(size=(8,4,6));ct=r.normal(size=(4,4,6))
    for method in ['selective','closure','codesigned','target','bridge','source_bound','variance_bridge']:
        x=evaluate(z,cs,ct,seed=4,bound=3,reference_draws=127,inner_draws=127,method=method)
        assert x['status']=='completed'
        assert np.all(x['p']*128==np.round(x['p']*128))

def test_observable_low_precision_source_is_downweighted():
    def bank(logs):
        y=np.c_[np.eye(4),np.zeros(4)]@fc.contrasts(6).T
        blocks=[]
        for loga in logs:
            x=np.array([np.sqrt(2*np.exp(loga)),0,0,0])
            blocks.append(y+x[:,None])
        return np.array(blocks)
    source=bank([-10]*4+[5]*8);target=bank([0]*4)
    p,r=reference(891,32,source,target,127,1023)
    assert p.a<.2  # count-only would be.75, opposite information conclusion.
    assert r['meta']['source_variance_MC']>4*r['meta']['target_variance_MC']

def test_test_fold_does_not_change_its_selector_or_learners():
    r=np.random.default_rng(83);z=r.normal(size=(32,4,6));cs=r.normal(size=(8,4,6));ct=r.normal(size=(4,4,6))
    a=evaluate(z,cs,ct,seed=4,bound=3,reference_draws=127,inner_draws=127)
    changed=z.copy();changed[::4]+=2.
    b=evaluate(changed,cs,ct,seed=4,bound=3,reference_draws=127,inner_draws=127)
    assert a['status']==b['status']=='completed'
    assert a['calibration']==b['calibration']
    assert np.array_equal(a['folds'][0]['gamma'],b['folds'][0]['gamma'])
    assert a['folds'][0]['calibrators']==b['folds'][0]['calibrators']
    for field in ['direction_profiles','inference_shape']:
        assert np.array_equal(a['folds'][0][field],b['folds'][0][field])
