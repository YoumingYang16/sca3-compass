"""One deterministic engineering fixture, not independent confirmation."""
from datetime import datetime,timezone
import time
import numpy as np
from threadpoolctl import threadpool_limits
from r4_common import model,PHASE,DEPENDENCIES,write,sha
from r4_model import evaluate

if __name__=='__main__':
    start=time.perf_counter()
    ss=np.random.SeedSequence(202609181226).spawn(3)
    z=np.random.default_rng(ss[0]).normal(size=(32,4,6))
    cs=np.random.default_rng(ss[1]).normal(size=(32,4,6))
    ct=np.random.default_rng(ss[2]).normal(size=(4,4,6))
    with threadpool_limits(1):
        expected=model.evaluate(z,ct,seed=7701,reference_draws=4095)
        actual=evaluate(z,cs,ct,seed=7701,mode='target_only')
    if actual['status']!='completed': raise RuntimeError(actual)
    arrays={}
    for group in ['evidence','p','references','decisions']:
        for k,v in expected[group].items():
            assert np.array_equal(v,actual[group][k]),(group,k)
            arrays[group+'_'+k]=v
    dest=PHASE/'checks/baseline-intermediates.npz';dest.parent.mkdir(exist_ok=True)
    with dest.open('xb') as f: np.savez_compressed(f,z=z,calibration_source=cs,calibration_target=ct,**arrays)
    write(PHASE/'checks/baseline-identity.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'status':'EXACT_TARGET_ONLY_IDENTITY','scope':'ENGINEERING_FIXTURE_NOT_CONFIRMATION',
        'reference_draws':4095,'N_target':4,'G':32,'seconds':time.perf_counter()-start,
        'arrays_sha256':sha(dest),'dependencies':DEPENDENCIES,'folds':actual['folds'],
        'source_sha256':{k:sha(PHASE/k) for k in ['baseline_check.py','r4_common.py','r4_model.py']}})
    print('EXACT_TARGET_ONLY_IDENTITY: all p/e/reference/decision arrays; seconds',time.perf_counter()-start)
