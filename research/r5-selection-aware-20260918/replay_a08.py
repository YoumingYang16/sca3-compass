"""One actual fresh-reference API call; verifies stored D013 deterministic replay."""
import gzip,json
import numpy as np
from r5_common import PHASE,R4,read,write,sha
from r5_kernel import evaluate

if __name__=='__main__':
    source=PHASE/'D007/raw/case-01/rep-00000.json.gz'
    row=json.load(gzip.open(source,'rt',encoding='utf8'));arts=row['source_artifacts']
    with np.load(R4/'C001'/arts['observed']['path'],allow_pickle=False) as f:obs=dict(f)
    result=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=4.,seed=row['algorithm_seed'],
                    reference_draws=4095,inner_draws=4095,method='soft_closure')
    assert result['status']=='completed',result.get('error')
    evidence=PHASE/'D013/raw/case-01-rep-00000.npz'
    with np.load(evidence,allow_pickle=False) as f:
        equality={k:bool(np.array_equal(result[k],f[k])) for k in ['p','e','decision']}
    assert all(equality.values()),equality
    write(PHASE/'checks/a08-actual-api-replay.json',{'status':'ONE_INPUT_BITWISE_REPLAY_PASS','equality':equality,
        'script_sha':sha(__file__),'source_record_sha':sha(source),'D013_evidence_sha':sha(evidence),
        'API':'r5_kernel.evaluate(...method=soft_closure, reference_draws=4095,inner_draws=4095)',
        'seconds':result['seconds'],'reference_seconds':result['reference_seconds'],'G3':'NOT_PASSED',
        'limitations':'one same-input reproducibility check, not independent confirmation or new statistical replicate'})
    print(equality,result['seconds'])
