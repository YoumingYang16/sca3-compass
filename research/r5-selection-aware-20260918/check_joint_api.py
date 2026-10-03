"""One fixed existing input, actual learnedH, reproducibility/cost only; no truth."""
import gzip,json
import numpy as np
from threadpoolctl import threadpool_limits
from r5_common import PHASE,R4,sha,write
from joint_power_kernel import evaluate

if __name__=='__main__':
    files=['joint_power_kernel.py','joint_power_reference.py','ancillary_calibration.py','selection_profile.py','r5_common.py','JOINT_MOMENT_BUDGET.md']
    write(PHASE/'checks/a09-api-code.json',{'files':{f:sha(PHASE/f) for f in files},
        'protocol':'same existing C1rep0, two identicalcalls, no truth, 64successes/endpoint, meta4095, numerator1024, cap1M'})
    source=R4/'C001/raw/case-01/rep-00000.json.gz';row=json.load(gzip.open(source,'rt',encoding='utf8'))
    path=R4/'C001'/row['observed_path'];assert sha(path)==row['observed_sha256']
    with np.load(path,allow_pickle=False) as f:obs=dict(f)
    kwargs=dict(bound=4.,seed=row['algorithm_seed'],successes=64,meta_draws=4095,numerator_draws=1024,proposal_cap=1000000)
    with threadpool_limits(1):
        a=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],**kwargs)
        b=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],**kwargs)
    exact={k:bool(np.array_equal(a[k],b[k])) for k in ['e','decision']}
    result={k:v for k,v in a.items() if k not in ['e','decision','pc_e']}
    result.update(reproducibility=exact,source_observed_sha=sha(path),source_record_sha=sha(source),
        algorithm_seed=row['algorithm_seed'],truth_read=False,stage='SINGLE_INPUT_CORRECTNESS_COST_NOT_UTILITY_ACCEPTANCE',
        unscored_rejections=int(a['decision'].sum()),max_e=float(a['e'].max()))
    dest=PHASE/'checks/a09-api-evidence.npz'
    with dest.open('xb') as f:np.savez_compressed(f,**{k:a[k] for k in ['e','decision','pc_e'] if k in a})
    result['evidence_sha']=sha(dest);write(PHASE/'checks/a09-api-result.json',result)
    print(a['status'],exact,a['seconds'],result['unscored_rejections'],result['max_e'],flush=True)
    if a['status']!='completed' or not all(exact.values()):raise RuntimeError('prototype not correct; inspect savedfailure')
