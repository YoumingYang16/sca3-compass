"""One preselected existing C1rep0 API correctness/cost check; no truth scoring."""
import gzip,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from r5_common import R4,PHASE,read,sha,write
from focused_kernel import evaluate

def main():
    prefix=PHASE/'checks/a010-api'
    files=['focused_kernel.py','focused_reference.py','joint_power_kernel.py','joint_power_reference.py',
           'ancillary_calibration.py','r5_common.py','FOCUSED_BUDGET.md']
    write(str(prefix)+'-protocol.json',{'source':'R4C001C1rep0','no_truth_access':True,
        'purpose':'bitwise replay, observed work/cost and required reference data; no Power/FDR claim',
        'n_calls':2,'outer_draws':8191,'meta_draws':4095,'stop':'two calls only, no sample expansion',
        'files':{f:sha(PHASE/f) for f in files}})
    rec=json.load(gzip.open(R4/'C001/raw/case-01/rep-00000.json.gz','rt',encoding='utf8'))
    path=R4/'C001'/rec['observed_path'];assert sha(path)==rec['observed_sha256']
    with np.load(path,allow_pickle=False) as f:obs=dict(f)
    D=read(R4/'C001/protocol.json')['cases'][1]['D']
    with threadpool_limits(1):
        a=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=D,seed=rec['algorithm_seed'])
        b=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=D,seed=rec['algorithm_seed'])
    same=all(np.array_equal(a['evidence'][k],b['evidence'][k]) for k in a['evidence'])
    result={'status':a['status'],'bitwise_e_replay':same,'seconds':[a['seconds'],b['seconds']],
        'no_truth_or_outcome_scoring':True,'unscored_rejections':{k:int(v.sum()) for k,v in a['decisions'].items()},
        'cover':a.get('reference',{}).get('cover'),'thresholds':a.get('reference',{}).get('thresholds'),
        'reference_totals':a.get('reference',{}).get('totals'),
        'failure_detail':{k:v for k,v in a.items() if k in ['error','stage','receipt']}}
    write(str(prefix)+'-result.json',result)
    with open(str(prefix)+'-evidence.npz','xb') as f:np.savez_compressed(f,**a['evidence'])
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
