"""Observed-data-only research CLI. No clinical or data-design certification."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,sys,time
import numpy as np
from threadpoolctl import threadpool_limits
from r4_common import write,sha,verify_runtime
from r4_model import evaluate,load_observed,VERSION

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',required=True,choices=['target_only','source_matched','source_bound','bridge_count'])
    p.add_argument('--external-bound',type=float);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--acknowledge-model',action='store_true')
    a=p.parse_args()
    if not a.acknowledge_model: p.error('explicit model/independence acknowledgement required; not verified by software')
    inp=Path(a.input).resolve();out=Path(a.out).resolve()
    if out.exists(): raise FileExistsError('output must be a new directory')
    observed=load_observed(inp)
    begin=time.perf_counter()
    with threadpool_limits(1):
        result=evaluate(**observed,seed=a.seed,mode=a.mode,external_drift_bound=a.external_bound)
    out.mkdir(parents=True)
    if a.mode=='bridge_count':
        arrays={k:result[k] for k in ['p','e','ordinary_e','decision','ordinary_decision','reference']}
        details={k:v for k,v in result.items() if k not in arrays and k!='reference_tuples'}
    else:
        arrays={group+'_'+k:v for group in ['p','evidence','references','decisions'] for k,v in result[group].items()}
        details={k:v for k,v in result.items() if k not in ['p','evidence','references','decisions']}
    with (out/'evidence.npz').open('xb') as f: np.savez_compressed(f,**arrays)
    write(out/'result.json',details)
    write(out/'invocation.json',{'utc':datetime.now(timezone.utc).isoformat(),'argv':sys.argv,
        'version':VERSION,'input_sha256':sha(inp),'evidence_sha256':sha(out/'evidence.npz'),
        'result_sha256':sha(out/'result.json'),'dependencies':verify_runtime(),
        'seconds':time.perf_counter()-begin,'input_is_not_patient_certification':True})
    print(result['status'],str(out))

if __name__=='__main__': main()
