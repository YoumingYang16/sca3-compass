"""User-facing frozen-model invocation. Loads no truth labels."""
from pathlib import Path
import argparse
import sys
PHASE=Path(__file__).resolve().parent
sys.path.insert(0,str(PHASE/'C001'))
from predictive_bridge import evaluate
from provenance import verify_freeze
from experiment import write,sha
import numpy as np
from threadpoolctl import threadpool_limits


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True); p.add_argument('--output',required=True)
    p.add_argument('--seed',required=True,type=int); p.add_argument('--acknowledge-model',action='store_true')
    a=p.parse_args()
    if not a.acknowledge_model: raise ValueError('Read README input assumptions and explicitly acknowledge model scope')
    source=Path(a.input).resolve(); output=Path(a.output).resolve()
    if output.exists(): raise FileExistsError('preserve previous invocation')
    verify_freeze(PHASE/'C001')
    with np.load(source,allow_pickle=False) as data:
        # Deliberately no truth, true shape, true kappa or patient metadata.
        z,cal=data['z'],data['calibration']
    with threadpool_limits(1): result=evaluate(z,cal,seed=a.seed)
    artifacts=output.with_suffix('.npz')
    output.parent.mkdir(parents=True,exist_ok=True)
    with artifacts.open('xb') as stream:
        np.savez_compressed(stream,p=result['p'],e=result['evidence']['PB_grid'],decision=result['decisions']['PB_grid'],reference=result['reference'])
    write(output,{'version':result['version'],'status':result['status'],'seed':a.seed,
                  'input_sha256':sha(source),'frozen_manifest_sha256':sha(PHASE/'C001/freeze.json'),
                  'model_assumptions_acknowledged_not_verified':True,'discoveries':int(result['decisions']['PB_grid'].sum()),
                  'interpretation':'rejections, NOT proven real biological findings',
                  'artifact_path':str(artifacts),'artifact_sha256':sha(artifacts),
                  'error':result.get('error'),'seconds':result['seconds']})
    print(output,flush=True)


if __name__=='__main__': main()
