"""Run the frozen R3 method on explicit arrays, or one saved SIMULATED family."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'C001'))
from r3_experiment import validate
from r3_model import evaluate
from r3_common import sha
import numpy as np
from threadpoolctl import threadpool_limits

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=BASE/'C001/raw/case-02/rep-00000-input.npz')
    parser.add_argument('--seed',type=int,default=22)
    parser.add_argument('--output',type=Path)
    a=parser.parse_args();validate(BASE/'C001')
    with np.load(a.input,allow_pickle=False) as f:
        z=f['z'];cal=f['calibration']
    with threadpool_limits(1):result=evaluate(z,cal,seed=a.seed)
    report={'version':result['version'],'status':result['status'],
            'discoveries':int(result['decisions']['R3_main'].sum()),'seconds_with_ablations':result['seconds'],
            'input':str(a.input.resolve()),'input_sha256':sha(a.input),'seed':a.seed,
            'freeze_sha256':sha(BASE/'C001/freeze.json'),
            'scope':'Research only. Default saved data are SIMULATION, not patient data or biological findings. Input assumptions require external justification.'}
    if a.output:
        with a.output.open('x',encoding='utf8') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
