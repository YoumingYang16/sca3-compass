"""Training-only floor sensitivity on preserved SCALE fixtures; NOT confirmation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_continuous_scale import fit_continuous_scale
from sca3_compass.robustness_io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),root/'src/sca3_compass/robustness_continuous_scale.py',
        root/'src/sca3_compass/robustness_io.py']
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in paths:
        target=args.output/'source'/p.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    write_json(args.output/'protocol.json',{'phase':'DEVELOPMENT_SCALE_SENSITIVITY_NOT_FDR',
        'floors':[.1,.25,.5],'source_sha256':hashes,'case_indices':[2,3,10],
        'noise_fixture_seed':80317,'interpretation':'assumed lower null mass, never true labels supplied to fitting'})
    fixtures=[]
    for i in [2,3,10]:
        path=root/f'artifacts/robustness/continuous-floor-generator-dev1/case-{i:02}-inputs.json'
        value=json.loads(path.read_text(encoding='utf-8'))
        fixtures.append((f'generator_case{i}',value))
    noise=np.random.default_rng(80317).normal(size=(128,4))*3
    fixtures.append(('gaussian_noise9',{'means':noise.tolist(),'base_variance':[1.]*128,
        'shape':np.eye(4).tolist(),'df':None}))
    with threadpool_limits(limits=1):
        for name,value in fixtures:
            write_json(args.output/(name+'-input.json'),value)
            for floor in [.1,.25,.5]:
                start=time.perf_counter()
                result=fit_continuous_scale(np.asarray(value['means']),np.asarray(value['base_variance']),
                    np.asarray(value['shape']),np.inf if value['df'] is None else value['df'],
                    minimum_null_weight=floor)
                write_json(args.output/f'{name}-floor{floor}.json',{'result':result,'seconds':time.perf_counter()-start})
                print(name,floor,result['variance_multiplier'],result['converged'],result['weights'][0],flush=True)
    for p in paths:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=hashes[str(p.relative_to(root))]:
            raise RuntimeError('Source changed during fixed sensitivity run')


if __name__=='__main__':main()
