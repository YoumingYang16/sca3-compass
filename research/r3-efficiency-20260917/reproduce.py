"""Prespecified five saved-input R3/R2/strong exact replays, not new samples."""
from pathlib import Path
import sys,gzip,json,time
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent;OUT=BASE/'C001'
sys.path.insert(0,str(OUT))
from r3_experiment import validate
from r3_model import evaluate
from predictive_bridge import evaluate as evaluate_r2
from experiment import evaluate_v1
from r3_common import write,sha,verify_r2
from threadpoolctl import threadpool_limits
import numpy as np

def main():
    validate(OUT);started=time.perf_counter();checks=[]
    for c in [0,1,5,10,14]:
        path=OUT/f'raw/case-{c:02}/rep-00000.json.gz'
        with gzip.open(path,'rt',encoding='utf8') as f:row=json.load(f)
        if row['status']!='completed':raise ValueError('prespecified replay input failed; never substitute')
        if sha(OUT/row['input_path'])!=row['input_sha256'] or sha(OUT/row['evidence_path'])!=row['evidence_sha256']:raise ValueError('saved artifact changed')
        with np.load(OUT/row['input_path'],allow_pickle=False) as f:z=f['z'];cal=f['calibration']
        count=0
        with threadpool_limits(1),np.load(OUT/row['evidence_path'],allow_pickle=False) as saved:
            v=evaluate(z,cal,seed=row['algorithm_seed'])
            for field,prefix in [('p','p_'),('evidence','e_'),('references','reference_'),('decisions','decision_')]:
                for k,x in v[field].items():
                    if not np.array_equal(x,saved[prefix+k]):raise ValueError(f'replay mismatch:{c}:{prefix+k}')
                    count+=1
            b=evaluate_r2(z,cal,seed=row['algorithm_seed'])
            for k in ['PB_grid','PB_grid_ordinary']:
                for field,prefix in [('evidence','e_'),('decisions','decision_')]:
                    if not np.array_equal(b[field][k],saved[prefix+k]):raise ValueError('R2 replay mismatch')
                    count+=1
            old=evaluate_v1(z,cal.transpose(1,0,2),seed=row['algorithm_seed'],acknowledge_scope=True)
            for k in ['B_strong','K_NR']:
                if not np.array_equal(old['discoveries'][k],saved['decision_'+k]):raise ValueError('V1 replay mismatch')
                count+=1
            if c==14:
                for label,fn,key in [('R3_Delta5',evaluate,'R3_main'),('R2_Delta5',evaluate_r2,'PB_grid')]:
                    extra=fn(z,cal,seed=row['algorithm_seed'],mismatch_bound=5.)
                    for field,prefix in [('evidence','e_'),('decisions','decision_')]:
                        if not np.array_equal(extra[field][key],saved[prefix+label]):raise ValueError('Delta replay mismatch')
                        count+=1
        checks.append({'case':c,'rep':0,'identical_arrays':count,'source_record_sha256':sha(path)})
        print(checks[-1],flush=True)
    verify_r2()
    write(BASE/'REPRODUCTION_AUDIT.json',{'status':'EXACT_REPLAY_PASS','checks':checks,'seconds':time.perf_counter()-started,
          'freeze_sha256':sha(OUT/'freeze.json'),'index_sha256':sha(OUT/'index.json'),
          'scope':'Five predetermined saved families; not additional independent repetitions.'})

if __name__=='__main__':main()
