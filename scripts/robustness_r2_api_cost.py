"""Ten existing DEV inputs, three delivered APIs; engineering cost only.

One unmeasured warm-up per API, then one timed call per input/API with rotated
order. Single BLAS thread, process CPU and wall time. Concurrent host work may
affect wall times. R1 returns72 e-labels; P/K individually return36, so these
are delivered API costs, NOT isolated equally optimized primary-kernel costs.
"""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sca3_compass.robustness_bootstrap_guard import evaluate as r1
from sca3_compass.robustness_calibration_efficiency import evaluate as r2
from sca3_compass.robustness_io import write_json


def read(path):
    with (gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    base=ROOT/'artifacts/robustness'
    out=base/'R1R2-20260916/continuation-20260916-0153/R2-api-cost'
    out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'generate_source.py')
    protocol=read(base/'R0073-screen.protocol.json')
    for rel,digest in protocol['source_sha256'].items():
        if rel.startswith('src/') and sha(ROOT/rel)!=digest:
            raise ValueError('Frozen R1 source/dependency changed')
    calls={'R1':lambda z,c,s:r1(z,c,draws=16,seed=s),
           'P':lambda z,c,s:r2(z,c,draws=16,seed=s,mode='P'),
           'K':lambda z,c,s:r2(z,c,draws=16,seed=s,mode='K')}
    rows=[]
    inputs=[(case,rep) for case in [0,27,59,63,75] for rep in [0,1]]
    with threadpool_limits(1):
        for position,(case,rep) in enumerate(inputs):
            oldpath=base/'R0072-repetitions'/f'case-{case:04}'/f'rep-{rep:06}.json.gz'
            old=read(oldpath)
            krecord=read(base/'R0074-repetitions'/f'case-{case:04}'/f'rep-{rep:06}.json.gz')
            if sha(old['input_path'])!=old['input_sha256'] or sha(old['evidence_path'])!=old['evidence_sha256'] or sha(krecord['evidence_path'])!=krecord['evidence_sha256']:
                raise ValueError('Archived array changed')
            with np.load(old['input_path']) as data:
                z,cal=data['z'],data['calibration']
            seed=int(np.random.SeedSequence([7205107,case,rep,913]).generate_state(1)[0])
            if position==0:
                for function in calls.values():
                    function(z,cal,seed)
            names=list(calls)
            names=names[position%3:]+names[:position%3]
            for name in names:
                start,cpu=time.perf_counter(),time.process_time()
                values,_=calls[name](z,cal,seed)
                elapsed,used=time.perf_counter()-start,time.process_time()-cpu
                expected_path=krecord['evidence_path'] if name=='K' else old['evidence_path']
                with np.load(expected_path) as expected:
                    for label,array in values.items():
                        key=label.replace('R2P_','R1B_plugin_') if name=='P' else label
                        np.testing.assert_array_equal(array,expected[key])
                rows.append({'case_index':case,'rep':rep,'api':name,'returned_evidence_labels':len(values),
                             'cpu_seconds':used,'wall_seconds':elapsed,'all_output_arrays_exact':True,
                             'input_sha256':old['input_sha256']})
    summary={name:{metric+'_median':float(np.median([r[metric] for r in rows if r['api']==name]))
                   for metric in ['cpu_seconds','wall_seconds']} for name in calls}
    write_json(out/'summary.json',{'kind':'ENGINEERING_EXISTING_DEV_INPUTS_NOT_INDEPENDENT_EXPERIMENT',
        'complete':True,'input_families':len(inputs),'timed_calls':len(rows),'untimed_warmups':3,
        'source_sha256':{'r1':sha(ROOT/'src/sca3_compass/robustness_bootstrap_guard.py'),
                         'r2':sha(ROOT/'src/sca3_compass/robustness_calibration_efficiency.py'),
                         'script':sha(__file__)},'rows':rows,'summary':summary,
        'limits':['Delivered API cost, not equal-output optimized primary kernels','R1 returns72 e-labels; P/K individually36',
                  'Concurrent R0075 can affect wall timing','Ten existing inputs, no timing confidence interval or general speedup claim']})
    print(json.dumps(summary,indent=2))
    print(out)


if __name__=='__main__':
    main()
