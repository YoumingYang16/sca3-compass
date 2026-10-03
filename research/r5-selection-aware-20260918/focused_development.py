"""D015 fixed20 old-family mechanism check; all versions frozen before run."""
import argparse,gzip,json,shutil,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from r5_common import PHASE,R4,sha,read,write,exp
from focused_kernel import evaluate

FILES=['focused_development.py','focused_kernel.py','focused_reference.py','joint_power_kernel.py','joint_power_reference.py',
       'ancillary_calibration.py','selection_profile.py','r5_common.py','test_focused_reference.py','FOCUSED_BUDGET.md',
       'JOINT_MOMENT_BUDGET.md','D015_RULE.md','R5_CLOSEST_WORK.md']

def prepare(out):
    out.mkdir(exist_ok=False)
    for f in FILES:shutil.copy2(PHASE/f,out/f)
    write(out/'protocol.json',{'id':'D015','kind':'BOUNDED_DEVELOPMENT_ONLY','cases':list(range(10)),
        'reps':[0,1],'outer_draws':8191,'meta_draws':4095,'workers':4,'safety_timeout_seconds':600,
        'new_observed_families':0,'formal_confirmation':False,'source_freeze':sha(R4/'C001/freeze.json')})
    write(out/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'files':{f:sha(out/f) for f in FILES+['protocol.json']}})

def verify(out):
    assert Path(__file__).resolve()==out/'focused_development.py'
    for f,h in read(out/'freeze.json')['files'].items():assert sha(out/f)==h
    for name in ['focused_reference','focused_kernel','joint_power_kernel','joint_power_reference']:
        assert Path(sys.modules[name].__file__).resolve()==out/(name+'.py')

def one(task):
    directory,c,rep=task;out=Path(directory);verify(out);start=time.perf_counter()
    source=R4/f'C001/raw/case-{c:02}/rep-{rep:05}.json.gz'
    old=json.load(gzip.open(source,'rt',encoding='utf8'));p=read(out/'protocol.json')
    row={'case':c,'rep':rep,'source_record_sha':sha(source),'freeze_sha':sha(out/'freeze.json'),
         'algorithm_seed':old['algorithm_seed'],'kind':'DEVELOPMENT_NOT_CONFIRMATION'};arrays={}
    try:
        artifacts={k:{'path':old[k+'_path'],'sha':old[k+'_sha256']} for k in ['observed','diagnostic','evidence']}
        for v in artifacts.values():assert sha(R4/'C001'/v['path'])==v['sha']
        with np.load(R4/'C001'/artifacts['observed']['path'],allow_pickle=False) as f:obs=dict(f)
        D=read(R4/'C001/protocol.json')['cases'][c]['D']
        with threadpool_limits(1):
            v=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=D,seed=old['algorithm_seed'],
                       outer_draws=p['outer_draws'],meta_draws=p['meta_draws'])
        # Truth is opened only after algorithm execution, for scoring.
        with np.load(R4/'C001'/artifacts['diagnostic']['path'],allow_pickle=False) as f:truth=f['truth']
        arrays.update({'e_'+k:e for k,e in v['evidence'].items()});dec=dict(v['decisions'])
        if v['status']=='completed':
            arrays.update({'reference_'+k:e for k,e in v['reference']['reference_arrays'].items()})
            arrays.update({'statistic_'+k:e for k,e in v['statistics'].items()})
            v['reference']={k:e for k,e in v['reference'].items() if k!='reference_arrays'}
        op=PHASE/f'D009/raw/case-{c:02}/rep-{rep:05}.json.gz';orec=json.load(gzip.open(op,'rt',encoding='utf8'))
        assert orec['source_artifacts']==artifacts
        ep=PHASE/'D009'/orec['evidence_path'];assert sha(ep)==orec['evidence_sha']
        with np.load(ep,allow_pickle=False) as f:
            for k in ['target','source_bound','bridge','fixed_e_mix','strong_target','strong_pool_bound','conditional_target','conditional_mix']:
                dec['old_'+k]=f['decision_'+k]
            dec['A07']=f['decision']
        arrays.update({'decision_'+k:d for k,d in dec.items()})
        row.update(status=v['status'],source_artifacts=artifacts,metrics={k:exp.score(d,truth) for k,d in dec.items()},
             candidate={k:e for k,e in v.items() if k not in ['evidence','decisions','statistics']},comparison_source_sha=sha(op))
    except Exception as err:row.update(status='hard_failure',error=repr(err),traceback=traceback.format_exc())
    dest=out/f'raw/case-{c:02}/rep-{rep:05}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    if arrays:
        ep=dest.with_name(dest.name.replace('.json.gz','-evidence.npz'))
        with ep.open('xb') as f:np.savez_compressed(f,**arrays)
        row.update(evidence_path=ep.relative_to(out).as_posix(),evidence_sha=sha(ep))
    row.update(seconds=time.perf_counter()-start,utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(dest,'xt',encoding='utf8') as f:json.dump(row,f,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
    return {'path':dest.relative_to(out).as_posix(),'sha':sha(dest),'case':c,'rep':rep,'status':row['status']}

def run(out):
    verify(out);p=read(out/'protocol.json');write(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat()});rows=[]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        fs=[pool.submit(one,(str(out),c,r)) for c in p['cases'] for r in p['reps']]
        for f in as_completed(fs,timeout=p['safety_timeout_seconds']):
            rows.append(f.result());print(len(rows),'/20',rows[-1]['status'],flush=True)
    write(out/'index.json',{'rows':sorted(rows,key=lambda r:(r['case'],r['rep']))})
    raw=[json.load(gzip.open(out/x['path'],'rt',encoding='utf8')) for x in rows]
    result=[]
    for c in p['cases']:
        rr=[x for x in raw if x['case']==c];means={}
        if all('metrics' in x for x in rr):
            means={k:{m:None if rr[0]['metrics'][k][m] is None else float(np.mean([v['metrics'][k][m] for v in rr]))
                       for m in ['power','fdp','tp','fp']} for k in rr[0]['metrics']}
        result.append({'case':c,'n':len(rr),'means':means,'statuses':[x['status'] for x in rr]})
        print(c,{k:v['power'] for k,v in means.items() if k in ['focused_joint','focused_target','focused_count_bridge','focused_fixed_mix','old_conditional_mix','A07']},flush=True)
    write(out/'summary.json',{'status':'BOUNDED_DIAGNOSTIC_COMPLETE_NOT_ACCEPTANCE','rows':result,
        'failures':[x for x in rows if x['status']!='completed'],'new_observed_inputs':0,'formal_confirmation':False})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--out',required=True)
    a=parser.parse_args();globals()[a.action](Path(a.out).resolve())
