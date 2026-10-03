"""Frozen D014 same-information development; do not run working tree directly."""
import argparse,gzip,json,shutil,time,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from threadpoolctl import threadpool_limits
from r5_common import PHASE,R4,sha,read,write,fc,exp
from joint_power_kernel import evaluate
from joint_power_reference import endpoint_inverse
from ancillary_calibration import from_bank

FILES=['joint_development.py','joint_power_kernel.py','joint_power_reference.py','test_joint_power_reference.py',
       'ancillary_calibration.py','selection_profile.py','r5_common.py','JOINT_MOMENT_BUDGET.md','ADAPTATION_MOMENT_COST.md',
       'D014_RULE.md','R5_CLOSEST_WORK.md']
def prepare(out):
    out.mkdir(exist_ok=False)
    write(out/'protocol.json',{'id':'D014','stage':'DEVELOPMENT_ONLY_ALREADY_USED_INPUTS','candidate':'R5-A0.9.1',
        'cases':list(range(10)),'repetitions':8,'alpha':4,'successes':64,'meta_draws':4095,'numerator_draws':1024,
        'proposal_cap':1000000,'workers':4,'timeout_seconds':1800,'source_freeze':sha(R4/'C001/freeze.json'),
        'stop':'fixed80wholefamilies; no newobservations or confirmation; D014_RULE.md',
        'confirmation_budget_used':0})
    for f in FILES:shutil.copy2(PHASE/f,out/f)
    write(out/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'files':{f:sha(out/f) for f in FILES+['protocol.json']}})

def verify(out):
    assert Path(__file__).resolve()==out/'joint_development.py','execute frozen driver'
    for f,h in read(out/'freeze.json')['files'].items():assert sha(out/f)==h
    for name in ['joint_power_kernel','joint_power_reference','ancillary_calibration']:
        assert Path(sys.modules[name].__file__).resolve()==out/(name+'.py')

def one(task):
    directory,c,rep=task;out=Path(directory);verify(out);p=read(out/'protocol.json');tick=time.perf_counter()
    source=R4/f'C001/raw/case-{c:02}/rep-{rep:05}.json.gz'
    old=json.load(gzip.open(source,'rt',encoding='utf8'));arts={k:{'path':old[k+'_path'],'sha':old[k+'_sha256']} for k in ['observed','diagnostic','evidence']}
    row={'case':c,'rep':rep,'source_record_sha':sha(source),'source_artifacts':arts,'algorithm_seed':old['algorithm_seed'],
         'freeze_sha':sha(out/'freeze.json'),'stage':p['stage']};arrays={}
    try:
        for d in arts.values():assert sha(R4/'C001'/d['path'])==d['sha']
        with np.load(R4/'C001'/arts['observed']['path'],allow_pickle=False) as f:obs=dict(f)
        with np.load(R4/'C001'/arts['diagnostic']['path'],allow_pickle=False) as f:truth=f['truth']
        with np.load(R4/'C001'/arts['evidence']['path'],allow_pickle=False) as f:old_ev=dict(f)
        D=read(R4/'C001/protocol.json')['cases'][c]['D']
        with threadpool_limits(1):
            v=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=D,seed=old['algorithm_seed'],
                successes=p['successes'],meta_draws=p['meta_draws'],numerator_draws=p['numerator_draws'],proposal_cap=p['proposal_cap'])
            if v['status']!='completed':
                row.update(status='INCOMPLETE_SHARED_LEARNING_REQUIRES_DIAGNOSIS',candidate=v)
            else:
                meta=v['meta'];u=v['calibration']['centered_u'];receipts={};ee={'joint_power':v['e']}
                ws=from_bank(obs['calibration_source']).w;wt=from_bank(obs['calibration_target']).w;g=len(obs['z'])
                weights={'power_target':0.,'power_source_bound':1.,'power_count_bridge':len(ws)/(len(ws)+len(wt)),'power_variance_bridge':meta['a']}
                seeds=np.random.SeedSequence(old['algorithm_seed']).spawn(8)[4:]
                for (name,a),ss in zip(weights.items(),seeds):
                    try:
                        lr,receipt=endpoint_inverse(ws,wt,meta['ms'],meta['mt'],a,g,int(ss.generate_state(1,dtype=np.uint64)[0]),
                            successes=p['successes'],numerator_draws=p['numerator_draws'],cap=p['proposal_cap'])
                        e=v['raw_e_power']*np.exp(lr-2*a*u)
                        if not np.isfinite(e).all():raise ArithmeticError('nonfinite endpoint e')
                        ee[name]=e;receipts[name]={'status':'completed',**receipt}
                    except (ArithmeticError,RuntimeError,np.linalg.LinAlgError) as err:
                        ee[name]=np.zeros_like(v['e']);receipts[name]={'status':'conservative_failure','error':repr(err),'receipt':getattr(err,'receipt',None)}
                ee['power_fixed_mix']=.5*ee['power_target']+.5*ee['power_count_bridge']
                ee['power_variance_mix']=.5*ee['power_target']+.5*ee['power_variance_bridge']
                dec={k:fc.ebh(e,.05) for k,e in ee.items()}
                oldrpath=PHASE/f'D009/raw/case-{c:02}/rep-{rep:05}.json.gz'
                oldr=json.load(gzip.open(oldrpath,'rt',encoding='utf8'))
                assert oldr['source_artifacts']==arts
                with np.load(PHASE/'D009'/oldr['evidence_path'],allow_pickle=False) as f:
                    for name in ['target','source_bound','bridge','fixed_e_mix','strong_target','strong_pool_bound','conditional_target','conditional_mix']:
                        dec['old_'+name]=f['decision_'+name]
                    dec['A07']=f['decision']
                arrays={**{'e_'+k:e for k,e in ee.items()},**{'decision_'+k:d for k,d in dec.items()},'raw_e_power':v['raw_e_power']}
                row.update(status='completed',metrics={k:exp.score(d,truth) for k,d in dec.items()},
                    candidate={k:x for k,x in v.items() if k not in ['e','decision','pc_e','raw_e_power']},endpoint_receipts=receipts,
                    comparison_source_sha=sha(oldrpath))
    except Exception as err:row.update(status='hard_failure',error=repr(err),traceback=traceback.format_exc())
    dest=out/f'raw/case-{c:02}/rep-{rep:05}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    if arrays:
        ep=dest.with_name(dest.name.replace('.json.gz','-evidence.npz'))
        with ep.open('xb') as f:np.savez_compressed(f,**arrays)
        row.update(evidence_path=ep.relative_to(out).as_posix(),evidence_sha=sha(ep))
    row.update(wall_seconds=time.perf_counter()-tick,utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(dest,'xt',encoding='utf8') as f:json.dump(row,f,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())
    return {'path':dest.relative_to(out).as_posix(),'sha':sha(dest),'case':c,'rep':rep,'status':row['status']}

def run(out):
    verify(out);p=read(out/'protocol.json');write(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat()});rows=[]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        fs=[pool.submit(one,(str(out),c,r)) for c in p['cases'] for r in range(p['repetitions'])]
        for f in as_completed(fs,timeout=p['timeout_seconds']):
            rows.append(f.result())
            if len(rows)%8==0:print(len(rows),'/80',datetime.now(timezone.utc).isoformat(),flush=True)
    write(out/'index.json',{'rows':sorted(rows,key=lambda r:(r['case'],r['rep']))})
    raw=[]
    for item in rows:
        assert sha(out/item['path'])==item['sha'];raw.append(json.load(gzip.open(out/item['path'],'rt',encoding='utf8')))
    if any(r['status']!='completed' for r in raw):
        write(out/'summary.json',{'status':'INCOMPLETE_REQUIRES_DIAGNOSIS','failures':[r for r in raw if r['status']!='completed']});return
    result=[]
    for c in p['cases']:
        rr=[r for r in raw if r['case']==c];means={};paired={}
        for method in rr[0]['metrics']:
            means[method]={k:None if rr[0]['metrics'][method][k] is None else float(np.mean([r['metrics'][method][k] for r in rr])) for k in ['power','fdp','tp','fp']}
            if method!='joint_power' and means[method]['power'] is not None:
                d=np.array([r['metrics']['joint_power']['power']-r['metrics'][method]['power'] for r in rr])
                paired[method]={'difference':float(d.mean()),'SE_descriptive':float(d.std(ddof=1)/np.sqrt(len(d)))}
        result.append({'case':c,'n':len(rr),'means':means,'paired':paired})
        print(c,{k:v['power'] for k,v in means.items() if k in ['joint_power','power_fixed_mix','old_conditional_mix','A07']},flush=True)
    write(out/'summary.json',{'status':'DEVELOPMENT_COMPLETE_NOT_CONFIRMATION','rows':result,
        'endpoint_failures':sum(v['status']!='completed' for r in raw for v in r['endpoint_receipts'].values()),
        'total_wall_seconds_sum':sum(r['wall_seconds'] for r in raw),'G1_G2_G3_G4':'NOT_AUTOMATICALLY_PASSED'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--out',required=True)
    a=p.parse_args();globals()[a.action](Path(a.out).resolve())
