"""D019: bounded same-input calibration contribution check, not confirmation."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,gzip,json,shutil,time,sys,traceback
import numpy as np
from threadpoolctl import threadpool_limits
from r5_common import PHASE,R4,write,read,sha,fc,exp,inherited
from smooth_borrowing import evaluate

FILES=['triangular_development.py','triangular_calibration.py','ancillary_calibration.py',
    'r5_common.py','r5_kernel.py','conditional_reference.py','selective_reference.py','smooth_borrowing.py',
    'selection_profile.py','codesigned_profile.py','switch_closure.py','soft_drift_closure.py',
    'joint_reference.py','continuous_profile.py','D019_RULE.md','TRIANGULAR_CALIBRATION.md',
    'SELECTIVE_BRANCH_THEOREM.md','SMOOTH_SELECTIVE_E.md','test_triangular_calibration.py']
METHODS=['smooth','target','source_bound','bridge','variance_bridge']


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in FILES:shutil.copy2(PHASE/name,out/name)
    write(out/'protocol.json',{'stage':'OLD_INPUT_DEVELOPMENT_ONLY','cases':list(range(10)),
        'reps':[0,1],'outer':4095,'meta':4095,'workers':4,'no_sample_extension':True,
        'prior':'D009 radial exact same-input reference','confirmation':False,
        'scope':'All original legal cases retained;9 explicitly OUTSIDE',
        'stop':'20 input families; evidence gaps handled by mechanism, not repeat extension'})
    write(out/'freeze.json',{'files':{name:sha(out/name) for name in FILES+['protocol.json']}})


def verify(out):
    for name,digest in read(out/'freeze.json')['files'].items():
        if sha(out/name)!=digest:raise ValueError('freeze mismatch:'+name)
    for name in ['triangular_calibration','ancillary_calibration','smooth_borrowing','r5_kernel',
                 'conditional_reference','selective_reference','r5_common']:
        if Path(sys.modules[name].__file__).resolve()!=out/(name+'.py'):raise ValueError('import mismatch:'+name)


def one(task):
    directory,c,r=task;out=Path(directory)
    import triangular_calibration
    verify(out);start=time.perf_counter();p=read(out/'protocol.json')
    oldpath=R4/f'C001/raw/case-{c:02}/rep-{r:05}.json.gz'
    old=json.load(gzip.open(oldpath,'rt',encoding='utf8'));row={'case':c,'rep':r,
        'source_record':str(oldpath.relative_to(PHASE.parent.parent)),'source_sha':sha(oldpath)}
    dest=out/f'raw/case-{c:02}/rep-{r:05}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError('duplicate task')
    try:
        for k in ['observed','diagnostic','evidence']:
            if sha(R4/'C001'/old[k+'_path'])!=old[k+'_sha256']:raise ValueError('input hash')
        with np.load(R4/'C001'/old['observed_path'],allow_pickle=False) as f:obs=dict(f)
        with np.load(R4/'C001'/old['diagnostic_path'],allow_pickle=False) as f:truth=f['truth']
        with np.load(R4/'C001'/old['evidence_path'],allow_pickle=False) as f:baseline=dict(f)
        bound=read(R4/'C001/protocol.json')['cases'][c]['D'];values={}
        with threadpool_limits(1):
            for method in METHODS:
                values[method]=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],
                    bound=bound,seed=old['algorithm_seed'],reference_draws=p['outer'],inner_draws=p['meta'],
                    method=method,calibration_law='triangular')
        arrays={};receipts={};decisions={m:val['decision'] for m,val in values.items()}
        for method in ['bridge','variance_bridge']:
            e=.5*values['target']['e']+.5*values[method]['e'];arrays['e_mix_'+method]=e
            decisions['mix_'+method]=fc.ebh(e,.05)
        for method,val in values.items():
            for key in ['e','p','decision','ordinary_e','component_e']:
                if key in val:arrays[method+'_'+key]=val[key]
            rt=val['reference_tuples']
            for key in ['es','et','base','inner_target','inner_bridge']:
                if key in rt:arrays[method+'_ref_'+key]=rt[key]
            receipts[method]={k:v for k,v in val.items() if k not in ['e','p','decision','ordinary_e','ordinary_decision','component_e','reference_tuples']}
            receipts[method]['reference']={k:v for k,v in rt.items() if k not in ['es','et','base','inner_target','inner_bridge']}
        for method in ['strong_target','strong_pool_bound','target','source_bound','bridge']:
            decisions['R4_'+method]=baseline['decision_'+method]
        old_r5=PHASE/f'D009/raw/case-{c:02}/rep-{r:05}.json.gz'
        oldrow=json.load(gzip.open(old_r5,'rt',encoding='utf8'))
        row['radial_reference']={'path':str(old_r5.relative_to(PHASE)),'sha':sha(old_r5),'metrics':oldrow['metrics']}
        row['metrics']={m:exp.score(v,truth) for m,v in decisions.items()}
        ev=dest.with_name(dest.name.replace('.json.gz','-evidence.npz'))
        with ev.open('xb') as f:np.savez_compressed(f,**arrays,**{'decision_'+m:v for m,v in decisions.items()})
        row.update(status='completed',receipts=receipts,evidence_path=ev.relative_to(out).as_posix(),
                   evidence_sha=sha(ev),algorithm_seed=old['algorithm_seed'])
    except Exception as err:row.update(status='hard_failure',error=repr(err),traceback=traceback.format_exc())
    row.update(seconds=time.perf_counter()-start,utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(dest,'xt',encoding='utf8') as f:json.dump(row,f,allow_nan=False,default=inherited.js)
    return {'case':c,'rep':r,'path':dest.relative_to(out).as_posix(),'sha':sha(dest),'status':row['status']}


def run(out):
    import triangular_calibration
    verify(out);p=read(out/'protocol.json');write(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat()})
    jobs=[(str(out),c,r) for c in p['cases'] for r in p['reps']];items=[]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        for result in as_completed([pool.submit(one,j) for j in jobs],timeout=1800):
            item=result.result();items.append(item);print(json.dumps({'done':len(items),'total':len(jobs),**item}),flush=True)
    write(out/'index.json',{'rows':items});analyze(out)


def analyze(out):
    rows=[]
    for item in read(out/'index.json')['rows']:
        if sha(out/item['path'])!=item['sha']:raise ValueError('record hash')
        row=json.load(gzip.open(out/item['path'],'rt',encoding='utf8'));rows.append(row)
        if row['status']=='completed' and sha(out/row['evidence_path'])!=row['evidence_sha']:raise ValueError('evidence hash')
    summary=[]
    for c in range(10):
        rr=[r for r in rows if r['case']==c];ok=[r for r in rr if r['status']=='completed']
        if len(ok)!=2:summary.append({'case':c,'status':'INCOMPLETE'});continue
        means={m:{k:None if ok[0]['metrics'][m][k] is None else float(np.mean([r['metrics'][m][k] for r in ok]))
                  for k in ['power','fdp','tp','fp','discoveries']} for m in ok[0]['metrics']}
        oldmeans={m:None if ok[0]['radial_reference']['metrics'][m]['power'] is None else
                  float(np.mean([r['radial_reference']['metrics'][m]['power'] for r in ok]))
                  for m in ok[0]['radial_reference']['metrics']}
        failures={m:sum(r['receipts'][m]['status']!='completed' for r in ok) for m in METHODS}
        summary.append({'case':c,'means':means,'radial_power':oldmeans,'method_failures':failures})
        print(c,{m:round(v['power'],4) if v['power'] is not None else None for m,v in means.items()},flush=True)
    write(out/'summary.json',{'stage':'DEVELOPMENT_ONLY','n':len(rows),'rows':summary,
        'failures':[r for r in rows if r['status']!='completed']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--out',required=True)
    args=p.parse_args();globals()[args.action](Path(args.out).resolve())
