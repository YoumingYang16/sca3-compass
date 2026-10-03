"""D020: fixed pending three-action mechanism on the SAME20 D019 families.

No observation generation, parameter scan, independent confirmation or promotion.
The protocol is written before results. Original and DIR-matched source controls
are both retained; only paired whole families are analysis units.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,gzip,json,shutil,time,sys,traceback
import numpy as np
from threadpoolctl import threadpool_limits
from r5_common import PHASE,R4,write,read,sha,fc,exp,inherited
from action_borrowing import evaluate
from r5_kernel import evaluate as endpoint
import triangular_calibration,action_reference

FILES=['action_development.py','action_borrowing.py','action_reference.py','triangular_calibration.py',
    'ancillary_calibration.py','r5_common.py','r5_kernel.py','conditional_reference.py','selective_reference.py',
    'selection_profile.py','codesigned_profile.py','switch_closure.py','soft_drift_closure.py',
    'joint_reference.py','continuous_profile.py','ACTION_KERNEL_THEOREM.md','TRIANGULAR_CALIBRATION.md',
    'SELECTIVE_BRANCH_THEOREM.md','SMOOTH_SELECTIVE_E.md','test_action_borrowing.py']


def prepare(out):
    out.mkdir(exist_ok=False)
    for name in FILES:shutil.copy2(PHASE/name,out/name)
    protocol={'id':'D020','stage':'OLD_INPUT_DEVELOPMENT_ONLY','client_date':'2026-10-03',
        'candidate':'R5-A0.14-ACTION-SELECTIVE-DEVELOPMENT','cases':list(range(10)),'reps':[0,1],
        'outer_draws':4095,'meta_draws':4095,'workers':4,'reused_input_families':20,
        'source':'D019 and underlying R4 C001, all already development',
        'mechanism':'K1 three logconcave action kernels: target, variance bridge, source; fixed c=sigma,tau=sigma*sqrt3/pi',
        'paired_controls':'ALL D019 controls; added source with Ct DIR, half target/source and equal target/bridge/source e mixtures',
        'frozen_values_not_tuned':True,'unit':'whole family; case1/8 paired, not independent genes',
        'stop':'Exactly20; no sample extension on result. No final utility/novelty acceptance from this diagnostic.',
        'precision':'Two per scenario is mechanism diagnosis only; paired observations and all raw scores retained.',
        'confirmation':False,'scope':'legal cases0-8, X9 outside D premise separately visible',
        'failure_policy':'All computational failures retained; zero-family conservative failures not removed',
        'resources':'4workers,1BLAS thread; finite selected-pair caps,1800s collector timeout not claimed per-worker kill'}
    write(out/'protocol.json',protocol)
    write(out/'freeze.json',{'files':{n:sha(out/n) for n in FILES+['protocol.json']}})


def verify(out):
    for name,digest in read(out/'freeze.json')['files'].items():
        if sha(out/name)!=digest:raise ValueError('frozen file changed:'+name)
    for name in ['action_borrowing','action_reference','triangular_calibration','r5_kernel','r5_common',
                 'conditional_reference','selective_reference','ancillary_calibration']:
        if Path(sys.modules[name].__file__).resolve()!=out/(name+'.py'):raise ValueError('wrong imported module:'+name)


def one(task):
    folder,c,r=task;out=Path(folder);verify(out);start=time.perf_counter();p=read(out/'protocol.json')
    dest=out/f'raw/case-{c:02}/rep-{r:05}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError('duplicate record')
    row={'case':c,'rep':r,'stage':p['stage'],'freeze_sha':sha(out/'freeze.json')}
    try:
        oldpath=R4/f'C001/raw/case-{c:02}/rep-{r:05}.json.gz'
        old=json.load(gzip.open(oldpath,'rt',encoding='utf8'))
        previous=PHASE/f'D019/raw/case-{c:02}/rep-{r:05}.json.gz'
        prev=json.load(gzip.open(previous,'rt',encoding='utf8'))
        if old['status']!='completed' or prev['status']!='completed':raise ValueError('incomplete input record')
        if prev['source_sha']!=sha(oldpath):raise ValueError('paired input record hash')
        for tag in ['observed','diagnostic']:
            if sha(R4/'C001'/old[tag+'_path'])!=old[tag+'_sha256']:raise ValueError('input artifact hash')
        evold=PHASE/'D019'/prev['evidence_path']
        if sha(evold)!=prev['evidence_sha']:raise ValueError('old control evidence changed')
        with np.load(R4/'C001'/old['observed_path'],allow_pickle=False) as f:obs=dict(f)
        with np.load(R4/'C001'/old['diagnostic_path'],allow_pickle=False) as f:truth=f['truth']
        with np.load(evold,allow_pickle=False) as f:control=dict(f)
        bound=read(R4/'C001/protocol.json')['cases'][c]['D']
        kw=dict(bound=bound,seed=old['algorithm_seed'],reference_draws=p['outer_draws'],inner_draws=p['meta_draws'])
        with threadpool_limits(1):
            v=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],**kw)
            matched=endpoint(obs['z'],obs['calibration_source'],obs['calibration_target'],method='source_bound',
                calibration_law='triangular',direction_calibration='target',**kw)
        evidence={'A014':v['e'],'matched_source':matched['e'],
            'matched_TS_mix':.5*(control['target_e']+matched['e']),
            'matched_three_mix':(control['target_e']+control['variance_bridge_e']+matched['e'])/3,
            'original_TS_mix':.5*(control['target_e']+control['source_bound_e'])}
        decisions={k:fc.ebh(e,.05) for k,e in evidence.items()}
        arrays={'e_'+k:e for k,e in evidence.items()};arrays.update({'decision_'+k:d for k,d in decisions.items()})
        for k in ['p','component_e','ordinary_e']: 
            if k in v:arrays['A014_'+k]=v[k]
        refs=v['reference_tuples'].get('branches',[]);receipts=[]
        for j,ref in enumerate(refs):
            for k in ['es','et','base']:arrays[f'action{j}_ref_{k}']=ref[k]
            receipts.append({k:val for k,val in ref.items() if k not in ['es','et','base','inner_target','inner_bridge']})
        ev=dest.with_name(dest.name.replace('.json.gz','-evidence.npz'))
        with ev.open('xb') as f:np.savez_compressed(f,**arrays)
        row.update(status='completed',algorithm_seed=old['algorithm_seed'],source_record_sha=sha(oldpath),
            previous_record=previous.relative_to(PHASE).as_posix(),previous_record_sha=sha(previous),
            previous_evidence=evold.relative_to(PHASE).as_posix(),previous_evidence_sha=sha(evold),
            evidence_path=ev.relative_to(out).as_posix(),evidence_sha=sha(ev),
            metrics={k:exp.score(d,truth) for k,d in decisions.items()},prior_metrics=prev['metrics'],
            candidate={k:val for k,val in v.items() if k not in ['p','e','ordinary_e','component_e','decision','ordinary_decision','reference_tuples']},
            reference_receipts=receipts,matched_source={k:matched.get(k) for k in ['status','error','seconds','calibration','folds']})
    except Exception as err:row.update(status='hard_failure',error=repr(err),traceback=traceback.format_exc())
    row['seconds']=time.perf_counter()-start;row['host_utc']=datetime.now(timezone.utc).isoformat()
    with gzip.open(dest,'xt',encoding='utf8') as f:json.dump(row,f,allow_nan=False,default=inherited.js)
    return {'case':c,'rep':r,'path':dest.relative_to(out).as_posix(),'sha':sha(dest),'status':row['status']}


def run(out):
    verify(out);p=read(out/'protocol.json');write(out/'started.json',{'host_utc':datetime.now(timezone.utc).isoformat()})
    jobs=[(str(out),c,r) for c in p['cases'] for r in p['reps']];rows=[]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        for future in as_completed([pool.submit(one,j) for j in jobs],timeout=1800):
            result=future.result();rows.append(result);print(json.dumps({'done':len(rows),'total':len(jobs),**result}),flush=True)
    write(out/'index.json',{'rows':rows});analyze(out)


def analyze(out):
    verify(out);rows=[]
    for item in read(out/'index.json')['rows']:
        if sha(out/item['path'])!=item['sha']:raise ValueError('record hash')
        row=json.load(gzip.open(out/item['path'],'rt',encoding='utf8'));rows.append(row)
        if row['status']=='completed' and sha(out/row['evidence_path'])!=row['evidence_sha']:raise ValueError('evidence hash')
    summary=[]
    for c in range(10):
        rr=[r for r in rows if r['case']==c and r['status']=='completed']
        if len(rr)!=2:summary.append({'case':c,'status':'INCOMPLETE'});continue
        def means(field):
            return {m:{k:None if rr[0][field][m][k] is None else float(np.mean([r[field][m][k] for r in rr]))
                       for k in ['power','fdp','tp','fp','discoveries']} for m in rr[0][field]}
        new=means('metrics');old=means('prior_metrics')
        paired={}
        if c not in [4,5,6]:
            for field,prefix in [('metrics','new/'),('prior_metrics','D019/')]:
                for m in rr[0][field]:
                    d=[r['metrics']['A014']['power']-r[field][m]['power'] for r in rr]
                    paired[prefix+m]={'mean':float(np.mean(d)),'replicate_differences':d}
        summary.append({'case':c,'n':2,'means':new,'prior_means':old,'paired':paired,
            'candidate_failures':sum(r['candidate']['status']!='completed' for r in rr),
            'matched_source_failures':sum(r['matched_source']['status']!='completed' for r in rr)})
        print(c,{m:v['power'] for m,v in new.items()},flush=True)
    write(out/'summary.json',{'stage':'DEVELOPMENT_ONLY_NOT_ACCEPTANCE','n':len(rows),'rows':summary,
        'failures':[r for r in rows if r['status']!='completed']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--out',required=True)
    args=p.parse_args();globals()[args.action](Path(args.out).resolve())
