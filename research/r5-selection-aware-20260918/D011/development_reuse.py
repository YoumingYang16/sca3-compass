"""Prespecified small R5 diagnosis on already-used R4 observed families.

No new simulation or protected data. These are R5 DEVELOPMENT data forever.
Frozen baseline evidence is reused, not recomputed or retroactively relabeled.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,gzip,json,shutil,time,sys,traceback
import numpy as np
from threadpoolctl import threadpool_limits
from r5_common import PHASE,R4,write,read,sha,fc,exp,inherited
from smooth_borrowing import evaluate

FILES=['r5_common.py','selection_profile.py','continuous_profile.py','codesigned_profile.py','joint_reference.py','r5_kernel.py',
       'development_reuse.py','test_selection_profile.py','test_r5_kernel.py','test_continuous_profile.py',
       'test_codesigned_profile.py','R5_THEORY.md','SELECTION_COST.md','ancillary_calibration.py',
       'conditional_reference.py','test_ancillary_calibration.py','test_conditional_reference.py',
       'ANCILLARY_NOTE.md','CONDITIONAL_PRECISION_PROPOSITION.md','INFORMATION_ADAPTATION.md',
       'switch_closure.py','test_switch_closure.py','MONOTONE_CLOSURE.md',
       'selective_reference.py','test_selective_reference.py','SELECTIVE_BRANCH_THEOREM.md',
       'smooth_borrowing.py','test_smooth_borrowing.py','SMOOTH_SELECTIVE_E.md','D011_DEVELOPMENT_RULE.md']

def prepare(out):
    if out.exists():raise FileExistsError('never overwrite a development batch')
    out.mkdir()
    p={'id':'R5-D011','stage':'DEVELOPMENT_VALIDATION_EXISTING_R4_INPUTS_NOT_CONFIRMATION',
       'candidate':'R5-A0.7.1-metadata-repair-same-successful-statistics','source':'R4-C001','source_freeze':sha(R4/'C001/freeze.json'),
       'cases':list(range(10)),'repetitions':24,'rep_start':8,'reference_draws':4095,'inner_draws':4095,
       'workers':4,'safety_timeout_seconds':1800,'selection_rule':'rep8-31 inclusive in ALL10 existingcases; old8keptseparate',
       'methods':['A0','target','source_bound','bridge','fixed_e_mix','strong_target','strong_pool_bound',
                  'conditional_target','conditional_source_bound','conditional_bridge','conditional_mix',
                  'conditional_variance_bridge','conditional_variance_mix'],
       'purpose':'Fixed A0.7.1 bounded development validation: check whether initialC1/C3 gains survive beyond8inputs, not new candidate tuning. See D011_DEVELOPMENT_RULE.md. No new observation generation, formal confirmation, priorversion rerun or novelty acceptance.',
       'stop':'Exactly240newly processed families (24each). No extending on significance. Freeze unchanged. Previous80keptseparate.',
       'precision':'Descriptive SE about half old8/case at eventualtotal32 if variance comparable; not a promised CI or acceptance test. Families not genes are units.',
       'cost':'Two computations of SAME4095 shape tuples (not8190independent shapes), two selected errorpair streams, shared4095meta errors perbank recomputedidentically. Same observationbudget;100Mpaircap perbranch; costs saved.',
       'confirmation_budget_used':0}
    write(out/'protocol.json',p)
    for f in FILES:shutil.copy2(PHASE/f,out/f)
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in FILES+['protocol.json']},
                           'utc':datetime.now(timezone.utc).isoformat()})

def verify(out):
    for f,d in read(out/'freeze.json')['files'].items():
        if sha(out/f)!=d:raise ValueError('batch file changed:'+f)
    if Path(__file__).resolve()!=out/'development_reuse.py':raise ValueError('execute frozen script')
    for name in ['r5_common','selection_profile','codesigned_profile','switch_closure','selective_reference','smooth_borrowing','ancillary_calibration','conditional_reference','r5_kernel']:
        if Path(sys.modules[name].__file__).resolve()!=out/(name+'.py'):raise ValueError('wrong module '+name)

def one(task):
    directory,c,r=task;out=Path(directory);verify(out);p=read(out/'protocol.json')
    dest=out/f'raw/case-{c:02}/rep-{r:05}.json.gz';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError('duplicate record')
    oldpath=R4/f'C001/raw/case-{c:02}/rep-{r:05}.json.gz'
    old=json.load(gzip.open(oldpath,'rt',encoding='utf8'));oldbase=R4/'C001'
    row={'case':c,'rep':r,'stage':p['stage'],'freeze':sha(out/'freeze.json'),
         'source_record':oldpath.relative_to(PHASE.parent.parent).as_posix(),'source_sha':sha(oldpath)}
    start=time.perf_counter()
    try:
        if old['status']!='completed':raise ValueError('source has unresolved failure')
        for tag in ['observed','diagnostic','evidence']:
            if sha(oldbase/old[tag+'_path'])!=old[tag+'_sha256']:raise ValueError('source artifact changed')
        with np.load(oldbase/old['observed_path'],allow_pickle=False) as f:obs=dict(f)
        with np.load(oldbase/old['diagnostic_path'],allow_pickle=False) as f:truth=f['truth']
        with np.load(oldbase/old['evidence_path'],allow_pickle=False) as f:base=dict(f)
        D=read(oldbase/'protocol.json')['cases'][c]['D']
        with threadpool_limits(1):
            v=evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=D,
                       seed=old['algorithm_seed'],reference_draws=p['reference_draws'],inner_draws=p['inner_draws'])
            conditional={method:evaluate(obs['z'],obs['calibration_source'],obs['calibration_target'],bound=D,
                       seed=old['algorithm_seed'],reference_draws=p['reference_draws'],inner_draws=p['inner_draws'],method=method)
                       for method in ['target','source_bound','bridge','variance_bridge']}
        decisions={k:base['decision_'+k] for k in ['target','source_bound','bridge','strong_target','strong_pool_bound']}
        mix=.5*base['e_target']+.5*base['e_bridge'];decisions['fixed_e_mix']=fc.ebh(mix,.05)
        decisions['A0']=v['decision']
        # Old R5 versions were evaluated only on reps0-7. Do not read nonexistent
        # prior outputs or pretend they are paired with this new development set.
        for method,val in conditional.items():decisions['conditional_'+method]=val['decision']
        conditional_mix=.5*conditional['target']['e']+.5*conditional['bridge']['e']
        decisions['conditional_mix']=fc.ebh(conditional_mix,.05)
        conditional_variance_mix=.5*conditional['target']['e']+.5*conditional['variance_bridge']['e']
        decisions['conditional_variance_mix']=fc.ebh(conditional_variance_mix,.05)
        evpath=dest.with_name(dest.name.replace('.json.gz','-evidence.npz'))
        rt=v['reference_tuples'];arrays={k:v[k] for k in ['p','e','decision']}
        if 'component_e' in v:arrays['component_e']=v['component_e']
        arrays.update({'e_fixed_e_mix':mix,**{'decision_'+k:x for k,x in decisions.items()}})
        arrays['e_conditional_mix']=conditional_mix
        arrays['e_conditional_variance_mix']=conditional_variance_mix
        for method,val in conditional.items():
            for field in ['p','e']:arrays[field+'_conditional_'+method]=val[field]
        for k in ['es','et','base','inner_target','inner_bridge']:
            if k in rt:arrays['reference_'+k]=rt[k]
        with evpath.open('xb') as f:np.savez_compressed(f,**arrays)
        row.update(status='completed',method_status=v['status'],error=v.get('error'),calibration=v['calibration'],
            p_kind=v.get('p_kind','one_valid_PC_p_array'),branch_folds=v.get('branch_folds'),branch_failures=v.get('branch_failures'),
            metrics={k:exp.score(x,truth) for k,x in decisions.items()},seconds=v['seconds'],
            reference_seconds=v['reference_seconds'],profile_counts=v['profile_counts'],folds=v['folds'],
            evidence_path=evpath.relative_to(out).as_posix(),evidence_sha=sha(evpath),
            baseline_seconds=old['times'],algorithm_seed=old['algorithm_seed'],
            conditional_receipts={method:{field:val.get(field) for field in ['status','error','calibration','seconds','folds','reference_seconds']}
                                  for method,val in conditional.items()},
            reference_receipt={key:val for key,val in rt.items() if key not in ['es','et','base','inner_target','inner_bridge']},
            source_artifacts={k:{'path':old[k+'_path'],'sha':old[k+'_sha256']} for k in ['observed','diagnostic','evidence']})
    except Exception as e:row.update(status='hard_failure',error=repr(e),traceback=traceback.format_exc())
    row.update(wall_seconds=time.perf_counter()-start,utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(dest,'xt',encoding='utf8') as f:json.dump(row,f,allow_nan=False,default=inherited.js)
    return {'case':c,'rep':r,'path':dest.relative_to(out).as_posix(),'sha':sha(dest),'status':row['status']}

def run(out):
    verify(out);p=read(out/'protocol.json');write(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat()})
    jobs=[(str(out),c,r) for c in p['cases'] for r in range(p.get('rep_start',0),p.get('rep_start',0)+p['repetitions'])];rows=[]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        fs=[pool.submit(one,j) for j in jobs]
        for f in as_completed(fs,timeout=p['safety_timeout_seconds']):
            rows.append(f.result())
            if len(rows)%8==0:print(json.dumps({'done':len(rows),'total':len(jobs),'utc':datetime.now(timezone.utc).isoformat()}),flush=True)
    write(out/'index.json',{'rows':sorted(rows,key=lambda x:(x['case'],x['rep']))})
    analyze(out)

def analyze(out):
    verify(out);idx=read(out/'index.json');p=read(out/'protocol.json');rows=[]
    for item in idx['rows']:
        if sha(out/item['path'])!=item['sha']:raise ValueError('record hash')
        row=json.load(gzip.open(out/item['path'],'rt',encoding='utf8'));rows.append(row)
        if row['status']=='completed' and sha(out/row['evidence_path'])!=row['evidence_sha']:raise ValueError('evidence hash')
    if any(r['status']!='completed' for r in rows):
        write(out/'summary.json',{'status':'INCOMPLETE','failures':[r for r in rows if r['status']!='completed']});return
    summary=[]
    for c in p['cases']:
        rr=[r for r in rows if r['case']==c];means={};paired={}
        for k in p['methods']:
            means[k]={f:None if rr[0]['metrics'][k][f] is None else float(np.mean([r['metrics'][k][f] for r in rr]))
                      for f in ['power','fdp','tp','fp','discoveries']}
            if k!='A0' and means[k]['power'] is not None:
                d=np.array([r['metrics']['A0']['power']-r['metrics'][k]['power'] for r in rr])
                paired[k]={'mean':float(d.mean()),'range':[float(d.min()),float(d.max())],
                           'paired_se_descriptive_only':float(d.std(ddof=1)/np.sqrt(len(d)))}
        row={'case':c,'n':len(rr),'means':means,'paired':paired,
             'borrow_fraction':float(np.mean([r['calibration']['borrow'] for r in rr])) if all('borrow' in r['calibration'] for r in rr) else None,
             'mean_borrow_probability':float(np.mean([r['calibration']['borrow_probability'] for r in rr])) if all('borrow_probability' in r['calibration'] for r in rr) else None,
             'mean_seconds':float(np.mean([r['seconds'] for r in rr])),
             'numerical_failures':sum(r['method_status']!='completed' for r in rr)}
        summary.append(row);print(c,{k:round(v['power'],4) if v['power'] is not None else None for k,v in means.items()},flush=True)
    write(out/'summary.json',{'status':'DEVELOPMENT_COMPLETE_NOT_CONFIRMATION','rows':summary})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--out',required=True)
    a=p.parse_args();globals()[a.action](Path(a.out).resolve())
