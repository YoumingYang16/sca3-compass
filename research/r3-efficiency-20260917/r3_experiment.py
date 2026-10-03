"""Identity-bound R3 development/confirmation; preserves R2 and V1."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor
import argparse,gzip,json,os,shutil,subprocess,sys,time,traceback
sys.dont_write_bytecode=True
import r3_common
from r3_common import PHASE,ROOT,R2,read,write,sha,js,score,generate,verify_r2
import r3_model
from r3_model import evaluate
from predictive_bridge import evaluate as evaluate_r2
from experiment import evaluate_v1
from provenance import check_record
from threadpoolctl import threadpool_limits
import numpy as np

FILES=['r3_experiment.py','r3_common.py','r3_model.py','test_r3.py','R3_THEORY.md',
       'bounded_intervals.py','analyze_confirm.py','test_intervals.py','INTERVAL_PROTOCOL.md','CALIBRATION_MOMENTS.md',
       'test_execution.py']
OLD_INDEX=None

def validate(out):
    global OLD_INDEX
    verify_r2();fr=read(out/'freeze.json');p=read(out/'protocol.json')
    for name,digest in fr['files'].items():
        if sha(out/name)!=digest:raise ValueError('frozen file differs:'+name)
    for module,name in [(sys.modules[__name__],'r3_experiment.py'),(r3_common,'r3_common.py'),(r3_model,'r3_model.py')]:
        path=Path(module.__file__).resolve()
        if path!=out/name or sha(path)!=fr['files'][name]:raise ValueError('actually imported source differs:'+name)
    if p['family_G']!=256 or p['reference_draws']!=4095:raise ValueError('declared prototype design differs')
    if p['input_mode'] not in ['new','reuse']:raise ValueError('unknown input mode')
    if any(type(p[k]) is not int or p[k]<=0 for k in ['workers','repetitions','reference_draws']):raise ValueError('invalid execution budget')
    if p['input_mode']=='new':
        if p['version']!=r3_model.VERSION or p['primary']!='R3_main':raise ValueError('primary identity differs')
        if len(p['cases'])!=15 or p['core_cases']!=[0,1,2,3,4,5] or len(set(p['methods']))!=8:raise ValueError('confirmatory roster differs')
    OLD_INDEX={(r['case'],r['rep']):r for r in read(R2/'C001/index.json')['rows']}

def seed_for(p,c,r):
    if not 0<=c<32 or not 0<=r<4096:raise ValueError('seed identity range')
    return 2*((int(p['seed'])*32+c)*4096+r)+1

def load_source(case,rep):
    old=R2/'C001';item=OLD_INDEX[(case,rep)]
    rel=f'raw/case-{case:02}/rep-{rep:05}.json.gz'
    if item['path']!=rel or sha(old/rel)!=item['sha256']:raise ValueError('source index binding failed')
    row=json.load(gzip.open(old/rel,'rt',encoding='utf8'))
    check_record(row,old,read(old/'protocol.json'),case,rep,metrics=True)
    with np.load(old/row['input_path'],allow_pickle=False) as a:
        values={k:a[k] for k in a.files}
    return row,item,values

def one(task):
    folder,case,rep=task;out=Path(folder);p=read(out/'protocol.json')
    record=out/f'raw/case-{case:02}/rep-{rep:05}.json.gz'
    if record.exists():raise FileExistsError('no implicit duplicate task')
    record.parent.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    row={'case':case,'rep':rep,'freeze_sha256':sha(out/'freeze.json'),'scope':p['stage']}
    try:
        with threadpool_limits(1):
            if p['input_mode']=='reuse':
                prev,item,inputs=load_source(case,rep);seed=prev['algorithm_seed']
                row.update(source_record=item,source_input_sha256=prev['input_sha256'],source_evidence_sha256=prev['evidence_sha256'])
            else:
                seed=seed_for(p,case,rep)
                z,cal,truth,shape,kappa=generate(p,case,rep)
                inputs={'z':z,'calibration':cal,'truth':truth,'shape':shape,'kappa':kappa}
                ip=record.with_name(record.name.replace('.json.gz','-input.npz'))
                with ip.open('xb') as stream:np.savez_compressed(stream,**inputs)
                row.update(input_path=ip.relative_to(out).as_posix(),input_sha256=sha(ip))
            z,cal,truth=inputs['z'],inputs['calibration'],inputs['truth']
            result=evaluate(z,cal,seed=seed,reference_draws=p['reference_draws'])
            decisions=result['decisions'].copy();saved={}
            if p['input_mode']=='reuse':
                with np.load(R2/'C001'/prev['evidence_path'],allow_pickle=False) as a:
                    if result['status']=='completed' and not np.array_equal(result['references']['R2_replay'],a['reference']):raise ValueError('paired old reference differs')
                    for label in ['PB_grid','PB_grid_ordinary','B_strong','K_NR']:decisions[label]=a['decision_'+label]
                old_times={'r2':prev['pb_seconds'],'v1':prev['v1_seconds'],'qualified':'timings reused from original workload'}
            else:
                b=evaluate_r2(z,cal,seed=seed);v=evaluate_v1(z,cal.transpose(1,0,2),seed=seed,acknowledge_scope=True)
                for label in ['PB_grid','PB_grid_ordinary']:decisions[label]=b['decisions'][label]
                for label in ['B_strong','K_NR']:decisions[label]=v['discoveries'][label]
                saved.update({'e_'+k:b['evidence'][k] for k in ['PB_grid','PB_grid_ordinary']})
                old_times={'r2':b['seconds'],'v1':v['diagnostics']['elapsed_seconds'],'r2_status':b['status'],'v1_status':v['status']}
            if set(decisions)!=set(p['methods']):raise ValueError('method roster changed')
            sensitivity={}
            if p['input_mode']=='new' and p['cases'][case].get('outside_M0'):
                for tag,fn,key in [('R3_Delta5',evaluate,'R3_main'),('R2_Delta5',evaluate_r2,'PB_grid')]:
                    extra=fn(z,cal,seed=seed,mismatch_bound=5.)
                    decisions[tag]=extra['decisions'][key];saved['e_'+tag]=extra['evidence'][key]
                    sensitivity[tag]={'status':extra['status'],'seconds':extra['seconds'],'error':extra.get('error')}
            ep=record.with_name(record.name.replace('.json.gz','-evidence.npz'))
            saved.update({'decision_'+k:v for k,v in decisions.items()})
            saved.update({'e_'+k:v for k,v in result['evidence'].items()})
            saved.update({'p_'+k:v for k,v in result['p'].items()})
            saved.update({'reference_'+k:v for k,v in result['references'].items()})
            with ep.open('xb') as stream:np.savez_compressed(stream,**saved)
            row.update(status=result['status'],algorithm_seed=seed,version=result['version'],error=result.get('error'),
                       evidence_path=ep.relative_to(out).as_posix(),evidence_sha256=sha(ep),
                       metrics={k:score(v,truth) for k,v in decisions.items()},folds=result['folds'],
                       kappa_geometric=result.get('kappa_geometric'),kappa_true=float(inputs['kappa']),
                       kappa_median=result.get('kappa_median_learning_only'),r3_seconds=result['seconds'],
                       reference_seconds=result.get('reference_seconds'),old_times=old_times,sensitivity=sensitivity)
    except Exception as error:
        row.update(status='hard_failure',error=repr(error),traceback=traceback.format_exc())
    row.update(wall_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    with gzip.open(record,'xt',encoding='utf8') as stream:json.dump(row,stream,default=js,allow_nan=False)
    return {'case':case,'rep':rep,'path':record.relative_to(out).as_posix(),'sha256':sha(record),'status':row['status']}

def freeze(out,protocol):
    verify_r2();out.mkdir();shutil.copy2(protocol,out/'protocol.json')
    for name in FILES:shutil.copy2(PHASE/name,out/name)
    write(out/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'files':{f:sha(out/f) for f in FILES+['protocol.json']},
      'r2_manifest_sha256':sha(R2/'DELIVERY_MANIFEST.json'),'r2_index_sha256':sha(R2/'C001/index.json'),
      'r2_freeze_sha256':sha(R2/'C001/freeze.json'),'r2_protocol_sha256':sha(R2/'C001/protocol.json')})

def checkpoint_row(out,case,rep):
    """Retain completed attempts, reject unbound/partial artifacts; never reroll."""
    rel=f'raw/case-{case:02}/rep-{rep:05}.json.gz';path=out/rel
    if not path.exists():
        if any((out/rel.replace('.json.gz',suffix)).exists() for suffix in ['-input.npz','-evidence.npz']):
            raise ValueError('orphaned interrupted artifacts require explicit audit: '+rel)
        return None
    with gzip.open(path,'rt',encoding='utf8') as stream:row=json.load(stream)
    if row['case']!=case or row['rep']!=rep or row['freeze_sha256']!=sha(out/'freeze.json'):
        raise ValueError('checkpoint identity differs')
    for key in ['input','evidence']:
        if key+'_path' in row:
            expected=rel.replace('.json.gz','-'+key+'.npz')
            if row[key+'_path']!=expected or sha(out/expected)!=row[key+'_sha256']:
                raise ValueError('checkpoint artifact differs')
    return {'case':case,'rep':rep,'path':rel,'sha256':sha(path),'status':row['status']}

def run(out,resume=False):
    validate(out);p=read(out/'protocol.json')
    if (out/'index.json').exists():raise FileExistsError('finished index exists; do not restart')
    if resume:
        receipts=[out/'started.json']+list(out.glob('resumed-*.json'))
        for path in receipts:
            if not path.exists():continue
            pid=read(path)['pid']
            if os.name=='nt':
                query=subprocess.run(['powershell','-NoProfile','-Command',
                  f"$ErrorActionPreference='Stop'; try {{ $r3p=Get-Process; if ($r3p.Id -contains {int(pid)}) {{exit 0}} else {{exit 10}} }} catch {{exit 2}}"],capture_output=True)
                if query.returncode not in [0,10]:raise RuntimeError('cannot establish prior process status')
                alive=query.returncode==0
            else:
                try:os.kill(pid,0);alive=True
                except ProcessLookupError:alive=False
            if alive:raise RuntimeError('prior owner PID still exists; inspect before resume:'+str(pid))
        receipt=out/f'resumed-{time.time_ns()}.json'
    else:receipt=out/'started.json'
    cases=p.get('case_indices',list(range(len(p.get('cases',[])))))
    tasks=[];rows=[]
    for c in cases:
        for r in range(p['repetitions']):
            old=checkpoint_row(out,c,r)
            if old is not None:
                if not resume:raise FileExistsError('existing attempts require explicit resume')
                rows.append(old)
            else:tasks.append((str(out),c,r))
    write(receipt,{'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),'retained':len(rows),'remaining':len(tasks)})
    total=len(tasks)+len(rows);started=time.perf_counter();step=256 if p['input_mode']=='new' else 24
    with ProcessPoolExecutor(max_workers=p['workers'],initializer=validate,initargs=(out,)) as pool:
        for row in pool.map(one,tasks,timeout=p['safety_timeout_seconds']):
            rows.append(row)
            if len(rows)%step==0:
                receipt={'completed':len(rows),'total':total,'failures':sum(x['status']!='completed' for x in rows),'seconds':time.perf_counter()-started,'utc':datetime.now(timezone.utc).isoformat()}
                print(receipt,flush=True)
                write(out/f'progress-{len(rows):06}.json',receipt)
    rows.sort(key=lambda x:(x['case'],x['rep']))
    write(out/'index.json',{'rows':rows,'complete':len(rows)==total,'seconds_current_invocation':time.perf_counter()-started})

def check_input_schema(a,p,case):
    if set(a.files)!={'z','calibration','truth','shape','kappa'}:raise ValueError('input schema keys')
    g=p['family_G'];expected={'z':(g,4,6),'calibration':(p['cases'][case]['n'],4,6),'truth':(g,2),'shape':(4,4),'kappa':()}
    for key,shape in expected.items():
        x=a[key]
        if x.shape!=shape or x.dtype!=(np.dtype(bool) if key=='truth' else np.dtype('float64')) or not np.isfinite(x).all():raise ValueError('input schema:'+key)
    if float(a['kappa'])<=0 or not np.allclose(a['shape'],a['shape'].T) or np.linalg.eigvalsh(a['shape']).min()<=0:raise ValueError('invalid recorded true nuisance')

def check_evidence_schema(a,row,p):
    status=row['status']
    if status not in ['completed','conservative_numerical_failure']:raise ValueError('unknown result status')
    r3={'R3_main','R3_scale','R3_ordinary','R3_median'};pb={'PB_grid','PB_grid_ordinary'}
    delta={'R3_Delta5','R2_Delta5'} if p['cases'][row['case']].get('outside_M0') else set()
    methods=set(p['methods'])|delta;ev=r3|pb|delta
    pp={'R3_main','R3_scale','R3_median'} if status=='completed' else set()
    refs={'R3_main','R3_scale','R3_median','R2_replay'} if status=='completed' else set()
    expected={'decision_'+k for k in methods}|{'e_'+k for k in ev}|{'p_'+k for k in pp}|{'reference_'+k for k in refs}
    if set(a.files)!=expected:raise ValueError('evidence schema keys:'+str(set(a.files)^expected))
    g=p['family_G']
    for k in methods:
        if a['decision_'+k].shape!=(g,2) or a['decision_'+k].dtype!=np.dtype(bool):raise ValueError('decision schema:'+k)
    from finite_calibration import ebh
    for k in ev:
        x=a['e_'+k]
        if x.shape!=(g,2) or x.dtype!=np.dtype('float64') or not np.isfinite(x).all() or np.any(x<0):raise ValueError('e schema:'+k)
        if not np.array_equal(ebh(x,.05),a['decision_'+k]):raise ValueError('saved mandatory eBH mismatch:'+k)
    if status!='completed' and any(np.any(a['e_'+k]) or np.any(a['decision_'+k]) for k in r3):raise ValueError('nonzero conservative failure')
    for k in pp:
        x=a['p_'+k]
        if x.shape!=(g,2,2) or x.dtype!=np.dtype('float64') or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('p schema:'+k)
    for k in refs:
        x=a['reference_'+k]
        if x.shape!=(p['reference_draws'],) or x.dtype!=np.dtype('float64') or not np.isfinite(x).all() or np.any(np.diff(x)<0):raise ValueError('reference schema:'+k)
    if row['old_times']['r2_status'] not in ['completed','conservative_numerical_failure']:raise ValueError('unknown R2 status')
    if row['old_times']['r2_status']!='completed' and any(np.any(a['e_'+k]) for k in pb):raise ValueError('R2 failure not zero')
    for k in delta:
        if row['sensitivity'][k]['status'] not in ['completed','conservative_numerical_failure']:raise ValueError('unknown sensitivity status')
        if row['sensitivity'][k]['status']!='completed' and np.any(a['e_'+k]):raise ValueError('sensitivity failure not zero')

def audit_rows(out):
    validate(out);p=read(out/'protocol.json');index=read(out/'index.json')
    cases=p.get('case_indices',list(range(len(p.get('cases',[])))))
    expected={(c,r) for c in cases for r in range(p['repetitions'])}
    if not index['complete'] or len(index['rows'])!=len(expected) or {(x['case'],x['rep']) for x in index['rows']}!=expected:raise ValueError('task Cartesian set differs')
    rows=[]
    for item in index['rows']:
        c,r=item['case'],item['rep'];rel=f'raw/case-{c:02}/rep-{r:05}.json.gz'
        if item['path']!=rel or sha(out/rel)!=item['sha256']:raise ValueError('record identity changed')
        row=json.load(gzip.open(out/rel,'rt',encoding='utf8'))
        if row['case']!=c or row['rep']!=r or row['freeze_sha256']!=sha(out/'freeze.json') or row['status']!=item['status']:raise ValueError('row task differs')
        if row['status']=='hard_failure':raise ValueError('retained hard failure:'+row['error'])
        if p['input_mode']=='reuse':
            prev,old,inputs=load_source(c,r)
            if row['source_record']!=old or row['algorithm_seed']!=prev['algorithm_seed']:raise ValueError('old task identity differs')
            truth=inputs['truth']
        else:
            if row['algorithm_seed']!=seed_for(p,c,r) or row['input_path']!=rel.replace('.json.gz','-input.npz') or sha(out/row['input_path'])!=row['input_sha256']:raise ValueError('input identity differs')
            with np.load(out/row['input_path'],allow_pickle=False) as a:
                check_input_schema(a,p,c);truth=a['truth']
        if row['evidence_path']!=rel.replace('.json.gz','-evidence.npz') or sha(out/row['evidence_path'])!=row['evidence_sha256']:raise ValueError('evidence identity differs')
        expected_methods=set(p['methods'])
        if p['input_mode']=='new' and p['cases'][c].get('outside_M0'):expected_methods|={'R3_Delta5','R2_Delta5'}
        if set(row['metrics'])!=expected_methods:raise ValueError('metric roster differs')
        if p.get('version') and row['version']!=p['version']:raise ValueError('method version differs')
        with np.load(out/row['evidence_path'],allow_pickle=False) as a:
            if p['input_mode']=='new':check_evidence_schema(a,row,p)
            if {k[9:] for k in a.files if k.startswith('decision_')}!=expected_methods:raise ValueError('decision roster differs')
            for key in expected_methods:
                if score(a['decision_'+key],truth)!=row['metrics'][key]:raise ValueError('raw metric mismatch')
                if 'e_'+key in a.files:
                    from finite_calibration import ebh
                    if not np.array_equal(ebh(a['e_'+key],.05),a['decision_'+key]):raise ValueError('saved evidence/decision mismatch')
        rows.append(row)
    return rows

def analyze_dev(out):
    rows=audit_rows(out);p=read(out/'protocol.json');result=[]
    for c in p['case_indices']:
        rr=[r for r in rows if r['case']==c];entry={'case':c,'n':len(rr),'methods':{},'paired':{}}
        for key in p['methods']:
            entry['methods'][key]={metric:None if rr[0]['metrics'][key][metric] is None else float(np.mean([r['metrics'][key][metric] for r in rr])) for metric in ['power','fdp','tp','fp']}
        for key in ['PB_grid','R3_scale','R3_ordinary','B_strong']+(['R3_median'] if 'R3_median' in p['methods'] else []):
            if rr[0]['metrics'][key]['power'] is None:continue
            v=np.array([r['metrics']['R3_main']['power']-r['metrics'][key]['power'] for r in rr])
            entry['paired'][key]={'mean':float(v.mean()),'mcse':float(v.std(ddof=1)/np.sqrt(len(v)))}
        entry['mean_seconds']=float(np.mean([r['r3_seconds'] for r in rr if r['r3_seconds'] is not None]))
        entry['failures']=sum(r['status']!='completed' for r in rr)
        result.append(entry);print(c,{k:v['power'] for k,v in entry['methods'].items()},flush=True)
    write(out/'summary.json',{'stage':p['stage'],'rows':result,'index_sha256':sha(out/'index.json'),
       'freeze_sha256':sha(out/'freeze.json'),'r2_index_sha256':sha(R2/'C001/index.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','resume','analyze_dev']);p.add_argument('--out',required=True);p.add_argument('--protocol')
    a=p.parse_args();out=Path(a.out).resolve()
    if a.action=='freeze':freeze(out,Path(a.protocol))
    elif a.action=='resume':run(out,resume=True)
    else:globals()[a.action](out)
