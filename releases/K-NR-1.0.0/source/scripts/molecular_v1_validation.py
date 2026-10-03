"""Fixed, source-frozen V1 validation using the existing generator/registry.

Independent unit = one complete generated family. Resume existing records only;
no result-based sample-size increase, relabeling of C2, or silent missing rows.
"""
from research_window import wait_start_gate,cooperative_stop
wait_start_gate()
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.molecular_v1 import evaluate,METHODS,VERSION
from sca3_compass.robustness_io import write_json,write_json_gzip,content_digest
from sca3_compass.robustness_registry import update_registry
from sca3_compass.robustness_dispatch import bounded_results
from sca3_compass.robustness_pattern_test import _json_value
from sca3_compass.robustness_paired_bounds import paired_interval
from sca3_compass.robustness_confirmation_bounds import kl_interval
from sca3_compass.molecular_methods import fdr_adjust,ebh
from robustness_r1_window import generate
CONTEXT={}
OUTSIDE=[62,66,67,68,69,70,73,75]
FAIR=['B_fair_conditional_eBH','B_fair_conditional_BY']
METRICS=['fdp','power','tp','fp','discoveries']


def validate_plan(plan):
    counts=plan['counts']
    if (len(counts)!=84 or any(isinstance(n,bool) or not isinstance(n,int) or n<2 for n in counts)
            or len(set(counts[:54]))!=1):
        raise ValueError('84 fixed integer counts and equal core54 counts required')
    allocation=plan['error_allocation']
    if (set(allocation)!={'paired','fdr','local'} or
            any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or x<=0 for x in allocation.values()) or
            sum(allocation.values())>.05+1e-14 or plan['report_budget']!=.05):
        raise ValueError('Positive finite allocated report budget <=.05 required')
    if plan['candidate_version']!=VERSION or plan['methods']!=list(METHODS):
        raise ValueError('Candidate or comparator identity changed')
    if plan['outside_indices']!=OUTSIDE or plan['confirmation_attempts']!=1:
        raise ValueError('Scope or confirmation budget changed')
    if isinstance(plan['seed'],bool) or not isinstance(plan['seed'],int) or plan['seed']<0:
        raise ValueError('Fixed nonnegative integer seed required')


def read(path):
    path=Path(path)
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix=='.gz' else json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def score(reject,truth):
    total=int(reject.sum());fp=int((reject&~truth).sum());tp=int((reject&truth).sum())
    return {'fdp':fp/max(1,total),'power':tp/int(truth.sum()) if truth.any() else None,
            'tp':tp,'fp':fp,'discoveries':total}


def create(root,phase):
    base=root/'artifacts/robustness'
    run='R0077' if phase=='V1_DEV' else 'R0078'
    if (base/f'{run}-v1.protocol.json').exists():raise FileExistsError('Use resume; preserve protocol')
    registry=read(base/'EXPERIMENT_REGISTRY.json')
    if any(e['id']==run for e in registry['experiments']):raise ValueError('Experiment ID already used')
    if phase=='V1_CONFIRM':
        prior=[e for e in registry['experiments'] if e.get('settings',{}).get('phase')=='V1_CONFIRM']
        if prior:raise ValueError('V1 confirmation attempt already used')
        decision=read(base/'V1-reference/freeze-decision.json')
        if decision['freeze_allowed'] is not True:raise ValueError('No approved scientific freeze')
        plan=read(root/'configs/molecular_v1_confirmation.json')
        validate_plan(plan)
        if sha(root/'src/sca3_compass/molecular_v1.py')!=plan['candidate_sha256']:
            raise ValueError('Candidate changed before freeze')
    else:
        plan={'counts':[32]*84,'seed':77091623,'report_budget':None,
              'purpose':'Fixed necessary DEV checks, not confirmation; all existing84 scenarios, no new route'}
    cases=read(root/'configs/robustness_C2_analysis.json')['cases']
    protocol={'run_id':run,'phase':phase,'version':VERSION,'cases':cases,'counts':plan['counts'],
              'seed':plan['seed'],'plan':plan,'workers':8,'methods':list(METHODS),
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'provenance':'SIMULATION_NOT_PATIENT_DATA','outside_indices':OUTSIDE,
              'source_sha256':{},'acceptance_sha256':sha(root/'V1_ACCEPTANCE.md')}
    if len(protocol['counts'])!=84 or min(protocol['counts'])<2:raise ValueError('Invalid fixed counts')
    snap=base/f'{run}-v1-source';snap.mkdir(exist_ok=False)
    paths=[*sorted((root/'src/sca3_compass').glob('*.py')),*[root/'scripts'/n for n in
           ['molecular_v1_validation.py','molecular_v1.py','robustness_r1_window.py','robustness_screen.py','research_window.py']]]
    for path in paths:
        rel=path.relative_to(root);dest=snap/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,dest);protocol['source_sha256'][rel.as_posix()]=sha(dest)
    shutil.copy2(root/'V1_ACCEPTANCE.md',snap/'V1_ACCEPTANCE.md')
    import platform,scipy
    protocol['environment']={'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
                             'platform':platform.platform(),'BLAS_threads_per_worker':1}
    write_json(base/f'{run}-v1.protocol.json',protocol)
    update_registry(base/'EXPERIMENT_REGISTRY.json',{'id':run,'status':'frozen',
                    'purpose':phase,'settings':protocol,'started_utc':protocol['created_utc']},create=True)
    return protocol


def initialize(root,protocol):
    CONTEXT.update(root=Path(root),protocol=protocol,digest=content_digest(protocol))


def one(case,rep):
    root,p=CONTEXT['root'],CONTEXT['protocol']
    folder=root/'artifacts/robustness'/f"{p['run_id']}-v1-repetitions"/f'case-{case:04}'
    folder.mkdir(parents=True,exist_ok=True)
    record_path=folder/f'rep-{rep:06}.json.gz';ip=folder/f'rep-{rep:06}-input.npz';ep=folder/f'rep-{rep:06}-evidence.npz'
    if record_path.exists():
        old=read(record_path)
        if old['protocol_digest']!=CONTEXT['digest']:raise ValueError('Protocol mismatch')
        if old['case_index']!=case or old['rep']!=rep:raise ValueError('Identity mismatch')
        for kind in ['input','evidence']:
            if kind+'_sha256' in old and sha(old[kind+'_path'])!=old[kind+'_sha256']:raise ValueError('Checkpoint hash mismatch')
        return {'case_index':case,'rep':rep,'status':old['status'],'path':str(record_path),'sha256':sha(record_path)}
    begun,cpu=time.perf_counter(),time.process_time()
    row={'case_index':case,'rep':rep,'protocol_digest':CONTEXT['digest'],'seed_sequence':[p['seed'],case,rep]}
    try:
        if ip.exists() or ep.exists():raise FileExistsError('Orphan data retained; audit before recovery')
        with threadpool_limits(1):
            z,cal,truth=generate(p['seed'],case,rep,p['cases'][case])
            with ip.open('xb') as stream:np.savez_compressed(stream,z=z,calibration=cal,truth=truth)
            row.update(input_path=str(ip),input_sha256=sha(ip),true_signed=int(truth.sum()))
            algorithm_seed=int(np.random.SeedSequence([p['seed'],case,rep,913]).generate_state(1)[0])
            result=evaluate(z,cal,seed=algorithm_seed,acknowledge_scope=True)
            values={**{'p_'+m:v for m,v in result['p_values'].items()},
                    **{'e_'+m:v for m,v in result['e_values'].items()}}
            with ep.open('xb') as stream:np.savez_compressed(stream,**values)
            row.update(status='completed',metrics={m:score(result['discoveries'][m],truth) for m in METHODS},
                       algorithm_seed=algorithm_seed,evidence_path=str(ep),evidence_sha256=sha(ep),
                       diagnostics=_json_value(result['diagnostics']),computation_status=result['status'])
    except Exception as error:
        row.update(status='failed',error=repr(error),traceback=traceback.format_exc())
    row.update(elapsed_seconds=time.perf_counter()-begun,cpu_seconds=time.process_time()-cpu,
               finished_utc=datetime.now(timezone.utc).isoformat())
    write_json_gzip(record_path,row)
    return {'case_index':case,'rep':rep,'status':row['status'],'path':str(record_path),'sha256':sha(record_path)}


def batch(case,reps):
    rows=[]
    for rep in reps:
        if cooperative_stop():break
        rows.append(one(case,rep))
    return rows


def execute(root,run):
    base=root/'artifacts/robustness';p=read(base/f'{run}-v1.protocol.json')
    for rel,digest in p['source_sha256'].items():
        if sha(SOURCE/rel)!=digest:raise ValueError('Frozen source mismatch '+rel)
    entry=next(e for e in read(base/'EXPERIMENT_REGISTRY.json')['experiments'] if e['id']==run)
    if entry['status']=='completed':raise ValueError('Already completed; no duplicate')
    entry.update(status='running',pid=os.getpid(),execution_started_utc=datetime.now(timezone.utc).isoformat())
    update_registry(base/'EXPERIMENT_REGISTRY.json',entry)
    begun=time.perf_counter();receipts=[];planned=sum(p['counts'])
    jobs=[((i,j),batch,(i,list(range(j,min(j+4,n)))),{}) for i,n in enumerate(p['counts']) for j in range(0,n,4)]
    try:
        with ProcessPoolExecutor(max_workers=p['workers'],initializer=initialize,initargs=(str(root),p)) as pool:
            for key,rows in bounded_results(pool,jobs,max_pending=p['workers']):
                receipts.extend(rows)
                progress={'run_id':run,'completed':sum(r['status']=='completed' for r in receipts),
                          'failed':sum(r['status']!='completed' for r in receipts),'planned':planned,
                          'last_batch':list(key),'updated_utc':datetime.now(timezone.utc).isoformat()}
                write_json(base/f'{run}-v1-progress.json',progress)
                if len(receipts)%128==0:print(progress,flush=True)
                if cooperative_stop():break
        complete=len(receipts)==planned and all(r['status']=='completed' for r in receipts)
        index=base/f'{run}-v1-index.json'
        write_json(index,{'protocol':p,'complete':complete,'receipts':receipts,'wall_seconds':time.perf_counter()-begun})
        entry.update(status='completed' if complete else 'interrupted' if cooperative_stop() else 'failed',
                     result=str(index),sha256=sha(index),wall_seconds=time.perf_counter()-begun,
                     finished_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        entry.update(status='failed',error=repr(error));raise
    finally:update_registry(base/'EXPERIMENT_REGISTRY.json',entry)
    if not complete:raise SystemExit(2)


def analyze(root,run,out):
    if out.exists():raise FileExistsError('New audit output required')
    base=root/'artifacts/robustness';index=read(base/f'{run}-v1-index.json');p=index['protocol']
    if content_digest(p)!=content_digest(read(base/f'{run}-v1.protocol.json')):
        raise ValueError('Index versus frozen protocol mismatch')
    entry=next(e for e in read(base/'EXPERIMENT_REGISTRY.json')['experiments'] if e['id']==run)
    if sha(base/f'{run}-v1-index.json')!=entry['sha256']:
        raise ValueError('Registered index hash mismatch')
    if p['phase']=='V1_CONFIRM':
        validate_plan(p['plan'])
        expected_source=(base/f'{run}-v1-source').resolve()
        if SOURCE.resolve()!=expected_source:
            raise ValueError('Formal analysis must execute the archived script and its dependencies')
        for rel,digest in p['source_sha256'].items():
            if sha(SOURCE/rel)!=digest:
                raise ValueError('Frozen analysis dependency mismatch: '+rel)
    methods_list=p['methods']
    arrays={i:{m:{v:np.full(n,np.nan) for v in METRICS} for m in methods_list} for i,n in enumerate(p['counts'])}
    seen=set();failed=[];soft=[];cpu=0.
    for ref in index['receipts']:
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Record hash mismatch')
        row=read(ref['path']);key=(row['case_index'],row['rep'])
        if key in seen:raise ValueError('Duplicate family')
        if not 0<=key[0]<84 or not 0<=key[1]<p['counts'][key[0]]:
            raise ValueError('Out-of-range family identity')
        seen.add(key)
        if row['protocol_digest']!=content_digest(p):raise ValueError('Protocol mismatch')
        if row['status']!='completed':failed.append({'case':key[0],'rep':key[1],'error':row['error']});continue
        for name in ['input','evidence']:
            if sha(row[name+'_path'])!=row[name+'_sha256']:raise ValueError('Raw array hash mismatch')
        with np.load(row['input_path'],allow_pickle=False) as inputs,np.load(row['evidence_path'],allow_pickle=False) as evidence:
            truth=inputs['truth']
            if int(truth.sum())!=row['true_signed']:raise ValueError('Truth mismatch')
            for m in methods_list:
                reject=ebh(evidence['e_'+m],.05) if 'e_'+m in evidence else fdr_adjust(evidence['p_'+m],'BY')<=.05
                metrics=score(reject,truth)
                if metrics!=row['metrics'][m]:raise ValueError('Independent rescore mismatch')
                for v in METRICS:arrays[key[0]][m][v][key[1]]=np.nan if metrics[v] is None else metrics[v]
        if row['computation_status']!='COMPUTED':soft.append(list(key))
        cpu+=row['cpu_seconds']
    complete=index['complete'] and not failed and len(seen)==sum(p['counts'])
    if complete and seen!={(i,j) for i,n in enumerate(p['counts']) for j in range(n)}:
        raise ValueError('Fixed family set incomplete')
    confirm=p['phase']=='V1_CONFIRM' and complete
    plan=p['plan'];scene_rows=[];paired={};local={}
    for i,case in enumerate(p['cases']):
        methods={}
        for m in methods_list:
            values=arrays[i][m];observed=np.isfinite(values['fdp'])
            stats={v:float(np.nanmean(x)) if np.isfinite(x).any() else None for v,x in values.items()}
            stats['observed_families']=int(observed.sum())
            if confirm:
                delta=plan['error_allocation']['fdr']/(2*84*len(methods_list))
                stats['fdr_interval']=list(kl_interval(stats['fdp'],len(values['fdp']),delta))
            methods[m]=stats
        scene_rows.append({'index':i,'case':case,'scope':'OUTSIDE' if i in OUTSIDE else 'D1','methods':methods})
    strata={'core':list(range(54)),'normal':list(range(27)),'t5':list(range(27,54))}
    for name,ids in strata.items():
        comparisons={}
        for m in methods_list[1:]:
            diffs=[arrays[i]['K_NR']['power']-arrays[i][m]['power'] for i in ids]
            if any(not np.isfinite(x).all() for x in diffs):
                comparisons[m]={'incomplete':True};continue
            delta=plan.get('error_allocation',{}).get('paired',.02)/(2*3*(len(methods_list)-1))
            comparisons[m]=paired_interval(diffs,delta) if confirm else {
                'mean_difference':float(np.mean(diffs)),
                'paired_mcse':float(np.sqrt(sum(x.var(ddof=1)/len(x) for x in diffs))/len(diffs))}
            comparisons[m]['additional_tp_per_family']=float(np.mean([
                np.mean(arrays[i]['K_NR']['tp']-arrays[i][m]['tp']) for i in ids]))
        paired[name]={'comparisons':comparisons,
            'power':{m:float(np.mean([np.mean(arrays[i][m]['power']) for i in ids])) for m in methods_list}}
    if confirm:
        for i in range(84):
            if np.isfinite(arrays[i]['K_NR']['power']).all():
                local[str(i)]={m:paired_interval([arrays[i]['K_NR']['power']-arrays[i][m]['power']],
                          plan['error_allocation']['local']/(2*84*(len(methods_list)-1))) for m in methods_list[1:]}
    candidate_valid=confirm and all(row['methods']['K_NR']['fdr_interval'][1]<=.05 for row in scene_rows if row['scope']=='D1')
    fair_valid=confirm and all(row['methods'][m]['fdr_interval'][1]<=.05 for row in scene_rows if row['scope']=='D1' for m in FAIR)
    superior=confirm and all(paired[s]['comparisons'][m]['lower']>0 for s in strata for m in FAIR)
    negative=[{'case':int(i),'comparator':m,'interval':v} for i,by in local.items()
              for m,v in by.items() if int(i) not in OUTSIDE and m in FAIR and v['upper']<0]
    summary={'run_id':run,'phase':p['phase'],'complete':complete,'formal_confirmation':confirm,
             'evidence_level':'EMPIRICAL_ONLY','methods':methods_list,'attempts':len(seen),
             'planned':sum(p['counts']),'failed':failed,'soft_fallback_families':soft,'cpu_seconds':cpu,
             'wall_seconds':index['wall_seconds'],'scene_rows':scene_rows,'paired':paired,
             'local':local,'clear_D1_fair_losses':negative,
             'gates':{'candidate_empirical_FDR':candidate_valid,'fair_baselines_empirical_FDR':fair_valid,
                      'paired_superiority':superior,'no_clear_local_fair_losses':not negative if confirm else None,
                      'scientific_pass_pending_replay_review':candidate_valid and fair_valid and superior and not negative},
             'statistical_notice':'DEV descriptive; confirmation joint fixed-n scope only, no finite-estimation theorem.',
             'protocol_sha256':sha(base/f'{run}-v1.protocol.json')}
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'analyzer_source.py')
    for i,data in arrays.items():
        with (out/f'case-{i:04}.npz').open('xb') as stream:
            np.savez_compressed(stream,**{m+'__'+v:a for m,vs in data.items() for v,a in vs.items()})
    write_json(out/'summary.json',_json_value(summary))
    print(json.dumps({'run_id':run,'complete':complete,'gates':summary['gates'],'paired':paired},allow_nan=False))


def replay(root,run,out):
    if out.exists():raise FileExistsError('New replay receipt required')
    base=root/'artifacts/robustness';index=read(base/f'{run}-v1-index.json');p=index['protocol']
    if p['phase']!='V1_CONFIRM' or not index['complete']:raise ValueError('Completed frozen confirmation required')
    if SOURCE.resolve()!=(base/f'{run}-v1-source').resolve():raise ValueError('Execute archived replay')
    for rel,digest in p['source_sha256'].items():
        if sha(SOURCE/rel)!=digest:raise ValueError('Frozen replay source mismatch')
    selected={(i,j) for i in p['plan']['replay']['cases'] for j in p['plan']['replay']['reps']}
    refs={}
    for ref in index['receipts']:
        row=read(ref['path'])
        key=(row['case_index'],row['rep'])
        if row['computation_status']!='COMPUTED':selected.add(key)
        refs[key]=ref
    begin=time.perf_counter();results=[]
    with threadpool_limits(1):
        for i,j in sorted(selected):
            if cooperative_stop():raise InterruptedError('Replay safely interrupted; preserve original confirmation')
            ref=refs[(i,j)]
            if sha(ref['path'])!=ref['sha256']:raise ValueError('Replay record hash')
            row=read(ref['path'])
            z,cal,truth=generate(p['seed'],i,j,p['cases'][i])
            for name in ['input','evidence']:
                if sha(row[name+'_path'])!=row[name+'_sha256']:raise ValueError('Replay array hash')
            with np.load(row['input_path'],allow_pickle=False) as old:
                if any(not np.array_equal(old[n],v) for n,v in [('z',z),('calibration',cal),('truth',truth)]):
                    raise ValueError('Seed regeneration differs')
            result=evaluate(z,cal,seed=row['algorithm_seed'],acknowledge_scope=True)
            arrays={**{'p_'+m:v for m,v in result['p_values'].items()},**{'e_'+m:v for m,v in result['e_values'].items()}}
            with np.load(row['evidence_path'],allow_pickle=False) as old:
                if set(old.files)!=set(arrays) or any(not np.array_equal(old[m],v) for m,v in arrays.items()):
                    raise ValueError('Fresh frozen replay differs')
            for m in p['methods']:
                if score(result['discoveries'][m],truth)!=row['metrics'][m]:
                    raise ValueError('Replay score mismatch')
            results.append({'case':i,'rep':j,'arrays_exact':len(arrays),'input_exact':True,
                            'status':result['status'],'source_record_sha256':ref['sha256']})
    write_json(out,{'run_id':run,'complete':True,'kind':'EXACT_FROZEN_REPLAY_NOT_NEW_INDEPENDENT_DATA',
                    'families':len(results),'results':results,'seconds':time.perf_counter()-begin,
                    'protocol_sha256':sha(base/f'{run}-v1.protocol.json')})
    print('Exact replay',len(results))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['dev','confirm','execute','analyze','replay'])
    parser.add_argument('--project-root',type=Path,default=SOURCE);parser.add_argument('--run')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args();root=args.project_root.resolve()
    if args.action=='execute':return execute(root,args.run)
    if args.action=='analyze':return analyze(root,args.run,args.out.resolve())
    if args.action=='replay':return replay(root,args.run,args.out.resolve())
    phase='V1_DEV' if args.action=='dev' else 'V1_CONFIRM'
    run='R0077' if phase=='V1_DEV' else 'R0078'
    p=read(root/f'artifacts/robustness/{run}-v1.protocol.json') if args.resume else create(root,phase)
    snap=root/f'artifacts/robustness/{run}-v1-source'
    env=dict(os.environ,PYTHONPATH=str(snap/'src')+os.pathsep+str(snap/'scripts'),PYTHONDONTWRITEBYTECODE='1')
    raise SystemExit(subprocess.run([sys.executable,str(snap/'scripts/molecular_v1_validation.py'),
        'execute','--project-root',str(root),'--run',p['run_id']],cwd=snap,env=env).returncode)


if __name__=='__main__':main()
