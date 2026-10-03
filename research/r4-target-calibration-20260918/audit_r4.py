"""Independent identity/schema/decision audit; no new random families.

Use --implementation during run. Use --results ONLY after completed index.
Replays take prespecified rep0 of cases0,1,2,3,5,6,7,8,9, never best outcomes.
"""
from pathlib import Path
import sys,json,gzip,time,argparse,collections
import numpy as np
from threadpoolctl import threadpool_limits
PHASE=Path(__file__).resolve().parent
sys.path.insert(0,str(PHASE/'C001'))
from r4_common import read,write,sha,model,fc,exp,ci,verify_runtime
from r4_kernel import evaluate_kernel
from r4_simulation import generate
from r4_experiment import validate

def independent_score(decision,truth):
    if decision.dtype!=bool or truth.dtype!=bool or decision.shape!=truth.shape: raise ValueError('score shape/dtype')
    true_count=int(np.count_nonzero(truth));tp=int(np.count_nonzero(decision & truth))
    fp=int(np.count_nonzero(decision & ~truth));nr=tp+fp
    return {'power':tp/true_count if true_count else None,'fdp':fp/max(1,nr),'tp':tp,'fp':fp,'discoveries':nr}

def independent_ebh(e):
    # Separate implementation, not a call to frozen eBH. Largest qualifying
    # ordered rank determines a single threshold for the whole signed family.
    v=np.asarray(e);m=v.size;rank=0
    for k,x in enumerate(sorted(map(float,v.ravel()),reverse=True),start=1):
        if x>=m/(.05*k): rank=k
    return np.zeros(v.shape,bool) if not rank else v>=m/(.05*rank)

def summary_structure(p,summary,index_digest,freeze_digest):
    expected_header={'status':'COMPLETE','stage':p['stage'],'alpha':p['alpha'],
        'interval_cap':p['interval_cap'],'intervals_count':p['planned_intervals'],
        'index_sha256':index_digest,'freeze_sha256':freeze_digest}
    if any(summary.get(k)!=v for k,v in expected_header.items()): raise ValueError('summary metadata binding')
    tabs=summary['rows'];nc=len(p['cases'])
    if len(tabs)!=nc or {r['case'] for r in tabs}!=set(range(nc)):
        raise ValueError('missing/duplicate/extra summary scene')
    bycase={r['case']:r for r in tabs};keys={a+'_minus_'+b for a,b in p['comparisons']}
    for c,scene in enumerate(p['cases']):
        tab=bycase[c]
        if tab['scene']!=scene or tab['n']!=p['repetitions'] or set(tab['methods'])!=set(p['methods']):
            raise ValueError('scene/method/sample identity')
        isnull=scene.get('mixed_sign_null') or scene.get('truth') in ['global_null','single_study_only']
        if set(tab['paired_power'])!=(set() if isnull else keys): raise ValueError('per-scene comparison roster')
    if set(summary['core_paired_power'])!=keys: raise ValueError('core comparison roster')
    allocation={f'case{a}_minus_case{b}_{m}' for a,b in p['allocation_comparisons'] for m in ['target','bridge','strong_target']}
    if set(summary['allocation_comparisons'])!=allocation: raise ValueError('allocation comparison roster')
    return bycase

def implementation():
    start=time.perf_counter();out=PHASE/'C001';validate(out)
    p=read(out/'protocol.json');obs,diag,seed=generate(p,0,0)
    with threadpool_limits(1):
        # Same frozen confirmation input rep0, not an extra independent family.
        z=obs['z'];cs=obs['calibration_source'];ct=obs['calibration_target']
        a=evaluate_kernel(z,cs,ct,seed=seed,weight=len(ct)/(len(cs)+len(ct)),bound=p['cases'][0]['D'])
        zz=z.copy();zz[np.arange(len(z))%4==0]+=7.
        b=evaluate_kernel(zz,cs,ct,seed=seed,weight=len(ct)/(len(cs)+len(ct)),bound=p['cases'][0]['D'])
        x,y=a['folds'][0],b['folds'][0]
        for k in ['gamma','direction_profiles','inference_shape']:
            assert np.array_equal(x[k],y[k]),'TEST leaked into '+k
        assert x['calibrators']==y['calibrators']
        for xx,yy in zip(x['directions'],y['directions']):
            assert np.array_equal(xx['direction'],yy['direction'])
        old=model.evaluate(z,ct,seed=seed)
        endpoint=evaluate_kernel(z,cs,ct,seed=seed,weight=1.,bound=1.)
        for k,v in [('p',old['p']['R3_main']),('e',old['evidence']['R3_main']),
                    ('reference',old['references']['R3_main']),('decision',old['decisions']['R3_main'])]:
            assert np.array_equal(endpoint[k],v),k
        # Two allocation cases share Z, but MUST NOT count as independent twins.
        o1,_,s1=generate(p,1,0);o8,_,s8=generate(p,8,0)
        assert np.array_equal(o1['z'],o8['z']) and s1==s8
    write(PHASE/'checks/implementation-audit.json',{'status':'PASS','utc_scope':'ENGINEERING_REPLAY_NOT_EXTRA_CONFIRMATION',
        'checks':['G256 target endpoint exact','TEST-to-learning computational isolation',
                  'fixed-total allocation shares Z and reference seed by design','all source hashes and runtime paths'],
        'source_sha256':sha(Path(__file__)),'C001_freeze_sha256':sha(out/'freeze.json'),
        'seconds':time.perf_counter()-start,'dependencies':verify_runtime()})
    print('IMPLEMENTATION_AUDIT_PASS')

def tagged(value,path=''):
    if isinstance(value,dict):
        if set(value)=={'nonfinite_float'}:
            yield path,value['nonfinite_float']
        else:
            for k,v in value.items(): yield from tagged(v,path+'/'+k)
    elif isinstance(value,list):
        for v in value: yield from tagged(v,path+'/*')

def results():
    start=time.perf_counter();out=PHASE/'C001';validate(out)
    p=read(out/'protocol.json');idx=read(out/'index.json');summary=read(out/'summary.json')
    if summary['intervals_count']!=p['planned_intervals'] or p['planned_intervals']!=133:
        raise ValueError('actual comparison family differs from the prespecified133')
    tabs=summary_structure(p,summary,sha(out/'index.json'),sha(out/'freeze.json'))
    if not idx['complete'] or len(idx['rows'])!=len(p['cases'])*p['repetitions']:
        raise ValueError('not a completed experiment')
    expected={(c,r) for c in range(len(p['cases'])) for r in range(p['repetitions'])}
    if {(r['case'],r['rep']) for r in idx['rows']}!=expected: raise ValueError('task identities')
    nf=collections.Counter();statuses=collections.Counter();rows={};fail=[];times=[]
    learning=collections.Counter();method_costs=collections.defaultdict(list);cpu_seconds=0.
    for item in idx['rows']:
        c,r=item['case'],item['rep'];path=out/item['path']
        if item['path']!=f'raw/case-{c:02}/rep-{r:05}.json.gz' or sha(path)!=item['sha256']:
            raise ValueError('record binding')
        row=json.load(gzip.open(path,'rt',encoding='utf8'));rows[(c,r)]=row
        if (row['case'],row['rep'],row['stage'],row['status'])!=(c,r,p['stage'],item['status']):
            raise ValueError('raw/index/protocol identity')
        if row['freeze_sha256']!=sha(out/'freeze.json'): raise ValueError('record freeze')
        if row['status']!='completed': fail.append(item);continue
        if set(row['metrics'])!=set(p['methods']): raise ValueError('raw metric roster')
        kernel_methods=['source','source_bound','target','bridge','naive_pool']
        if set(row['method_status'])!=set(kernel_methods+['strong_target','strong_pool_bound']):
            raise ValueError('raw status roster')
        for k in kernel_methods:
            if row['method_status'][k] not in ['completed','conservative_numerical_failure']:
                raise ValueError('unknown kernel status')
        for k in ['strong_target','strong_pool_bound']:
            if row['method_status'][k] not in ['COMPUTED','COMPUTED_WITH_DECLARED_FOLD_FALLBACK']:
                raise ValueError('unknown strong status')
        expectedseed=int(np.random.SeedSequence([p['seed'],p['cases'][c].get('pair_id',c),r]).spawn(4)[3].generate_state(1,dtype=np.uint64)[0])
        if row['algorithm_seed']!=expectedseed: raise ValueError('algorithm stream identity')
        for tag in ['observed','diagnostic','evidence']:
            expectedpath=f'raw/case-{c:02}/rep-{r:05}-{tag}.npz'
            if row[tag+'_path']!=expectedpath or sha(out/expectedpath)!=row[tag+'_sha256']:
                raise ValueError('bound artifact')
        with np.load(out/row['observed_path'],allow_pickle=False) as a:
            if set(a.files)!={'z','calibration_source','calibration_target'}: raise ValueError('observed whitelist')
            shapes={'z':(256,4,6),'calibration_source':(p['cases'][c]['ns'],4,6),'calibration_target':(p['cases'][c]['nt'],4,6)}
            for k,s in shapes.items():
                if a[k].shape!=s or a[k].dtype!=np.float64 or not np.isfinite(a[k]).all(): raise ValueError('observed schema')
        with np.load(out/row['diagnostic_path'],allow_pickle=False) as d:
            if set(d.files)!={'truth','mu','shape_DIAGNOSTIC_ONLY','kappa_target_DIAGNOSTIC_ONLY',
                              'kappa_source_DIAGNOSTIC_ONLY','radii_DIAGNOSTIC_ONLY'}:
                raise ValueError('diagnostic key completeness')
            truth=d['truth'];mu=d['mu']
            if not np.array_equal(truth,np.stack([(mu>0).sum(1)>=2,(mu<0).sum(1)>=2],1)):
                raise ValueError('partial-conjunction truth')
        with np.load(out/row['evidence_path'],allow_pickle=False) as a:
            for k in p['methods']:
                if independent_score(a['decision_'+k],truth)!=row['metrics'][k]: raise ValueError('metric')
                e=a['e_'+k]
                if e.shape!=(256,2) or not np.isfinite(e).all() or np.any(e<0): raise ValueError('e schema')
                if not np.array_equal(independent_ebh(e),a['decision_'+k]): raise ValueError('decision')
            for k in ['source','source_bound','target','bridge','naive_pool']:
                pp=a['p_'+k];rr=a['reference_'+k];rank=pp*4096
                if pp.shape!=(256,2,2) or np.any((pp<=0)|(pp>1)) or np.any(rank!=np.floor(rank)):
                    raise ValueError('rank schema')
                if np.any((rank[:,:,0]!=4096)&(rank[:,:,0]%3!=0)): raise ValueError('ordinary support')
                if row['method_status'][k]=='completed' and (rr.shape!=(4095,) or np.any(np.diff(rr)<0) or not np.isfinite(rr).all()):
                    raise ValueError('reference schema')
                if row['method_status'][k]=='conservative_numerical_failure':
                    if np.any(a['e_'+k]) or np.any(a['decision_'+k]) or not np.all(pp==1) or rr.size:
                        raise ValueError('nonconservative failure output')
                    if k in ['target','bridge'] and (np.any(a['e_'+k+'_ordinary']) or np.any(a['decision_'+k+'_ordinary'])):
                        raise ValueError('nonconservative ordinary failure')
            if p['cases'][c]['D']==1.:
                for field in ['p','e','decision','reference']:
                    if not np.array_equal(a[field+'_bridge'],a[field+'_naive_pool']):
                        raise ValueError('D1 bridge/uncorrected identity')
        for tag,status in row['method_status'].items(): statuses[(tag,status)]+=1
        for tag in kernel_methods:
            folds=row['receipts'][tag]['folds']
            if row['method_status'][tag]=='completed':
                if len(folds)!=4 or {f['test_fold'] for f in folds}!=set(range(4)):
                    raise ValueError('fold learning receipt completeness')
                for f in folds:
                    j=f['test_fold']
                    if (f['pilot_fold'],f['direction_fold'],f['shape_fold'])!=((j+1)%4,(j+2)%4,(j+3)%4):
                        raise ValueError('fold role binding')
                    for role in ['direction','pilot']:
                        converged=f[role+'_converged']
                        if type(converged) is not bool: raise ValueError('convergence flag schema')
                        learning[(c,tag,role,converged)]+=1
                    if not f['pilot_converged'] and np.any(f['gamma']):
                        raise ValueError('nonconverged pilot must retain frozen ordinary fallback')
            elif folds: raise ValueError('numerical failure learning receipt')
        if set(row['times'])!=set(kernel_methods+['strong_target','strong_pool_bound']):
            raise ValueError('runtime label completeness')
        if any(not np.isfinite(row[k]) or row[k]<0 for k in ['cpu_seconds','wall_seconds']):
            raise ValueError('family runtime schema')
        for tag,value in row['times'].items():
            if not np.isfinite(value) or value<0: raise ValueError('runtime record')
            method_costs[tag].append(value)
        cpu_seconds+=row['cpu_seconds']
        for tag in ['strong_target','strong_pool_bound']:
            for path,value in tagged(row['receipts'][tag]): nf[(tag,path,value)]+=1
        times.append(row['wall_seconds'])
    if fail: raise ValueError('retained failures preclude complete audit: '+str(len(fail)))
    # Independent aggregation from stored decisions/metrics, plus deterministic
    # endpoint replays of prespecified first repetitions, never best results.
    assertions=0;actual_intervals=0
    for c in range(len(p['cases'])):
        tab=tabs[c];rr=[rows[(c,r)] for r in range(p['repetitions'])]
        for k in p['methods']:
            for metric in ['power','fdp','tp','fp','discoveries']:
                vals=[x['metrics'][k][metric] for x in rr]
                v=None if vals[0] is None else float(np.mean(vals))
                if v!=tab['methods'][k][metric]: raise ValueError('summary means')
                assertions+=1
            interval=ci.interval([x['metrics'][k]['fdp'] for x in rr],cap=p['interval_cap'],alpha=p['alpha'])
            if interval!=tab['methods'][k]['FDR_interval']: raise ValueError('FDR interval recomputation')
            actual_intervals+=1
        if rr[0]['metrics']['target']['power'] is not None:
            for x,y in p['comparisons']:
                v=ci.interval([r['metrics'][x]['power']-r['metrics'][y]['power'] for r in rr],low=-1,high=1,cap=p['interval_cap'],alpha=p['alpha'])
                if v!=tab['paired_power'][x+'_minus_'+y]: raise ValueError('Power interval recomputation')
                actual_intervals+=1
    for x,y in p['comparisons']:
        vals=[np.mean([rows[(c,r)]['metrics'][x]['power']-rows[(c,r)]['metrics'][y]['power'] for c in p['core_cases']]) for r in range(p['repetitions'])]
        if ci.interval(vals,low=-1,high=1,cap=p['interval_cap'],alpha=p['alpha'])!=summary['core_paired_power'][x+'_minus_'+y]:
            raise ValueError('core interval')
        actual_intervals+=1
    for ca,cb in p['allocation_comparisons']:
        for k in ['target','bridge','strong_target']:
            vals=[rows[(ca,r)]['metrics'][k]['power']-rows[(cb,r)]['metrics'][k]['power'] for r in range(p['repetitions'])]
            if ci.interval(vals,low=-1,high=1,cap=p['interval_cap'],alpha=p['alpha'])!=summary['allocation_comparisons'][f'case{ca}_minus_case{cb}_{k}']:
                raise ValueError('allocation interval')
            actual_intervals+=1
    if actual_intervals!=133: raise ValueError('actual recomputation count differs from133')
    replay=[]
    with threadpool_limits(1):
        for c in [0,1,2,3,5,6,7,8,9]:
            row=rows[(c,0)]
            with np.load(out/row['observed_path'],allow_pickle=False) as a: obs={k:a[k] for k in a.files}
            regenerated,diagnostic,algorithm_seed=generate(p,c,0)
            if algorithm_seed!=row['algorithm_seed'] or any(not np.array_equal(obs[k],regenerated[k]) for k in obs):
                raise ValueError('input regeneration identity')
            with np.load(out/row['diagnostic_path'],allow_pickle=False) as d:
                if set(d.files)!=set(diagnostic): raise ValueError('regenerated diagnostic key completeness')
                if any(not np.array_equal(d[k],diagnostic[k]) for k in d.files): raise ValueError('truth regeneration identity')
            for k,w,D in [('target',1.,1.),('bridge',p['cases'][c]['nt']/(p['cases'][c]['ns']+p['cases'][c]['nt']),p['cases'][c]['D'])]:
                result=evaluate_kernel(obs['z'],obs['calibration_source'],obs['calibration_target'],seed=row['algorithm_seed'],weight=w,bound=D)
                with np.load(out/row['evidence_path'],allow_pickle=False) as e:
                    for field in ['p','e','reference','decision']:
                        if not np.array_equal(result[field],e[field+'_'+k]): raise ValueError('replay '+str(c)+k+field)
                replay.append({'case':c,'rep':0,'method':k,'arrays_exact':True})
    write(PHASE/'checks/final-result-audit.json',{'status':'PASS','families':len(rows),
        'decisions_recomputed':len(rows)*len(p['methods']),'intervals_recomputed':actual_intervals,
        'means_recomputed':assertions,'replays':replay,'source_sha256':sha(Path(__file__)),
        'C001_index_sha256':sha(out/'index.json'),'C001_freeze_sha256':sha(out/'freeze.json'),
        'C001_summary_sha256':sha(out/'summary.json'),
        'nonfinite_diagnostic_locations':[{'method':t,'path':p,'tag':v,'count':n} for (t,p,v),n in sorted(nf.items())],
        'statuses':[{'method':m,'status':s,'count':n} for (m,s),n in sorted(statuses.items())],
        'learning_convergence':[{'case':c,'method':m,'role':r,'converged':ok,'fold_count':n}
            for (c,m,r,ok),n in sorted(learning.items())],
        'recorded_family_cpu_seconds_sum':cpu_seconds,
        'method_wall_seconds':{m:{'n':len(v),'mean':float(np.mean(v)),'p90':float(np.quantile(v,.9)),
            'p99':float(np.quantile(v,.99))} for m,v in method_costs.items()},
        'family_wall_seconds':{'mean':float(np.mean(times)),'p90':float(np.quantile(times,.9)),'p99':float(np.quantile(times,.99))},
        'seconds':time.perf_counter()-start})
    print('RESULT_AUDIT_PASS',len(rows))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['implementation','results']);a=p.parse_args()
    globals()[a.action]()
