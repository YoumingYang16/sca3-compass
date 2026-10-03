"""Limited paired nuisance oracle ledger, NEVER deployable inference."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import combinations
import argparse, gzip, json, os, shutil, sys, time, traceback
sys.dont_write_bytecode=True
from r3_common import PHASE, R2, read, write, sha, js, score, verify_r2
import numpy as np
from threadpoolctl import threadpool_limits
from predictive_bridge import rank_tail, F42_MEDIAN
from finite_calibration import shape_fit, contrasts, positive_direction, ebh
from grid_calibration import grid_calibrate

LABELS=['R2','known_kappa','known_shape','known_both']

def banks(seed, n):
    """Exactly replay old RNG schedule; vary nuisance laws, not randomness."""
    rng=np.random.default_rng(seed); result={k:[] for k in LABELS}
    for start in range(0,4095,64):
        count=min(64,4095-start)
        h=shape_fit(rng.normal(size=(count,128,4,5)),2)
        ev=np.linalg.eigvalsh(h)
        b=np.median(rng.f(4,2,size=(count,n)),axis=1)/F42_MEDIAN
        u=rng.chisquare(5,size=(count,4))
        q=np.sum(u/ev,axis=1)
        normal=rng.normal(size=count)
        den={
            'R2':b*ev[:,0]*q/20,
            'known_kappa':ev[:,0]*q/20,
            'known_shape':b*u.sum(1)/20,
            'known_both':u.sum(1)/20}
        for key in LABELS: result[key].append(normal/np.sqrt(den[key]))
    return {k:np.sort(np.concatenate(v)) for k,v in result.items()}

def components(z,h,kappa,profiles,ref,direction_shape):
    z=z/np.max(np.abs(z),axis=(-1,-2),keepdims=True)
    y=z@contrasts(6); means=z.mean(-1)
    q=np.einsum('gik,ij,gjk->g',y,np.linalg.inv(h),y)
    den=np.sqrt(kappa*q/20); out=np.zeros((len(z),2,2))
    for d,sign in enumerate([1,-1]):
        marginal=rank_tail(sign*means/(den[:,None]*np.sqrt(np.diag(h))),ref)
        for subset in combinations(range(4),3):
            ids=list(subset); profile=profiles[d,ids]
            if not np.any(profile>0): profile=np.ones(3)
            a=positive_direction(profile,direction_shape[np.ix_(ids,ids)])
            stat=sign*(means[:,ids]@a)/(den*np.sqrt(a@h[np.ix_(ids,ids)]@a))
            out[:,d,0]=np.maximum(out[:,d,0],np.minimum(1,3*marginal[:,ids].min(1)))
            out[:,d,1]=np.maximum(out[:,d,1],rank_tail(stat,ref))
    return out

def one(task):
    out,case,rep=task; out=Path(out); started=time.perf_counter()
    dest=out/f'raw/case-{case:02}/rep-{rep:05}.json.gz'
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists(): raise FileExistsError('no implicit diagnostic restart')
    row={'case':case,'rep':rep,'scope':'DIAGNOSTIC_ONLY', 'freeze_sha256':sha(out/'freeze.json')}
    try:
        with threadpool_limits(1):
            source=R2/f'C001/raw/case-{case:02}/rep-{rep:05}.json.gz'
            r=json.load(gzip.open(source,'rt',encoding='utf8'))
            if r['status']!='completed': raise ValueError('original R2 failure cannot be dropped')
            for kind in ['input','evidence']:
                if sha(R2/'C001'/r[kind+'_path'])!=r[kind+'_sha256']: raise ValueError('old artifact mismatch')
            with np.load(R2/'C001'/r['input_path']) as data:
                z=data['z']; cal=data['calibration']; truth=data['truth']; true_r=data['shape']; true_k=float(data['kappa'])
            with np.load(R2/'C001'/r['evidence_path']) as data:
                old_p=data['p']; old_e=data['e_PB_grid']; old_ref=data['reference']
            refs=banks(r['algorithm_seed'],len(cal))
            if not np.array_equal(refs['R2'],old_ref): raise ValueError('base reference replay differs')
            ps={k:np.empty_like(old_p) for k in LABELS}
            es={k:np.empty_like(old_e) for k in LABELS}
            component_e=np.empty_like(old_p)
            for receipt in r['folds']:
                held=np.arange(len(z))%4==receipt['fold']
                h=np.asarray(receipt['shape']); profiles=np.asarray(receipt['profiles']); gamma=np.asarray(receipt['gamma_from_pilot'])
                for key in LABELS:
                    shape=true_r if key in ['known_shape','known_both'] else h
                    kap=true_k if key in ['known_kappa','known_both'] else r['kappa_hat']
                    p=components(z[held],shape,kap,profiles,refs[key],h); ps[key][held]=p
                    ev=[grid_calibrate(p[:,:,c],receipt['pilots'][c],512,4095,c)[0] for c in range(2)]
                    es[key][held]=(1-gamma)*ev[0]+gamma*ev[1]
                    if key=='R2': component_e[held]=np.stack(ev,axis=2)
            if not np.array_equal(ps['R2'],old_p) or not np.array_equal(es['R2'],old_e):
                raise ValueError('base scores/evidence replay differs')
            decisions={k:ebh(v,.05) for k,v in es.items()}
            metrics={k:score(v,truth) for k,v in decisions.items()}
            secondary={
                'relaxed_e_ge20_NO_FAMILY_GUARANTEE':score(old_e>=20,truth),
                'ordinary_p_le05_NO_FAMILY_GUARANTEE':score(old_p[:,:,0]<=.05,truth),
                'projection_p_le05_NO_FAMILY_GUARANTEE':score(old_p[:,:,1]<=.05,truth),
                'fixed_projection_component_ebh':score(ebh(component_e[:,:,1],.05),truth),
                'fixed_ordinary_component_ebh':score(ebh(component_e[:,:,0],.05),truth)}
            ap=dest.with_name(dest.name.replace('.json.gz','-arrays.npz'))
            with ap.open('xb') as stream:
                np.savez_compressed(stream,**{'p_'+k:v for k,v in ps.items()},**{'e_'+k:v for k,v in es.items()},**{'decision_'+k:v for k,v in decisions.items()})
            row.update(status='completed',metrics=metrics,secondary=secondary,
                       old_methods=r['metrics'],kappa_ratio=r['kappa_hat']/true_k,
                       pilot_focus_below_projection_floor=sum(f['pilots'][1]['threshold']<1/4096 for f in r['folds']),
                       projection_rank_floor_true_fraction=float(np.mean(old_p[:,:,1][truth]==1/4096)),
                       reference_quantiles={k:np.quantile(v,[.95,.99,.999]).tolist() for k,v in refs.items()},
                       source_sha256=sha(source),source_input_sha256=r['input_sha256'],
                       artifact_path=ap.relative_to(out).as_posix(),artifact_sha256=sha(ap))
    except Exception as exc:
        row.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
    row['seconds']=time.perf_counter()-started
    with gzip.open(dest,'xt',encoding='utf8') as stream: json.dump(row,stream,default=js,allow_nan=False)
    return {'case':case,'rep':rep,'path':dest.relative_to(out).as_posix(),'sha256':sha(dest),'status':row['status']}

def freeze(out):
    verify_r2()
    out.mkdir()
    for src,name in [(PHASE/'protocol-loss.json','protocol.json'),(Path(__file__),'loss_diagnostic.py'),(PHASE/'r3_common.py','r3_common.py')]:
        shutil.copy2(src,out/name)
    write(out/'freeze.json',{'utc':datetime.now(timezone.utc).isoformat(),'files':{p.name:sha(p) for p in out.iterdir()},'r2_manifest_sha256':sha(R2/'DELIVERY_MANIFEST.json')})

def run(out):
    verify_r2(); frozen=read(out/'freeze.json')
    for name,digest in frozen['files'].items():
        if sha(out/name)!=digest: raise ValueError('diagnostic source changed')
    p=read(out/'protocol.json')
    write(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid()})
    started=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=p['workers']) as pool:
        for item in pool.map(one,[(str(out),c,r) for c in p['cases'] for r in range(p['repetitions'])],timeout=1800):
            rows.append(item)
            if len(rows)%24==0: print({'complete':len(rows),'total':len(p['cases'])*p['repetitions'],'failures':sum(x['status']!='completed' for x in rows)},flush=True)
    write(out/'index.json',{'rows':rows,'seconds':time.perf_counter()-started})

def analyze(out):
    index=read(out/'index.json'); p=read(out/'protocol.json'); records=[]
    if len(index['rows'])!=len(p['cases'])*p['repetitions']: raise ValueError('incomplete')
    for item in index['rows']:
        if sha(out/item['path'])!=item['sha256']: raise ValueError('record changed')
        r=json.load(gzip.open(out/item['path'],'rt',encoding='utf8'))
        if r['status']!='completed': raise ValueError('failure retained: '+r.get('error',''))
        records.append(r)
    result=[]
    for case in p['cases']:
        rows=[r for r in records if r['case']==case]
        summary={'case':case,'n':len(rows),'methods':{},'paired_vs_R2':{},'secondary':{}}
        for k in LABELS:
            summary['methods'][k]={m:float(np.mean([r['metrics'][k][m] for r in rows])) for m in ['power','fdp','tp','fp']}
            delta=np.array([r['metrics'][k]['power']-r['metrics']['R2']['power'] for r in rows])
            summary['paired_vs_R2'][k]={'mean':float(delta.mean()),'mcse':float(delta.std(ddof=1)/np.sqrt(len(rows)))}
        for k in rows[0]['secondary']:
            summary['secondary'][k]={m:float(np.mean([r['secondary'][k][m] for r in rows])) for m in ['power','fdp']}
        summary['mean_strong_power']=float(np.mean([r['old_methods']['B_strong']['power'] for r in rows]))
        summary['rank_floor_true_fraction']=float(np.mean([r['projection_rank_floor_true_fraction'] for r in rows]))
        summary['pilot_focus_below_floor_fraction']=float(np.mean([r['pilot_focus_below_projection_floor']/4 for r in rows]))
        result.append(summary)
        print(case,{k:round(v['power'],4) for k,v in summary['methods'].items()},flush=True)
    write(out/'summary.json',{'stage':'DIAGNOSTIC_ONLY','rows':result,'index_sha256':sha(out/'index.json')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','run','analyze']);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out).resolve();globals()[a.action](out)
