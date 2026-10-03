"""D013 fixed80 saved-tuple mechanism replay. No new randomness or fitting."""
import gzip,json,shutil,time
import numpy as np
from r5_common import PHASE,R4,sha,read,write,fc,grid,exp
from r5_kernel import components
from switch_closure import SwitchClosure
from soft_drift_closure import SoftDriftClosure

def main():
    out=PHASE/'D013';out.mkdir(exist_ok=False);old=PHASE/'D007';rows=[]
    files=['replay_soft_closure.py','soft_drift_closure.py','test_soft_drift_closure.py','SOFT_DRIFT_CLOSURE.md',
           'r5_common.py','r5_kernel.py','conditional_reference.py','switch_closure.py','selection_profile.py',
           'continuous_profile.py','codesigned_profile.py','ancillary_calibration.py','selective_reference.py']
    write(out/'protocol.json',{'id':'D013','kind':'DETERMINISTIC_SAVED_REFERENCE_REPLAY_DEVELOPMENT',
        'source':'D007 all10cases rep0-7','source_freeze':sha(old/'freeze.json'),'candidate':'A0.8-soft-drift-closure',
        'stop':'exact80, no fresh data/randomnumbers/fit; not formalconfirmation',
        'question':'Does eliminating separate selected references via complete-rule monotone score improve utility? G1unestablished; no largerbatch.'})
    for f in files:shutil.copy2(PHASE/f,out/f)
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in files+['protocol.json']}})
    for f,d in read(old/'freeze.json')['files'].items():assert sha(old/f)==d
    for it in read(old/'index.json')['rows']:
        tick=time.perf_counter();path=old/it['path'];assert sha(path)==it['sha']
        r=json.load(gzip.open(path,'rt',encoding='utf8'));assert r['method_status']=='completed'
        assert sha(old/r['evidence_path'])==r['evidence_sha']
        with np.load(old/r['evidence_path'],allow_pickle=False) as f:ev=dict(f)
        arts=r['source_artifacts']
        for k in ['observed','diagnostic']:assert sha(R4/'C001'/arts[k]['path'])==arts[k]['sha']
        with np.load(R4/'C001'/arts['observed']['path'],allow_pickle=False) as f:z=f['z'];cs=f['calibration_source'];ct=f['calibration_target']
        with np.load(R4/'C001'/arts['diagnostic']['path'],allow_pickle=False) as f:truth=f['truth']
        a=r['reference_receipt']['meta']['source_weight'];c=r['calibration']['switch_threshold']
        args=[ev['reference_'+k] for k in ['es','et','base']]
        inn=[ev['reference_'+k] for k in ['inner_target','inner_bridge']]
        check=SwitchClosure(*args,len(cs),len(ct),*inn,source_weight=a,switch_threshold=c)
        new=SoftDriftClosure(*args,a,c,*inn);new.observed_u=r['calibration']['observed_slack']
        mt=r['reference_receipt']['centering'][1];kt=r['calibration']['target_geometric']*np.exp(-mt)
        pcheck=np.empty_like(ev['p']);pn=np.empty_like(ev['p']);en=np.empty_like(ev['e']);g=len(z)
        for ff,fold in enumerate(r['folds']):
            ids=np.arange(g)%4==ff;dh=np.asarray(r['folds'][(ff+3)%4]['inference_shape'])
            h=np.asarray(fold['inference_shape']);profiles=np.asarray(fold['direction_profiles'])
            pcheck[ids],_=components(z[ids],h,np.exp(r['calibration']['log_scale']),profiles,check,dh,r['calibration']['borrow'])
            pn[ids],_=components(z[ids],h,kt,profiles,new,dh,False)
            ee=[grid.grid_calibrate(pn[ids,:,j],fold['calibrators'][j],2*g,4095,j)[0] for j in range(2)]
            gamma=np.asarray(fold['gamma']);en[ids]=(1-gamma)*ee[0]+gamma*ee[1]
        assert np.array_equal(pcheck,ev['p']),'old p replay mismatch'
        d=fc.ebh(en,.05)
        a07path=PHASE/f"D009/raw/case-{r['case']:02}/rep-{r['rep']:05}.json.gz"
        a07=json.load(gzip.open(a07path,'rt',encoding='utf8'))
        assert a07['source_sha']==r['source_sha'] and a07['source_artifacts']==r['source_artifacts']
        rr={'case':r['case'],'rep':r['rep'],'source_sha':it['sha'],'source_observation_artifacts':arts,
            'comparison_source_sha':sha(a07path),'metrics':{**{k:v for k,v in a07['metrics'].items() if not k.startswith('previous_')},
                'A0.7':a07['metrics']['A0'],'A0.5':r['metrics']['A0'],'A0.8':exp.score(d,truth)},
            'source_reference_seconds':r['reference_seconds'],'replay_seconds':time.perf_counter()-tick,
            'normalizer_fallback':new.normalizer_fallback,'log_normalizers':[new.logqt,new.logqb],
            'all_old_p_exact':True,'observed_u':new.observed_u,'meta_a':a,'meta_c':c}
        rr['metrics'].pop('A0')
        dest=out/f"raw/case-{r['case']:02}-rep-{r['rep']:05}.npz";dest.parent.mkdir(exist_ok=True)
        with dest.open('xb') as f:np.savez_compressed(f,p=pn,e=en,decision=d)
        rr['evidence']=dest.relative_to(out).as_posix();rr['evidence_sha']=sha(dest)
        write(dest.with_suffix('.json'),rr);rows.append(rr)
    write(out/'raw.json',rows)
    summary=[]
    for c in range(10):
        rr=[r for r in rows if r['case']==c];metrics={}
        for k in rr[0]['metrics']:
            metrics[k]={m:None if rr[0]['metrics'][k][m] is None else float(np.mean([r['metrics'][k][m] for r in rr])) for m in ['power','fdp','tp','fp']}
        summary.append({'case':c,'n':len(rr),'metrics':metrics})
        print(c,{k:v['power'] for k,v in metrics.items() if k in ['A0.8','A0.7','conditional_target','conditional_mix','fixed_e_mix']},flush=True)
    write(out/'summary.json',{'status':'COMPLETE_DEVELOPMENT_DIAGNOSTIC_NOT_ACCEPTANCE','rows':summary,'old_p_exact_all80':True,
        'no_new_random_draws':True,'confirmation_used':0,'important_novelty':'UNESTABLISHED'})
    for f,d in read(out/'freeze.json')['files'].items():assert sha(out/f)==d

if __name__=='__main__':main()
