"""D017 frozen deterministic localized-score diagnostic; no new draws."""
import gzip,json,shutil,time
from datetime import datetime,timezone
import numpy as np
from r5_common import PHASE,R4,sha,read,write,fc,exp
from localized_reference import profile,observed_e

def main():
    out=PHASE/'D017';out.mkdir(exist_ok=False)
    files=['replay_localized.py','localized_reference.py','test_localized_reference.py','LOCALIZED_JOINT_REFERENCE.md',
           'focused_kernel.py','focused_reference.py','joint_power_kernel.py','joint_power_reference.py',
           'ancillary_calibration.py','selection_profile.py','r5_common.py']
    for f in files:shutil.copy2(PHASE/f,out/f)
    write(out/'protocol.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':'DETERMINISTIC_DEVELOPMENT_DIAGNOSIS',
        'inputs':'ALL20D015families and reference arrays','new_draws':0,'candidate':'A0.11localizedjoint',
        'f_width':'2sigma','r_width':'sigma+logD','max_cells':256,'tolerance':.01,
        'stop':'20fixedreplays, no statistical confirmation or escalation ifweak'})
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in files+['protocol.json']}})
    rows=[]
    for item in read(PHASE/'D015/index.json')['rows']:
        start=time.perf_counter();rp=PHASE/'D015'/item['path'];assert sha(rp)==item['sha']
        r=json.load(gzip.open(rp,'rt',encoding='utf8'));ep=PHASE/'D015'/r['evidence_path'];assert sha(ep)==r['evidence_sha']
        with np.load(ep,allow_pickle=False) as f:arr=dict(f)
        cand=r['candidate'];ref=cand['reference'];meta=ref['meta'];qt=ref['thresholds'];u=cand['calibration']['observed_u']
        target=arr['reference_pivot']*np.exp(-arr['reference_error_target']/2)
        ru=arr['reference_error_source']-arr['reference_error_target']
        Q,receipt=profile(target,ru,u,a=meta['a'],c=meta['c'],tau=meta['tau'],qt=qt['target'],qb=qt['variance_bridge'],
                          bound=cand['calibration']['bound'],sigma=np.sqrt(meta['vs']+meta['vt']))
        evidence=np.empty_like(arr['e_focused_joint'])
        for fold in cand['folds']:
            held=np.arange(len(evidence))%4==fold['test'];gamma=np.asarray(fold['gamma'])
            em=observed_e(arr['statistic_marginal'][held],u,ref,Q);ee=observed_e(arr['statistic_projected'][held],u,ref,Q)
            evidence[held]=(1-gamma)*np.sort(em,axis=-1)[...,:3].mean(-1)+gamma*ee.min(-1)
        dec=fc.ebh(evidence,.05)
        dp=R4/'C001'/r['source_artifacts']['diagnostic']['path'];assert sha(dp)==r['source_artifacts']['diagnostic']['sha']
        with np.load(dp,allow_pickle=False) as f:truth=f['truth']
        row={'case':r['case'],'rep':r['rep'],'status':'completed','source_sha':sha(rp),'source_evidence_sha':sha(ep),
             'old_profile_Q':ref['Q'],'localized_profile':receipt,'metrics':dict(r['metrics'],localized=exp.score(dec,truth)),
             'seconds':time.perf_counter()-start}
        stem=out/f'case-{r["case"]:02}-rep-{r["rep"]:02}'
        with open(str(stem)+'.npz','xb') as f:np.savez_compressed(f,e=evidence,decision=dec)
        row['evidence_sha']=sha(str(stem)+'.npz');write(str(stem)+'.json',row);rows.append(row)
        print(r['case'],r['rep'],round(ref['Q'],4),round(Q,4),'cells',receipt['cells'],'power',row['metrics']['localized']['power'],flush=True)
    summary=[]
    for c in range(10):
        rr=[v for v in rows if v['case']==c]
        keys=['localized','focused_joint','focused_target','focused_count_bridge','focused_fixed_mix','old_conditional_mix','A07']
        means={k:{m:None if rr[0]['metrics'][k][m] is None else float(np.mean([r['metrics'][k][m] for r in rr]))
                  for m in ['power','fdp','tp','fp']} for k in keys}
        summary.append({'case':c,'means':means});print('summary',c,{k:v['power'] for k,v in means.items()},flush=True)
    write(out/'result.json',{'status':'DETERMINISTIC_DIAGNOSIS_COMPLETE_NOT_ACCEPTANCE','summary':summary,
        'rows':rows,'new_observed_or_reference_draws':0,'formal_confirmation':False,'EXTERNAL_REVIEW':'NOT_CONDUCTED'})

if __name__=='__main__':main()
