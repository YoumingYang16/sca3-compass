"""Frozen D018 score-only diagnostic: all20saved joint-reference families."""
import gzip,json,shutil,time,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from r5_common import PHASE,R4,sha,read,write,fc,exp
from focused_power_score import prepare_scores,scalar_values

FILES=['replay_powerfocus.py','focused_power_score.py','test_focused_power.py','FOCUSED_POWER_RULE.md',
       'focused_kernel.py','focused_reference.py','joint_power_kernel.py','joint_power_reference.py',
       'ancillary_calibration.py','selection_profile.py','r5_common.py']

def prepare():
    out=PHASE/'D018';out.mkdir(exist_ok=False)
    for f in FILES:shutil.copy2(PHASE/f,out/f)
    write(out/'protocol.json',{'utc':datetime.now(timezone.utc).isoformat(),'source':'ALL20D015savedtuples',
        'kind':'DETERMINISTIC_SCORE_DIAGNOSIS','new_draws':0,'power':4,'max_cells':256,'tolerance':.01,
        'stop':'20only, no newconfirmation or data','source_index_sha':sha(PHASE/'D015/index.json')})
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in FILES+['protocol.json']}})

def run():
    out=PHASE/'D018';assert Path(__file__).resolve()==out/'replay_powerfocus.py'
    for f,h in read(out/'freeze.json')['files'].items():assert sha(out/f)==h
    assert Path(sys.modules['focused_power_score'].__file__).resolve()==out/'focused_power_score.py'
    write(out/'started.json',{'utc':datetime.now(timezone.utc).isoformat()});rows=[]
    for item in read(PHASE/'D015/index.json')['rows']:
        start=time.perf_counter();rp=PHASE/'D015'/item['path'];assert sha(rp)==item['sha']
        r=json.load(gzip.open(rp,'rt',encoding='utf8'));ep=PHASE/'D015'/r['evidence_path'];assert sha(ep)==r['evidence_sha']
        with np.load(ep,allow_pickle=False) as f:arr=dict(f)
        v=r['candidate'];ref=v['reference'];u=v['calibration']['observed_u']
        target=arr['reference_pivot']*np.exp(-arr['reference_error_target']/2);ru=arr['reference_error_source']-arr['reference_error_target']
        budget=prepare_scores(target,ru,ref);evidence={};g=len(arr['e_focused_joint'])
        for fold in v['folds']:
            held=np.arange(g)%4==fold['test'];gamma=np.asarray(fold['gamma'])
            em=scalar_values(arr['statistic_marginal'][held],u,ref,budget)
            ee=scalar_values(arr['statistic_projected'][held],u,ref,budget)
            for name in em:
                if name not in evidence:evidence[name]=np.empty((g,2))
                evidence[name][held]=(1-gamma)*np.sort(em[name],axis=-1)[...,:3].mean(-1)+gamma*ee[name].min(-1)
        evidence['powerfocus_fixed_mix']=.5*evidence['powerfocus_target']+.5*evidence['powerfocus_count_bridge']
        evidence['powerfocus_variance_mix']=.5*evidence['powerfocus_target']+.5*evidence['powerfocus_variance_bridge']
        decision={k:fc.ebh(e,.05) for k,e in evidence.items()}
        dp=R4/'C001'/r['source_artifacts']['diagnostic']['path'];assert sha(dp)==r['source_artifacts']['diagnostic']['sha']
        with np.load(dp,allow_pickle=False) as f:truth=f['truth']
        row={'case':r['case'],'rep':r['rep'],'source_sha':sha(rp),'source_evidence_sha':sha(ep),
             'budget':budget,'metrics':dict(r['metrics'],**{k:exp.score(d,truth) for k,d in decision.items()}),
             'seconds':time.perf_counter()-start,'status':'completed'}
        dest=out/f'case-{r["case"]:02}-rep-{r["rep"]:02}'
        with open(str(dest)+'.npz','xb') as f:np.savez_compressed(f,**{'e_'+k:e for k,e in evidence.items()},**{'decision_'+k:d for k,d in decision.items()})
        row['evidence_sha']=sha(str(dest)+'.npz');write(str(dest)+'.json',row);rows.append(row)
        print(r['case'],r['rep'],{k:row['metrics'][k]['power'] for k in ['powerfocus_joint','powerfocus_max','powerfocus_target','powerfocus_fixed_mix','unfocused_joint_MC']},flush=True)
    summary=[]
    for c in range(10):
        rr=[v for v in rows if v['case']==c];means={}
        for name in rr[0]['metrics']:
            means[name]={k:None if rr[0]['metrics'][name][k] is None else float(np.mean([r['metrics'][name][k] for r in rr])) for k in ['power','fdp','tp','fp']}
        summary.append({'case':c,'means':means})
    write(out/'result.json',{'status':'DETERMINISTIC_DIAGNOSIS_COMPLETE_NOT_ACCEPTANCE','summary':summary,'rows':rows,
        'new_observational_or_reference_draws':0,'formal_confirmation':False,'EXTERNAL_REVIEW':'NOT_CONDUCTED'})
    for f,h in read(out/'freeze.json')['files'].items():assert sha(out/f)==h

if __name__=='__main__':
    if Path(__file__).resolve().parent==PHASE:prepare()
    else:run()
