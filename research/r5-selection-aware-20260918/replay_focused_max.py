"""D016 max-score control and deterministic score/budget attribution."""
import gzip,json,shutil
import numpy as np
from r5_common import PHASE,R4,sha,read,write,fc,exp
from focused_kernel import scalar_e,indicators
from focused_reference import mc_e

def main():
    out=PHASE/'D016_2';out.mkdir(exist_ok=False)
    files=['D016_RULE.md','replay_focused_max.py','focused_kernel.py','focused_reference.py',
           'joint_power_kernel.py','joint_power_reference.py','ancillary_calibration.py','selection_profile.py','r5_common.py']
    for f in files:shutil.copy2(PHASE/f,out/f)
    write(out/'protocol.json',{'source':'D015 all20records','no_new_draws':True,'no_formal_confirmation':True})
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in files+['protocol.json']}})
    rows=[]
    for item in read(PHASE/'D015/index.json')['rows']:
        rp=PHASE/'D015'/item['path'];assert sha(rp)==item['sha'];r=json.load(gzip.open(rp,'rt',encoding='utf8'))
        ep=PHASE/'D015'/r['evidence_path'];assert sha(ep)==r['evidence_sha']
        with np.load(ep,allow_pickle=False) as f:arr=dict(f)
        cand=r['candidate'];ref=cand['reference'];q=ref['thresholds'];a=ref['meta']['a'];u=cand['calibration']['observed_u']
        rt=arr['reference_pivot']*np.exp(-arr['reference_error_target']/2)
        ru=arr['reference_error_source']-arr['reference_error_target']
        it=rt>q['target'];ib=indicators(rt,ru,a,q['variance_bridge']).astype(bool)
        total=int(np.sum(it|ib));intersection=int(np.sum(it&ib))
        assert total==int(np.sum(it)+np.sum(ib))-intersection
        ev={k:np.empty_like(arr['e_'+k]) for k in r['metrics'] if k.startswith('focused_') and k not in ['focused_fixed_mix','focused_variance_mix']}
        maxev=np.empty_like(arr['e_focused_joint']);loss=[]
        for fold in cand['folds']:
            held=np.arange(len(maxev))%4==fold['test'];gamma=np.asarray(fold['gamma'],float);mm=arr['statistic_marginal'][held];pp=arr['statistic_projected'][held]
            em=scalar_e(mm,u,ref);ee=scalar_e(pp,u,ref)
            for name in ev:ev[name][held]=(1-gamma)*np.sort(em[name],axis=-1)[...,:3].mean(-1)+gamma*ee[name].min(-1)
            def maxscalar(t):
                raw=np.maximum(indicators(t,u,0.,q['target']),indicators(t,u,a,q['variance_bridge']))
                return mc_e(raw,total,ref['outer_draws'])
            m,p=maxscalar(mm),maxscalar(pp)
            maxev[held]=(1-gamma)*np.sort(m,axis=-1)[...,:3].mean(-1)+gamma*p.min(-1)
        ev['focused_fixed_mix']=.5*ev['focused_target']+.5*ev['focused_count_bridge']
        ev['focused_variance_mix']=.5*ev['focused_target']+.5*ev['focused_variance_bridge']
        assert all(np.array_equal(e,arr['e_'+name]) for name,e in ev.items()),'scalar replay differs'
        assert all(np.array_equal(fc.ebh(e,.05),arr['decision_'+name]) for name,e in ev.items())
        dp=R4/'C001'/r['source_artifacts']['diagnostic']['path'];assert sha(dp)==r['source_artifacts']['diagnostic']['sha']
        with np.load(dp,allow_pickle=False) as f:truth=f['truth']
        md=fc.ebh(maxev,.05)
        row={'case':r['case'],'rep':r['rep'],'source_sha':sha(rp),'all_focused_e_decisions_exact':True,
             'reference_counts':{'target':int(it.sum()),'bridge':int(ib.sum()),'intersection':intersection,'union':total},
             'joint_profile_upper_sum':ref['Q'],'joint_profile_lower_sum':ref['cover']['evaluated_lower_sum'],
             'borrow_probability':cand['calibration']['borrow_probability'],
             'metrics':dict(r['metrics'],focused_max=exp.score(md,truth))}
        dest=out/f'case-{r["case"]:02}-rep-{r["rep"]:02}'
        with open(str(dest)+'.npz','xb') as f:np.savez_compressed(f,e=maxev,decision=md)
        row['max_evidence_sha']=sha(str(dest)+'.npz');write(str(dest)+'.json',row);rows.append(row)
    sums=[]
    for c in range(10):
        rr=[r for r in rows if r['case']==c]
        powers={k:None if rr[0]['metrics'][k]['power'] is None else float(np.mean([r['metrics'][k]['power'] for r in rr]))
                for k in ['focused_joint','focused_max','focused_variance_bridge','focused_target','focused_fixed_joint_score','A07']}
        sums.append({'case':c,'power':powers});print(c,powers,flush=True)
    write(out/'result.json',{'status':'DETERMINISTIC_ATTRIBUTION_COMPLETE_NOT_ACCEPTANCE','summary':sums,'rows':rows,
        'new_observed_or_reference_samples':0,'EXTERNAL_REVIEW':'NOT_CONDUCTED'})

if __name__=='__main__':main()
