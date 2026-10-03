"""D006: deterministic attribution, NOT a new valid comparison method.

Remove max-selection protection on the SAME observed selected branch and SAME
outer tuples only to quantify its cost. Those naive p/e values are NOT valid
after selection. Never feed this diagnostic into a deployable selector.
"""
from pathlib import Path
import sys,gzip,json,numpy as np
ROOT=Path(__file__).resolve().parent;OLD=ROOT/'D005'
sys.path.insert(0,str(OLD))
from r5_common import read,write,sha,R4,fc,grid,exp
from r5_kernel import components
from codesigned_profile import CodesignedProfile

class BranchDiagnostic:
    def __init__(self,p,borrow):
        self.scores=np.sort(p.logbridge if borrow else p.logtarget);self.m=p.m
    def pvalues(self,stat,borrow):
        stat=np.asarray(stat);out=np.ones_like(stat);pos=stat>0
        out[pos]=(1+self.m-np.searchsorted(self.scores,np.log(stat[pos]),side='left'))/(self.m+1)
        return out

def run():
    out=ROOT/'D006';out.mkdir()
    write(out/'protocol.json',{'stage':'DETERMINISTIC_DEVELOPMENT_ATTRIBUTION_NOT_VALID_METHOD',
        'source':'D005 all80 existing families','source_freeze':sha(OLD/'freeze.json'),
        'script_sha':sha(__file__),'stop':'all80, no new randomness, no confirmatory claims',
        'purpose':'Distinguish selected scale and exact max-selection protection costs; no truth used to make decisions',
        'warning':'naive_selected values lack post-selection validity; their apparent gains are not achievable valid Power evidence'})
    for f,d in read(OLD/'freeze.json')['files'].items():
        if sha(OLD/f)!=d:raise ValueError('old freeze changed')
    rows=[]
    for item in read(OLD/'index.json')['rows']:
        path=OLD/item['path']
        if sha(path)!=item['sha']:raise ValueError('raw hash')
        row=json.load(gzip.open(path,'rt',encoding='utf8'))
        if row['method_status']!='completed':raise ValueError('must explicitly handle failed source')
        if sha(OLD/row['evidence_path'])!=row['evidence_sha']:raise ValueError('evidence hash')
        with np.load(OLD/row['evidence_path'],allow_pickle=False) as f:ev=dict(f)
        arts=row['source_artifacts']
        for k in ['observed','diagnostic']:
            if sha(R4/'C001'/arts[k]['path'])!=arts[k]['sha']:raise ValueError('R4 artifact hash')
        with np.load(R4/'C001'/arts['observed']['path'],allow_pickle=False) as f:z=f['z'];cs=f['calibration_source'];ct=f['calibration_target']
        with np.load(R4/'C001'/arts['diagnostic']['path'],allow_pickle=False) as f:truth=f['truth']
        a=row['reference_receipt']['meta']['source_weight'];borrow=row['calibration']['borrow']
        p=CodesignedProfile(ev['reference_es'],ev['reference_et'],ev['reference_base'],len(cs),len(ct),
              ev['reference_inner_target'],ev['reference_inner_bridge'],source_weight=a)
        naive=BranchDiagnostic(p,borrow);g=len(z);pcheck=np.empty_like(ev['p']);pn=np.empty_like(ev['p']);en=np.empty_like(ev['e'])
        for f,fold in enumerate(row['folds']):
            ids=np.arange(g)%4==f;dh=np.asarray(row['folds'][(f+3)%4]['inference_shape'])
            # fold(f+3)'s inference shape belongs to (f+2), the DIR fold here.
            h=np.asarray(fold['inference_shape']);profiles=np.asarray(fold['direction_profiles'])
            for ref,dest in [(p,pcheck),(naive,pn)]:
                dest[ids],_=components(z[ids],h,np.exp(row['calibration']['log_scale']),profiles,ref,dh,borrow)
            ee=[]
            for c in range(2):
                val,_=grid.grid_calibrate(pn[ids,:,c],fold['calibrators'][c],2*g,4095,c);ee.append(val)
            gamma=np.asarray(fold['gamma']);en[ids]=(1-gamma)*ee[0]+gamma*ee[1]
        if not np.array_equal(pcheck,ev['p']):raise ValueError('exact saved p replay failed')
        if not np.all(pn<=pcheck):raise ValueError('protection monotonicity failed')
        dn=fc.ebh(en,.05)
        if not np.all(~ev['decision']|dn):raise ValueError('naive eBH should include protected')
        samplers=[]
        def visit(x):
            if isinstance(x,dict):
                if 'max_log_envelope_excess' in x:samplers.append(x['max_log_envelope_excess'])
                for v in x.values():visit(v)
            elif isinstance(x,list):
                for v in x:visit(v)
        visit(row['reference_receipt'])
        rows.append({'case':row['case'],'rep':row['rep'],'source_record_sha':item['sha'],
            'protected':row['metrics']['A0'],'naive_selected_DIAGNOSTIC_INVALID':exp.score(dn,truth),
            'borrow':borrow,'source_weight':a,'count_weight':len(cs)/(len(cs)+len(ct)),
            'q_target':p.qt,'q_bridge':p.qb,'switch':p.c,
            'source_variance':row['reference_receipt']['meta']['source_variance_MC'],
            'target_variance':row['reference_receipt']['meta']['target_variance_MC'],
            'positive_envelope_excess_count':sum(x>0 for x in samplers),
            'maximum_envelope_excess':max(samplers),
            'fraction_pc_p_increased':float(np.mean(pcheck>pn)),
            'conditional_failures':{k:v['status'] for k,v in row['conditional_receipts'].items() if v['status']!='completed'},
            'seconds':row['seconds'],'conditional_target_seconds':row['conditional_receipts']['target']['seconds']})
    write(out/'raw.json',rows)
    summary=[]
    for c in range(10):
        rr=[r for r in rows if r['case']==c];has=rr[0]['protected']['power'] is not None
        s={'case':c,'n':len(rr),'mean_borrow':np.mean([r['borrow'] for r in rr]),
           'source_weight_mean':np.mean([r['source_weight'] for r in rr]),
           'source_weight_range':[min(r['source_weight'] for r in rr),max(r['source_weight'] for r in rr)],
           'protected_power':np.mean([r['protected']['power'] for r in rr]) if has else None,
           'naive_INVALID_power':np.mean([r['naive_selected_DIAGNOSTIC_INVALID']['power'] for r in rr]) if has else None,
           'protected_fdr':np.mean([r['protected']['fdp'] for r in rr]),
           'naive_INVALID_fdr':np.mean([r['naive_selected_DIAGNOSTIC_INVALID']['fdp'] for r in rr]),
           'mean_protection_extra_lost_tp':np.mean([r['naive_selected_DIAGNOSTIC_INVALID']['tp']-r['protected']['tp'] for r in rr]),
           'mean_seconds':np.mean([r['seconds'] for r in rr]),
           'mean_conditional_target_seconds':np.mean([r['conditional_target_seconds'] for r in rr])}
        summary.append(s);print(json.dumps(s),flush=True)
    write(out/'summary.json',{'rows':summary,'positive_envelope_receipts':sum(r['positive_envelope_excess_count'] for r in rows),
        'all_p_exactly_reproduced':True,'confirmation_used':0,'n_unique_families':80,
        'warning':'Naive selected figures remove required protection and are not a valid method or achievable oracle'})

if __name__=='__main__':run()
