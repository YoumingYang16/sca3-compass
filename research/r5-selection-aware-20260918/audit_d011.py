"""Deterministic saved-result audit/paired summaries; no fitted or simulated data."""
import gzip,json
import numpy as np
from r5_common import PHASE,R4,ROOT,sha,read,write,fc,exp

def load(name,expected):
    root=PHASE/name; freeze=read(root/'freeze.json');idx=read(root/'index.json');rows=[]
    for f,h in freeze['files'].items():
        assert sha(root/f)==h,('frozen file',name,f)
    for entry in idx['rows']:
        assert sha(root/entry['path'])==entry['sha']
        r=json.load(gzip.open(root/entry['path'],'rt',encoding='utf8'))
        assert r['status']=='completed','hard failure needs explicit handling'
        assert r['freeze']==sha(root/'freeze.json')
        assert sha(ROOT/r['source_record'])==r['source_sha']
        old=json.load(gzip.open(ROOT/r['source_record'],'rt',encoding='utf8'))
        assert r['algorithm_seed']==old['algorithm_seed']
        for tag,a in r['source_artifacts'].items():
            assert a['path']==old[tag+'_path'] and a['sha']==old[tag+'_sha256']
            assert sha(R4/'C001'/a['path'])==a['sha']
        assert sha(root/r['evidence_path'])==r['evidence_sha']
        with np.load(root/r['evidence_path'],allow_pickle=False) as f:ev=dict(f)
        with np.load(R4/'C001'/old['evidence_path'],allow_pickle=False) as f:base=dict(f)
        with np.load(R4/'C001'/old['diagnostic_path'],allow_pickle=False) as f:truth=f['truth']
        for method in ['target','source_bound','bridge','strong_target','strong_pool_bound']:
            assert np.array_equal(ev['decision_'+method],base['decision_'+method])
        assert np.array_equal(ev['decision'],fc.ebh(ev['e'],.05))
        for method,reported in r['metrics'].items():
            if method.startswith('previous_'):continue
            d=ev['decision'] if method=='A0' else ev['decision_'+method]
            assert exp.score(d,truth)==reported,('score',name,r['case'],r['rep'],method)
        for tag,ee in [('fixed_e_mix',.5*base['e_target']+.5*base['e_bridge']),
                       ('conditional_mix',ev['e_conditional_mix']),
                       ('conditional_variance_mix',ev['e_conditional_variance_mix'])]:
            assert np.array_equal(ev['decision_'+tag],fc.ebh(ee,.05))
        r['batch']=name;rows.append(r)
    assert len(rows)==len(expected)
    assert {(r['case'],r['rep']) for r in rows}==expected
    return rows,{'batch':name,'n':len(rows),'index_sha':sha(root/'index.json'),
        'freeze_sha':sha(root/'freeze.json'),'method_failures':sum(r['method_status']!='completed' for r in rows),
        'conditional_failures':sum(v['status']!='completed' for r in rows for v in r['conditional_receipts'].values()),
        'candidate_seconds':sum(r['seconds'] for r in rows)}

METHODS=['target','source_bound','bridge','fixed_e_mix','conditional_target',
         'conditional_source_bound','conditional_bridge','conditional_mix',
         'conditional_variance_bridge','conditional_variance_mix','strong_target','strong_pool_bound']
def paired(rr,k):
    d=np.array([r['metrics']['A0']['power']-r['metrics'][k]['power'] for r in rr])
    tp=np.array([r['metrics']['A0']['tp']-r['metrics'][k]['tp'] for r in rr])
    return {'difference':float(d.mean()),'SE_descriptive':float(d.std(ddof=1)/np.sqrt(len(d))),
            'range':[float(d.min()),float(d.max())],'mean_extra_true_claims':float(tp.mean())}
def summary(rows):
    out=[]
    for c in range(10):
        rr=sorted([r for r in rows if r['case']==c],key=lambda r:r['rep'])
        entry={'case':c,'n':len(rr),'metrics':{},'paired':{}}
        for k in ['A0']+METHODS:
            entry['metrics'][k]={f:None if rr[0]['metrics'][k][f] is None else float(np.mean([r['metrics'][k][f] for r in rr]))
                                 for f in ['power','fdp','tp','fp']}
            if k!='A0' and entry['metrics'][k]['power'] is not None:entry['paired'][k]=paired(rr,k)
        out.append(entry)
    # Case1/8 share observation streams: keep all six cases inside a rep cluster.
    core=read(R4/'C001/protocol.json')['core_cases'];clusters=[]
    for rep in sorted({r['rep'] for r in rows}):
        rr=[r for r in rows if r['rep']==rep and r['case'] in core]
        assert len(rr)==len(core)
        clusters.append({k:float(np.mean([r['metrics']['A0']['power']-r['metrics'][k]['power'] for r in rr])) for k in METHODS})
    return {'cases':out,'equal_weight_core_rep_cluster':{k:{'difference':float(np.mean([r[k] for r in clusters])),
            'SE_descriptive':float(np.std([r[k] for r in clusters],ddof=1)/np.sqrt(len(clusters)))} for k in METHODS}}

if __name__=='__main__':
    old,a=load('D009',{(c,r) for c in range(10) for r in range(8)})
    new,b=load('D011',{(c,r) for c in range(10) for r in range(8,32)})
    write(PHASE/'D011_AUDIT_ANALYSIS_v2.json',{'status':'AUDIT_PASS_DEVELOPMENT_ONLY','script_sha':sha(__file__),
        'audits':[a,b],'new24':summary(new),'old8':summary(old),'combined32_descriptive':summary(old+new),
        'distinct_observed_families':320,'formal_R5_confirmation':0,
        'limitations':['A07/A071 successful statistic identity supported by source diff and one bitwise replay; not all-input replay',
         'SEs are descriptive, no confidence/acceptance claim after method selection',
         'Combined32 mixes development ages; new24 must be shown separately',
         'Source C001 paired cases1/8 are dependent; core clusters by repetition',
         'Truth used solely by deterministic scoring; never method input']})
    for c in summary(new)['cases']:
        print(c['case'],{k:vv for k,vv in c['paired'].items() if k in ['fixed_e_mix','conditional_mix','conditional_target']})
