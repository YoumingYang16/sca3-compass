"""Post-run identity gap closure; no new data, no change to frozen diagnostic."""
import sys,gzip,json
sys.dont_write_bytecode=True
from r3_common import PHASE,R2,read,write,sha,score,verify_r2
from provenance import check_record
from finite_calibration import ebh
import numpy as np

def main():
    verify_r2();out=PHASE/'D001';p=read(out/'protocol.json');fr=read(out/'freeze.json')
    assert (p['repetitions'],p['reference_draws'],p['shape_iterations'])==(24,4095,2)
    assert p['r2_manifest_sha256']==sha(R2/'DELIVERY_MANIFEST.json')
    for name,digest in fr['files'].items(): assert sha(out/name)==digest
    old=R2/'C001';op=read(old/'protocol.json');oi=read(old/'index.json')
    indexed={(x['case'],x['rep']):x for x in oi['rows']}
    index=read(out/'index.json');plan={(c,r) for c in p['cases'] for r in range(24)}
    assert len(index['rows'])==len(plan) and {(x['case'],x['rep']) for x in index['rows']}==plan
    records=[];checks=0
    for item in index['rows']:
        c,r=item['case'],item['rep'];rel=f'raw/case-{c:02}/rep-{r:05}.json.gz'
        assert item['path']==rel and sha(out/rel)==item['sha256']
        row=json.load(gzip.open(out/rel,'rt',encoding='utf8'))
        assert (row['case'],row['rep'])==(c,r) and row['status']==item['status']=='completed'
        assert row['freeze_sha256']==sha(out/'freeze.json')
        source=indexed[(c,r)]
        assert source['path']==rel and sha(old/rel)==source['sha256']==row['source_sha256']
        previous=json.load(gzip.open(old/rel,'rt',encoding='utf8'))
        check_record(previous,old,op,c,r,metrics=True)
        assert row['source_input_sha256']==previous['input_sha256']
        assert row['artifact_path']==rel.replace('.json.gz','-arrays.npz')
        assert sha(out/row['artifact_path'])==row['artifact_sha256']
        with np.load(old/previous['input_path'],allow_pickle=False) as a: truth=a['truth']
        with np.load(out/row['artifact_path'],allow_pickle=False) as a:
            assert set(a.files)=={prefix+k for prefix in ['p_','e_','decision_'] for k in p['method_labels']}
            for key in p['method_labels']:
                assert np.array_equal(ebh(a['e_'+key],.05),a['decision_'+key])
                assert score(a['decision_'+key],truth)==row['metrics'][key];checks+=1
        assert row['metrics']['R2']==previous['metrics']['PB_grid']
        records.append(row)
    summary=read(out/'summary.json')
    assert summary['index_sha256']==sha(out/'index.json')
    for s in summary['rows']:
        sub=[r for r in records if r['case']==s['case']]; assert len(sub)==s['n']==24
        for k in p['method_labels']:
            for metric in ['power','fdp','tp','fp']:
                assert s['methods'][k][metric]==float(np.mean([r['metrics'][k][metric] for r in sub]))
    write(PHASE/'D001-identity-audit.json',{'status':'passed_postrun_identity_and_score_checks',
      'families':len(records),'saved_decisions_checked':checks,'scope':'engineering verification, no new experiment',
      'diagnostic_freeze_sha256':sha(out/'freeze.json'),'index_sha256':sha(out/'index.json'),
      'r2_index_sha256':sha(old/'index.json'),'r2_protocol_sha256':sha(old/'protocol.json'),
      'summary_sha256':sha(out/'summary.json'),
      'qualification':'Historical workers did not attest module paths; original command invoked frozen executor. Saved dependency/source/result identities now checked; no invented contemporaneous receipts.'})
    print({'status':'passed','families':len(records),'decisions':checks})

if __name__=='__main__':main()
