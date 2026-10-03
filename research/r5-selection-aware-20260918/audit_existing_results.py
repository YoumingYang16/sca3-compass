"""Read-only closure of batch/prior-baseline provenance gaps; no algorithms run."""
from pathlib import Path
import gzip,json,numpy as np
from r5_common import PHASE,R4,sha,read,write

def main():
    batches=['D001','D002','D003','D004','D005','D007','D008','D009'];records={};report=[]
    for name in batches:
        root=PHASE/name;freeze=read(root/'freeze.json');idx=read(root/'index.json')
        for f,d in freeze['files'].items():
            if sha(root/f)!=d:raise ValueError('frozen code mismatch:'+name+'/'+f)
        seen=set();rr=[]
        for item in idx['rows']:
            path=root/item['path']
            if sha(path)!=item['sha']:raise ValueError('raw record mismatch')
            row=json.load(gzip.open(path,'rt',encoding='utf8'));key=(row['case'],row['rep'])
            if key in seen:raise ValueError('duplicate family')
            seen.add(key)
            if row['status']!='completed':raise ValueError('hard failure must be explicitly audited')
            if row['freeze']!=sha(root/'freeze.json'):raise ValueError('record freeze mismatch')
            source=PHASE.parent.parent/row['source_record']
            if sha(source)!=row['source_sha']:raise ValueError('R4 source record mismatch')
            old=json.load(gzip.open(source,'rt',encoding='utf8'))
            if row['algorithm_seed']!=old['algorithm_seed']:raise ValueError('source reference seed identity')
            for k,a in row['source_artifacts'].items():
                if a['path']!=old[k+'_path'] or a['sha']!=old[k+'_sha256']:raise ValueError('source input path/hash identity')
                if sha(R4/'C001'/a['path'])!=a['sha']:raise ValueError('source input byte mismatch')
            if sha(root/row['evidence_path'])!=row['evidence_sha']:raise ValueError('evidence hash')
            with np.load(root/row['evidence_path'],allow_pickle=False) as f:ev=dict(f)
            with np.load(R4/'C001'/old['evidence_path'],allow_pickle=False) as f:
                for method in ['target','source_bound','bridge','strong_target','strong_pool_bound']:
                    if not np.array_equal(ev['decision_'+method],f['decision_'+method]):raise ValueError('frozen ordinary/strong decision mismatch')
            for tag,prior in [('previous_A04','D005'),('previous_A05','D007'),('previous_A06','D008')]:
                if 'decision_'+tag not in ev:continue
                previous=records[prior][key]
                if row['source_sha']!=previous['source_sha'] or row['source_artifacts']!=previous['source_artifacts']:
                    raise ValueError('cross-version observational identity mismatch')
                with np.load(PHASE/prior/previous['evidence_path'],allow_pickle=False) as f:
                    if not np.array_equal(ev['decision_'+tag],f['decision']):raise ValueError('prior decision mismatch')
            rr.append(row)
        expected={(c,r) for c in range(10) for r in range(8)}
        if seen!=expected:raise ValueError('case/repetition set differs')
        records[name]={(r['case'],r['rep']):r for r in rr}
        report.append({'batch':name,'freeze_sha':sha(root/'freeze.json'),'index_sha':sha(root/'index.json'),
            'frozen_files':len(freeze['files']),'families':len(rr),'candidate_numerical_failures':sum(r['method_status']!='completed' for r in rr),
            'conditional_comparator_failures':sum(v['status']!='completed' for r in rr for v in r.get('conditional_receipts',{}).values()),
            'seconds_candidate_sum':sum(r['seconds'] for r in rr),'checks':'code/raw/evidence/source/paired-prior identity PASS'})
    write(PHASE/'AUDIT_EXISTING_RESULTS.json',{'status':'PROVENANCE_AUDIT_PASS_NOT_STATISTICAL_ACCEPTANCE',
        'script_sha':sha(__file__),'batches':report,'distinct_observation_families':80,
        'not_counted_as_independent':'version repeats, genes, tests, quadrature patterns',
        'formal_R5_confirmation_count':0,'excluded_from_batch_audit':['D000 scalar diagnostic','D006 deterministic attribution','M001 quadrature'],
        'limitations':['Does not prove mathematics or prior experimental honesty','D004/D005 known numerical hazards not retroactively repaired','No protected data accessed']})
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
