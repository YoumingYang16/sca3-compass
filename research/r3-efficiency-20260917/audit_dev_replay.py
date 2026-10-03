"""Prove adding the median-only component did not alter prior DEV outputs."""
import gzip,json,sys
sys.dont_write_bytecode=True
from r3_common import PHASE,read,sha,write,verify_r2
import numpy as np

verify_r2();a=PHASE/'D002';b=PHASE/'D003';checks=0
ia=read(a/'index.json');ib=read(b/'index.json')
assert {(x['case'],x['rep']) for x in ia['rows']}=={(x['case'],x['rep']) for x in ib['rows']}
for old,new in zip(ia['rows'],ib['rows']):
    assert (old['case'],old['rep'])==(new['case'],new['rep'])
    assert sha(a/old['path'])==old['sha256'] and sha(b/new['path'])==new['sha256']
    x=json.load(gzip.open(a/old['path'],'rt'));y=json.load(gzip.open(b/new['path'],'rt'))
    assert x['status']==y['status']=='completed'
    assert x['algorithm_seed']==y['algorithm_seed'] and x['source_record']==y['source_record']
    with np.load(a/x['evidence_path'],allow_pickle=False) as p,np.load(b/y['evidence_path'],allow_pickle=False) as q:
        assert sha(a/x['evidence_path'])==x['evidence_sha256'] and sha(b/y['evidence_path'])==y['evidence_sha256']
        for key in p.files:
            assert np.array_equal(p[key],q[key]);checks+=1
    for key,value in x['metrics'].items():assert value==y['metrics'][key]
write(PHASE/'D003-replay-audit.json',{'families':len(ia['rows']),'identical_arrays':checks,
      'D002_index_sha256':sha(a/'index.json'),'D003_index_sha256':sha(b/'index.json'),
      'status':'all_preexisting_outputs_exactly_equal','scope':'not additional independent data'})
print({'families':len(ia['rows']),'exact_arrays':checks})
