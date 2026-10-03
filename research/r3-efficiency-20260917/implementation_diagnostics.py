"""Descriptive role-convergence audit of existing records, not a new comparison.

Planned2026-09-17 while C001 runs, before any confirmatory outcome inspection.
No model fits, truth-driven selection, intervals, acceptance gate or extra n.
Fold counts describe code paths; they are NOT independent statistical repeats.
"""
from pathlib import Path
from collections import Counter
import gzip,json,sys
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent;OUT=BASE/'C001'
sys.path.insert(0,str(OUT))
from r3_common import read,write,sha

def main():
    idx=read(OUT/'index.json');p=read(OUT/'protocol.json')
    if not idx['complete'] or len(idx['rows'])!=15360:raise ValueError('wait for complete fixed batch')
    keys=['direction_converged','pilot_converged','scale_ablation_train_converged']
    stats={c:{'families':0,'recorded_folds':0,'statuses':Counter(),
              'nonconverged_folds':Counter(),'families_with_nonconverged_role':Counter()} for c in range(15)}
    for item in idx['rows']:
        path=OUT/item['path']
        if sha(path)!=item['sha256']:raise ValueError('record hash changed')
        with gzip.open(path,'rt',encoding='utf8') as f:r=json.load(f)
        c=r['case'];v=stats[c];v['families']+=1;v['statuses'][r['status']]+=1
        folds=r.get('folds',[])
        if r['status']=='completed' and len(folds)!=4:raise ValueError('completed role count differs')
        v['recorded_folds']+=len(folds)
        for key in keys:
            flags=[f[key] for f in folds]
            if any(type(b) is not bool for b in flags):raise ValueError('convergence flag not boolean')
            count=sum(not b for b in flags);v['nonconverged_folds'][key]+=count
            v['families_with_nonconverged_role'][key]+=int(count>0)
    for c,v in stats.items():
        if v['families']!=1024:raise ValueError('scene count differs')
        v['name']=p['cases'][c]['name']
    write(BASE/'IMPLEMENTATION_DIAGNOSTICS.json',{'scope':'SUPPLEMENTAL_DESCRIPTIVE_ONLY; no new inferential comparisons or acceptance gates',
      'rows':stats,'index_sha256':sha(OUT/'index.json'),'freeze_sha256':sha(OUT/'freeze.json'),
      'limitations':'R2/V1 detailed role diagnostics were not archived in these R3 records. Do not infer them from a top-level completed status; V1 guard fallback count is separately in the formal summary.'})
    print(json.dumps(stats,indent=2))

if __name__=='__main__':main()
