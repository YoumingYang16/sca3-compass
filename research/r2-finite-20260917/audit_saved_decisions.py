"""Full deterministic e-BH replay of saved PB evidence, no new inference fits."""
from pathlib import Path
import sys
PHASE=Path(__file__).resolve().parent
sys.path.insert(0,str(PHASE/'C001'))
import gzip,json
import numpy as np
from experiment import read,write,sha
from provenance import verify_freeze
from finite_calibration import ebh


def main():
    out=PHASE/'C001'; verify_freeze(out); index=read(out/'index.json'); count=0; methods=0
    for item in index['rows']:
        path=out/item['path']
        if sha(path)!=item['sha256']: raise ValueError('record differs')
        with gzip.open(path,'rt',encoding='utf-8') as stream: row=json.load(stream)
        ep=out/row['evidence_path']
        if sha(ep)!=row['evidence_sha256']: raise ValueError('evidence differs')
        with np.load(ep,allow_pickle=False) as a:
            for key in ['PB_main','PB_grid','PB_ordinary','PB_grid_ordinary']+list(row['sensitivity']):
                if not np.array_equal(ebh(a['e_'+key],.05),a['decision_'+key]): raise ValueError('evidence/decision differs')
                methods+=1
            if np.any(a['decision_PB_main'] & ~a['decision_PB_grid']): raise ValueError('grid rejection nesting differs')
            if not np.isfinite(a['p']).all() or np.any((a['p']<=0)|(a['p']>1)): raise ValueError('invalid p')
            if np.any(a['p']*4096!=np.floor(a['p']*4096)): raise ValueError('rank grid differs')
        count+=1
    write(PHASE/'saved-decision-audit.json',{'families':count,'eBH_decision_replays':methods,
          'all_primary_grid_supersets':True,'meaning':'deterministic audit, no new observations',
          'freeze_sha256':sha(out/'freeze.json'),'index_sha256':sha(out/'index.json')})
    print(count,methods,flush=True)


if __name__=='__main__': main()
