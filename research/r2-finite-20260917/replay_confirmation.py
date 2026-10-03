"""Four prespecified frozen C001 replays, no new confirmation or selection."""
from pathlib import Path
import sys
PHASE=Path(__file__).resolve().parent
sys.path.insert(0,str(PHASE/'C001'))
import gzip,json,time
import numpy as np
from threadpoolctl import threadpool_limits
from experiment import read,sha,write,evaluate_v1,generate
from provenance import verify_freeze,check_record
from predictive_bridge import evaluate
from finite_calibration import evaluate as envelope


def main():
    out=PHASE/'C001'; verify_freeze(out); protocol=read(out/'protocol.json')
    receipts=[]; started=time.perf_counter()
    with threadpool_limits(1):
        for case in [0,5,10,14]:
            rp=out/f'raw/case-{case:02}/rep-00000.json.gz'
            with gzip.open(rp,'rt',encoding='utf-8') as stream: old=json.load(stream)
            check_record(old,out,protocol,case,0,metrics=True)
            with np.load(out/old['input_path']) as a:
                z,cal=a['z'],a['calibration']
                regenerated=generate(protocol,case,0)
                for key,value in zip(['z','calibration','truth','shape','kappa'],regenerated):
                    assert np.array_equal(a[key],value)
            pb=evaluate(z,cal,seed=old['algorithm_seed'])
            v1=evaluate_v1(z,cal.transpose(1,0,2),seed=old['algorithm_seed'],acknowledge_scope=True)
            fc=envelope(z,cal,seed=old['algorithm_seed'])
            with np.load(out/old['evidence_path']) as a:
                assert np.array_equal(pb['p'],a['p']) and np.array_equal(pb['reference'],a['reference'])
                for key,value in pb['evidence'].items(): assert np.array_equal(value,a['e_'+key])
                for key in protocol['methods']:
                    value=pb['decisions'][key] if key in pb['decisions'] else v1['discoveries'][key] if key in v1['discoveries'] else fc['decisions'][key]
                    assert np.array_equal(value,a['decision_'+key])
                if case==14:
                    for delta in [2,5]:
                        value=evaluate(z,cal,seed=old['algorithm_seed'],mismatch_bound=delta)
                        assert np.array_equal(value['p'],a[f'p_PB_Delta{delta}'])
                        assert np.array_equal(value['evidence']['PB_grid'],a[f'e_PB_Delta{delta}'])
            receipts.append({'case':case,'rep':0,'input_regenerated_exact':True,'all_declared_decisions_exact':True,'all_primary_e_rank_reference_exact':True,'record_sha256':sha(rp)})
    write(PHASE/'confirmation-replay.json',{'receipts':receipts,'seconds':time.perf_counter()-started,
          'meaning':'deterministic replay, NOT new statistical observations','frozen_sha256':sha(out/'freeze.json')})
    print(receipts,flush=True)


if __name__=='__main__': main()
