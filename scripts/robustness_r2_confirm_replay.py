"""Frozen, complete-only C2 exact replay including fresh non-memoized K.

Replay is deterministic verification of old samples, NOT independent evidence.
"""
from research_window import wait_start_gate,cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.robustness_io import content_digest,write_json
from sca3_compass.robustness_calibration_efficiency import evaluate as candidate
from sca3_compass.robustness_bootstrap_guard import evaluate as reference
from robustness_r1_window import generate,_r0
from robustness_r2_development import read,sha,score


def selection(protocol,result):
    setup=protocol['replay_plan']
    pairs={(i,r) for i in setup['cases'] for r in setup['reps']}
    pairs.update((r['case'],r['rep']) for r in result['R1_failed_folds'])
    for i in sorted({r['case'] for r in result['K_failed_folds']}):
        first=min(r['rep'] for r in result['K_failed_folds'] if r['case']==i)
        pairs.add((i,first))
    return sorted(pairs)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=ROOT);args=p.parse_args()
    root=args.project_root.resolve();base=root/'artifacts/robustness'
    protocol=read(base/'R0076-screen.protocol.json');result=read(base/'R0076-confirmation-analysis.json')
    index=read(base/'R0076-results-index.json')
    if not result['complete'] or not index['complete'] or sha(base/'R0076-results-index.json')!=result['index_sha256']:
        raise ValueError('Completed verified C2 only')
    for rel,digest in protocol['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen replay/source changed:'+rel)
    out=base/'R0076-replay.json'
    if out.exists():raise FileExistsError('Preserve existing replay')
    indexed={(r['case_index'],r['rep']):r for r in index['receipts']}
    start=time.perf_counter();receipts=[]
    with threadpool_limits(1):
        for i,rep in selection(protocol,result):
            if cooperative_stop():raise InterruptedError('Finite replay stop; no completed replay claim')
            receipt=indexed[(i,rep)];record=read(receipt['path'])
            if sha(receipt['path'])!=receipt['sha256'] or record['protocol_digest']!=content_digest(protocol):
                raise ValueError('Record identity/hash mismatch')
            for key in ['input','evidence']:
                if sha(record[key+'_path'])!=record[key+'_sha256']:raise ValueError('Array checksum mismatch')
            z,cal,truth=generate(protocol['seed'],i,rep,protocol['cases'][i])
            with np.load(record['input_path']) as old:
                for name,value in [('z',z),('calibration',cal),('truth',truth)]:np.testing.assert_array_equal(value,old[name])
            seed=int(np.random.SeedSequence([protocol['seed'],i,rep,913]).generate_state(1)[0])
            r0,_=_r0(z,cal);r1,d1=reference(z,cal,seed=seed,draws=16)
            r2,d2=candidate(z,cal,seed=seed,draws=16,mode='K') # no cached folds
            with np.load(record['evidence_path']) as old:
                for name,value in {**r0,**r1,**r2}.items():
                    np.testing.assert_array_equal(value,old[name])
                    if score(value,truth)!=record['metrics'][name]:raise ValueError('Replayed metric mismatch')
                for tag,d in [('R1',d1),('R2',d2)]:
                    for name,value in d['held_pvalues'].items():np.testing.assert_array_equal(value,old['p_'+tag+'_'+name])
            if [f['K_success'] for f in d2['folds']]!=[f['K_success'] for f in record['diagnostics_R2']['folds']]:
                raise ValueError('Fresh fallback differs from saved result')
            receipts.append({'case_index':i,'rep':rep,'status':'EXACT_INPUT_124_E_24_P_AND_SCORES_FRESH_K',
                'record_sha256':receipt['sha256'],'same_family_memoized':record['numerical_reuse'],
                'fresh_K_failed_folds':sum(not f['K_success'] for f in d2['folds'])})
            print('Exact fresh C2 replay',i,rep,flush=True)
    write_json(out,{'run_id':'R0076','complete':True,'kind':'REPRODUCTION_NOT_NEW_SAMPLES',
        'analysis_sha256':sha(base/'R0076-confirmation-analysis.json'),'protocol_sha256':sha(base/'R0076-screen.protocol.json'),
        'script_sha256':sha(__file__),'receipts':receipts,'elapsed_seconds':time.perf_counter()-start,
        'finished_utc':datetime.now(timezone.utc).isoformat()})
    print(out,flush=True)


if __name__=='__main__':main()
