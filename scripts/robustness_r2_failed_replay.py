"""Frozen-source engineering replay of retained successful failed-C2 records.

Original preregistered replay selection, no repaired method, no new samples,
no confirmation promotion. The two hard failures remain in the batch.
"""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    base=ROOT/'artifacts/robustness';out=args.out.resolve();archive=base/'R0076-source'
    if not out.is_relative_to(base.resolve()) or out.exists():raise ValueError('New local output required')
    protocol=json.loads((base/'R0076-screen.protocol.json').read_text(encoding='utf-8'))
    audit_path=base/'R0076-failed-audit/summary.json';audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if not audit['audit_complete'] or audit['independent_confirmation_complete']:raise ValueError('Failed descriptive audit required')
    if sha(base/'R0076-results-index.json')!=audit['original_index_sha256']:raise ValueError('Index changed')
    for rel,digest in protocol['source_sha256'].items():
        if sha(archive/rel)!=digest:raise ValueError('Frozen implementation changed')
    sys.path[:0]=[str(archive/'src'),str(archive/'scripts')]
    from sca3_compass.robustness_bootstrap_guard import evaluate as reference
    from sca3_compass.robustness_calibration_efficiency import evaluate as candidate
    from sca3_compass.robustness_io import write_json,content_digest
    from robustness_r1_window import generate,_r0
    from robustness_r2_development import read,score
    setup=protocol['replay_plan'];pairs={(i,r) for i in setup['cases'] for r in setup['reps']}
    pairs.update((r['case'],r['rep']) for r in audit['R1_soft_failed_folds'])
    for i in sorted({r['case'] for r in audit['K_soft_failed_folds']}):
        pairs.add((i,min(r['rep'] for r in audit['K_soft_failed_folds'] if r['case']==i)))
    index=read(base/'R0076-results-index.json');indexed={(r['case_index'],r['rep']):r for r in index['receipts']}
    start=time.perf_counter();receipts=[]
    with threadpool_limits(1):
        for i,rep in sorted(pairs):
            if cooperative_stop():raise InterruptedError('Finite replay cap; no completed replay claim')
            receipt=indexed[(i,rep)];record=read(receipt['path'])
            if sha(receipt['path'])!=receipt['sha256'] or record['protocol_digest']!=content_digest(protocol):raise ValueError('Record mismatch')
            if record['status']!='completed':raise ValueError('Do not replace failed record with replay')
            for key in ['input','evidence']:
                if sha(record[key+'_path'])!=record[key+'_sha256']:raise ValueError('Array changed')
            z,cal,truth=generate(protocol['seed'],i,rep,protocol['cases'][i])
            with np.load(record['input_path'],allow_pickle=False) as data:
                for name,value in [('z',z),('calibration',cal),('truth',truth)]:np.testing.assert_array_equal(value,data[name])
            seed=int(np.random.SeedSequence([protocol['seed'],i,rep,913]).generate_state(1)[0])
            r0,_=_r0(z,cal);r1,d1=reference(z,cal,seed=seed,draws=16)
            r2,d2=candidate(z,cal,seed=seed,draws=16,mode='K')
            with np.load(record['evidence_path'],allow_pickle=False) as old:
                for name,value in {**r0,**r1,**r2}.items():
                    np.testing.assert_array_equal(value,old[name])
                    if score(value,truth)!=record['metrics'][name]:raise ValueError('Score replay mismatch')
                for tag,d in [('R1',d1),('R2',d2)]:
                    for name,value in d['held_pvalues'].items():np.testing.assert_array_equal(value,old['p_'+tag+'_'+name])
            if [f['K_success'] for f in d2['folds']]!=[f['K_success'] for f in record['diagnostics_R2']['folds']]:raise ValueError('Fallback replay mismatch')
            receipts.append({'case':i,'rep':rep,'record_sha256':receipt['sha256'],
                'status':'EXACT_INPUT_124_E_24_P_SCORES_AND_FRESH_K_FALLBACK',
                'original_memoization':record['numerical_reuse']})
            print('Frozen failed-batch replay',i,rep,flush=True)
    write_json(out,{'complete':True,'kind':'ENGINEERING_REPLAY_FAILED_C2_NOT_INDEPENDENT_CONFIRMATION',
        'independent_confirmation_complete':False,'new_samples':0,'post_C2_repair_used':False,
        'audit_sha256':sha(audit_path),'protocol_sha256':sha(base/'R0076-screen.protocol.json'),
        'script_sha256':sha(__file__),'receipts':receipts,'hard_failure_records_preserved':audit['failures'],
        'elapsed_seconds':time.perf_counter()-start,'finished_utc':datetime.now(timezone.utc).isoformat()})
    print(out,flush=True)


if __name__=='__main__':main()
