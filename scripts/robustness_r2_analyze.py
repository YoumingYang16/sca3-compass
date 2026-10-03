"""Complete targeted DEV analysis, with raw rescoring; no confirmation CIs."""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import shutil
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sca3_compass.robustness_io import content_digest,write_json


def read(path):
    with (gzip.open(path,'rt',encoding='utf-8') if str(path).endswith('.gz') else Path(path).open(encoding='utf-8')) as stream:
        return json.load(stream)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def rescore(record,names):
    with np.load(record['input_path']) as archive:
        truth=archive['truth'].ravel().astype(bool)
    vectors={}
    paths=[record['evidence_path']]
    if 'reference_evidence' in record:
        paths.append(record['reference_evidence'])
    for path in paths:
        with np.load(path) as archive:
            for name in names:
                if name in archive.files:
                    if name in vectors:
                        raise ValueError('Unexpected duplicate evidence key')
                    vectors[name]=archive[name].ravel()
    if set(vectors)!=set(names):
        raise ValueError('Missing raw method evidence')
    e=np.array([vectors[m] for m in names])
    if e.shape!=(len(names),len(truth)) or not np.isfinite(e).all() or np.any(e<0):
        raise ValueError('Invalid raw evidence')
    ranks=np.arange(1,len(truth)+1)
    ordered=np.sort(e,axis=1)[:,::-1]
    k=np.where(ordered>=len(truth)/(.05*ranks),ranks,0).max(1)
    cutoff=np.where(k>0,len(truth)/(.05*np.maximum(k,1)),np.inf)
    selected=e>=cutoff[:,None]
    total,fp,tp=selected.sum(1),(selected&~truth).sum(1),(selected&truth).sum(1)
    for j,name in enumerate(names):
        expected={'discoveries':int(total[j]),'tp':int(tp[j]),'fp':int(fp[j]),
                  'power':float(tp[j]/max(1,truth.sum())),'fdp':float(fp[j]/max(1,total[j])),
                  'power_defined':bool(truth.sum())}
        if expected!=record['metrics'][name]:
            raise ValueError('Independent scoring disagreement:'+name)
    if int(truth.sum())!=record['n_true_signed']:
        raise ValueError('Truth-count mismatch')


def paired_summary(arrays):
    """Descriptive equal-case paired mean and MCSE; never a confidence claim."""
    ncase=len(arrays)
    return {'mean_difference':float(np.mean([x.mean() for x in arrays])),
            'paired_mcse':float(np.sqrt(sum(x.var(ddof=1)/len(x) for x in arrays))/ncase)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',choices=['R0074','R0075'],required=True)
    args=parser.parse_args()
    base=ROOT/'artifacts/robustness'
    index_path=base/f'{args.run_id}-results-index.json'
    index=read(index_path)
    protocol=read(base/f'{args.run_id}-screen.protocol.json')
    if protocol!=index['settings'] or protocol['phase']!='development':
        raise ValueError('Matching targeted development protocol required')
    for rel,digest in protocol['source_sha256'].items():
        if sha(Path(protocol['source_snapshot'])/rel)!=digest:
            raise ValueError('Frozen method source changed')
    target=base/f'{args.run_id}-analysis'
    target.mkdir(exist_ok=False)
    shutil.copy2(__file__,target/'generate_source.py')
    expected=len(protocol['cases'])*protocol['repetitions']
    result={'run_id':args.run_id,'complete':False,'phase':'DEVELOPMENT_NOT_CONFIRMATION',
            'index_sha256':sha(index_path),'script_sha256':sha(__file__),
            'protocol_digest':content_digest(protocol),'provenance':'SIMULATION_NOT_PATIENT_DATA',
            'not_claimed':['independent confirmation','general validity','posthoc calibrated FDR','novelty'],
            'failures':[],'rows':[],'comparisons':{},'total_recorded_cpu_seconds':0.,
            'cost_note':'Cached R0074 includes cache parsing, output generation and reference parity, not fresh fitting or deployment cost'}
    if not index['complete'] or len(index['receipts'])!=expected:
        result['failures']=[r for r in index['receipts'] if r['status']!='completed']
        result['status']='INCOMPLETE_NO_POWER_AGGREGATION'
        write_json(target/'summary.json',result)
        raise SystemExit(2)
    grouped={i:[] for i in range(len(protocol['cases']))}
    for receipt in index['receipts']:
        grouped[receipt['case_index']].append(receipt)
    matrices={}
    names=None
    start=time.perf_counter()
    for case,receipts in grouped.items():
        if cooperative_stop():
            write_json(target/'partial.json',result)
            raise InterruptedError('Finite analysis checkpoint')
        if len(receipts)!=protocol['repetitions'] or {r['rep'] for r in receipts}!=set(range(len(receipts))):
            raise ValueError('Incomplete/duplicated fixed sample')
        records=[]
        kfail=pfail=0
        for receipt in sorted(receipts,key=lambda r:r['rep']):
            if sha(receipt['path'])!=receipt['sha256']:
                raise ValueError('Record checksum mismatch')
            record=read(receipt['path'])
            if record['status']!='completed' or record['protocol_digest']!=result['protocol_digest'] or record['case_index']!=case or record['rep']!=len(records):
                raise ValueError('Record identity mismatch')
            for key in ['input','evidence']:
                if sha(record[key+'_path'])!=record[key+'_sha256']:
                    raise ValueError('Array checksum mismatch')
            if 'reference_evidence' in record and sha(record['reference_evidence'])!=record['reference_evidence_sha256']:
                raise ValueError('Original reference evidence changed')
            if names is None:
                names=list(record['metrics'])
            if set(names)!=set(record['metrics']):
                raise ValueError('Different method membership')
            rescore(record,names)
            records.append((record['metrics'],record['n_true_signed']))
            kfail+=sum(not f['K_success'] for f in record['diagnostics_R2']['folds'])
            pfail+=sum(f['pattern']['fallback'] for f in record['diagnostics_R2']['folds'])
            result['total_recorded_cpu_seconds']+=record['cpu_seconds']
        power=np.array([[r[0][m]['power'] for m in names] for r in records])
        fdp=np.array([[r[0][m]['fdp'] for m in names] for r in records])
        defined={bool(r[1]) for r in records}
        if len(defined)!=1:
            raise ValueError('Variable undefined-power status requires explicit estimand')
        row={'case_index':case,'case':protocol['cases'][case],'n':len(records),'K_failed_folds':kfail,
             'pattern_fallback_folds':pfail,'methods':{}}
        for j,name in enumerate(names):
            row['methods'][name]={'power':float(power[:,j].mean()) if True in defined else None,
                'power_mcse':float(power[:,j].std(ddof=1)/np.sqrt(len(records))) if True in defined else None,
                'fdp':float(fdp[:,j].mean()),'fdp_mcse':float(fdp[:,j].std(ddof=1)/np.sqrt(len(records)))}
        result['rows'].append(row)
        matrices[case]=power
        print('Verified R2 DEV case',case,len(records),flush=True)
    if protocol['kind']=='cached':
        idx={m:j for j,m in enumerate(names)}
        families={}
        original=read(base/'R0073-screen.protocol.json')['analysis_plan']
        families['H']=original['families']['H']
        for tag in ['P','K']:
            for label in ['F_safe','F_empirical']:
                families[f'{tag}/{label}']=[m.replace('R1B_guard_',f'R2{tag}_') for m in original['families'][label]]
        families['ALL_same_information_simple']=sum([families[t+'/'+f] for t in ['P','K'] for f in ['F_safe','F_empirical']],[])
        for label,cases in original['strata'].items():
            for tag in ['P','K']:
                candidate=f'R2{tag}_pilotc0.5_projection_gate_eBH'
                comparison={}
                for family,baselines in families.items():
                    differences=[matrices[i][:,idx[candidate],None]-matrices[i][:,[idx[m] for m in baselines]] for i in cases]
                    comparison[family]={'sample_envelope_difference':float(np.mean([x.mean(0).min() for x in differences])),
                                        'status':'DESCRIPTIVE_DEV_ENVELOPE_NO_CONFIDENCE_INTERVAL'}
                comparison['I_vs_R1']=paired_summary([matrices[i][:,idx[candidate]]-matrices[i][:,idx['R1B_guard_pilotc0.5_projection_gate_eBH']] for i in cases])
                result['comparisons'][label+'/'+tag]=comparison
        result['C1_validity_scope_indices']=original['validity_scope_indices']
    result.update(complete=True,method_labels=names,whole_family_repetitions=expected,analysis_elapsed_seconds=time.perf_counter()-start)
    write_json(target/'summary.json',result)
    print(json.dumps(result['comparisons'],indent=2))
    print(target)


if __name__=='__main__':
    main()
