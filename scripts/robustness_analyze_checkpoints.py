"""Memory-bounded DEVELOPMENT analysis of COMPLETE checksummed raw checkpoints.

All requested metrics are retained. Diagnostic traversal results are preserved
exactly, verified case by case; irrelevant long parameter histories are omitted
only from the in-memory analysis view, never from historical evidence.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import robustness_replay as replay
from sca3_compass.robustness_analysis import analyze_artifact,diagnostic_counts,NUMERICAL_KEYS
from sca3_compass.robustness_io import content_digest

_DROP=object()


def diagnostic_view(value):
    if isinstance(value,dict):
        result={}
        for key,item in value.items():
            if key in NUMERICAL_KEYS or key in ('converged','not_run'):
                result[key]=item
            else:
                child=diagnostic_view(item)
                if child is not _DROP:
                    result[key]=child
        return result if result else _DROP
    if isinstance(value,list):
        result=[child for item in value if (child:=diagnostic_view(item)) is not _DROP]
        return result if result else _DROP
    if value is None or isinstance(value,float) and not math.isfinite(value):
        return value
    return _DROP


def load_view(folder,run_id,methods):
    folder=Path(folder)
    protocol_path=folder/f'{run_id}-screen.protocol.json'
    protocol=replay.read_json(protocol_path)
    registry=replay.read_json(folder/'EXPERIMENT_REGISTRY.json')
    entries=[entry for entry in registry['experiments'] if entry['id']==run_id]
    if len(entries)!=1 or entries[0]['status']!='completed':
        raise ValueError('Only a registered complete run may be analyzed')
    entry=entries[0]
    if entry['settings']!=protocol:
        raise ValueError('Registry and protocol differ')
    audit,_=replay.audit_source(folder/f'{run_id}-source',protocol)
    expected_digest=content_digest(protocol)
    scenarios=[]
    receipt=[]
    for index,case in enumerate(protocol['cases']):
        paths=[p for p in (folder/f'{run_id}-cases').glob(f'case-{index:04}.json*') if p.suffix in ('.json','.gz')]
        if len(paths)!=1:
            raise ValueError('Each expected case requires one immutable checkpoint')
        path=paths[0]
        checkpoint=replay.read_json(path)
        if checkpoint['case_index']!=index or checkpoint['settings_digest']!=expected_digest:
            raise ValueError('Checkpoint index/protocol mismatch')
        result=checkpoint['result']
        if content_digest(result)!=checkpoint['result_digest'] or result['case']!=case:
            raise ValueError('Checkpoint content/case mismatch')
        view=[{} if (v:=diagnostic_view(d)) is _DROP else v for d in result['diagnostics']]
        if diagnostic_counts(view)!=diagnostic_counts(result['diagnostics']):
            raise ValueError('Diagnostic projection changed traversal counts')
        rows=[row for row in result['rows'] if row['method'] in methods]
        if {row['method'] for row in rows}!=set(methods):
            raise ValueError('Missing requested method')
        scenarios.append({**{key:result[key] for key in ('case','repetitions','power_defined',
            'replicated_signed_truths','elapsed_seconds')},'rows':rows,'diagnostics':view})
        receipt.append({'path':str(path.resolve()),'file_sha256':replay.sha(path),
            'result_digest':checkpoint['result_digest'],'diagnostic_projection_exact':True})
        print(f'{run_id}: verified case {index+1}/{len(protocol["cases"])}',flush=True)
    return {'settings':protocol,'elapsed_seconds':entry['elapsed_seconds'],'scenarios':scenarios}, {
        'source_audit':audit,'checkpoint_receipts':receipt,'source_registry_status':'completed',
        'scope':'selected methods; all diagnostic traversal counts preserved; DEVELOPMENT ONLY'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--folder',default='artifacts/robustness')
    parser.add_argument('--candidate',action='append',required=True)
    parser.add_argument('--baseline',action='append',required=True)
    parser.add_argument('--interval',choices=['betting','empirical_bernstein'],default='betting')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('Preserve existing analysis')
    artifact,receipt=load_view(args.folder,replay.run_id(args.run_id),args.candidate+args.baseline)
    result=analyze_artifact(artifact,candidates=args.candidate,baselines=args.baseline,interval=args.interval)
    result['checkpoint_analysis_receipt']=receipt
    result['analysis_sources']={str(Path(__file__).resolve()):replay.sha(__file__),
        **{str(p.resolve()):replay.sha(p) for p in (Path(__file__).resolve().parents[1]/'src/sca3_compass').glob('robustness_*.py')}}
    replay.write_new(output,result)
    print(json.dumps({'output':str(output),'status':'DEVELOPMENT_ONLY'},ensure_ascii=False))


if __name__=='__main__':
    main()
