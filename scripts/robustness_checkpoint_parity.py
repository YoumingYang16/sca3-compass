"""Memory-bounded exact metric parity across registered source-repair reruns.

Verifies protocol/data identity, archived source and checkpoint content. This
checks overlapping repetitions only; neither new science nor confirmation.
"""
import argparse
from pathlib import Path
import re

from robustness_replay import read_json,audit_source,raw_case,content_digest,sha,write_new,run_id,FEATURES


def metrics(path,protocol,index):
    checkpoint=read_json(path)
    result=checkpoint['result']
    if (checkpoint['case_index']!=index or checkpoint['settings_digest']!=content_digest(protocol)
            or checkpoint['result_digest']!=content_digest(result)
            or result['case']!=protocol['cases'][index]):
        raise ValueError('Checkpoint identity/integrity mismatch')
    return raw_case(result)


def substantive_protocol(protocol):
    execution=protocol.get('execution_dispatch')
    if execution not in (None,'bounded_worker_count_backlog_v1'):
        raise ValueError('Unreviewed execution-dispatch difference')
    return {**{k:v for k,v in protocol.items() if k not in ['run_id','source_sha256','execution_dispatch']},
        **{key:protocol.get(key,False) for key in FEATURES}}


def compare(folder,left,right):
    folder=Path(folder)
    protocols=[read_json(folder/f'{run_id(r)}-screen.protocol.json') for r in [left,right]]
    # Historical protocols predate later OPTIONAL false-default CLI features.
    # Normalize only those explicitly listed known booleans; any actual true
    # change, unknown field, seed or case mismatch still fails closed.
    substantive=[substantive_protocol(p) for p in protocols]
    if substantive[0]!=substantive[1]:
        raise ValueError('Parity requires identical seed, cases, counts and method flags')
    audits=[audit_source(folder/f'{r}-source',p)[0] for r,p in zip([left,right],protocols)]
    indexed=[]
    for rid in [left,right]:
        paths={}
        for path in (folder/f'{rid}-cases').glob('case-*.json*'):
            # Atomic writers intentionally retain incomplete *.tmp files.
            # These are never completed evidence and must not be parsed.
            if not re.fullmatch(r'case-\d{4}\.json(?:\.gz)?',path.name):
                continue
            index=int(path.name.split('.')[0].split('-')[-1])
            if index in paths:
                raise ValueError('Duplicate checkpoint index')
            paths[index]=path
        indexed.append(paths)
    overlap=sorted(set(indexed[0])&set(indexed[1]))
    if not overlap:
        raise ValueError('No common completed checkpoints')
    report=[]
    comparisons=0
    for index in overlap:
        # Reduce the first large diagnostic payload BEFORE loading the second.
        a=metrics(indexed[0][index],protocols[0],index)
        b=metrics(indexed[1][index],protocols[1],index)
        if set(a['rows'])!=set(b['rows']):
            raise ValueError('Method set mismatch: no favorable subset comparison')
        for key in ['case','repetitions','power_defined','replicated_signed_truths']:
            if a[key]!=b[key]:
                raise ValueError('Metric metadata mismatch')
        mismatch=[]
        for method in a['rows']:
            for metric in ['fdp_by_repetition','power_by_repetition']:
                comparisons+=1
                if a['rows'][method][metric]!=b['rows'][method][metric]:
                    mismatch.append({'method':method,'metric':metric})
        report.append({'case_index':index,'methods':len(a['rows']),'mismatches':mismatch,
            'checkpoint_sha256':[sha(paths[index]) for paths in indexed]})
        print(f'Parity verified case {index}: mismatches={len(mismatch)}',flush=True)
    return {'phase':'EXACT_REPAIR_REGRESSION_NOT_CONFIRMATION','runs':[left,right],
        'execution_dispatches':[p.get('execution_dispatch','legacy_submit_all') for p in protocols],
        'source_audits':audits,'compared_case_indices':overlap,
        'unpaired_case_indices':[sorted(set(paths)-set(overlap)) for paths in indexed],
        'repetitions':protocols[0]['repetitions'],'arrays_compared':comparisons,
        'exact_match':all(not row['mismatches'] for row in report),'cases':report}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('left');parser.add_argument('right')
    parser.add_argument('--folder',default='artifacts/robustness');parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError('Preserve parity evidence')
    result=compare(args.folder,args.left,args.right)
    write_new(args.output,result)
    if not result['exact_match']:
        raise SystemExit('Metrics differ; see preserved report')


if __name__=='__main__':
    main()
