"""Atomic research checkpoints with bounded Windows reader-lock retry.

The historical molecular_data writer is frozen and intentionally unchanged.
Failed temporary files are retained for recovery rather than silently deleted.
"""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import gzip
import math
from io import BufferedWriter


def content_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                    allow_nan=False,separators=(',',':')).encode('utf-8')).hexdigest()


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    for attempt in range(12):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==11:
                raise
            time.sleep(min(.02*2**attempt,.5))


def write_json_gzip(path,value):
    """Atomic compressed raw evidence; compression changes no JSON values."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=3) as compressed:
            # Streaming encoder avoids an extra giant serialized copy.
            encoder=json.JSONEncoder(ensure_ascii=False,allow_nan=False,separators=(',',':'))
            # JSON iterencode often emits one tiny token per float/key. Without
            # buffering this calls the compressor millions of times per case,
            # slowing archiving enough to accumulate process-pool results.
            # Bounded binary buffering changes neither JSON bytes nor values.
            with BufferedWriter(compressed,buffer_size=1024*1024) as buffered:
                for chunk in encoder.iterencode(value):
                    buffered.write(chunk.encode('utf-8'))
    for attempt in range(12):
        try:
            os.replace(temporary,path)
            return
        except PermissionError:
            if attempt==11:
                raise
            time.sleep(min(.02*2**attempt,.5))


def compact_case_result(result,checkpoint,*,audit_only=False):
    """Keep every raw metric; long diagnostics remain in checksummed evidence.

    Scalar convergence/fit summaries and short matrices remain directly
    visible. This is storage compaction, not filtering failed fits or seeds.
    """
    reference=str(checkpoint)
    if audit_only:
        from .robustness_analysis import diagnostic_counts,NUMERICAL_KEYS
        omitted=object()
        def audit_view(value):
            if isinstance(value,dict):
                answer={}
                for key,item in value.items():
                    if key in NUMERICAL_KEYS or key in ('converged','not_run'):
                        answer[key]=item
                    elif (child:=audit_view(item)) is not omitted:
                        answer[key]=child
                return answer if answer else omitted
            if isinstance(value,list):
                answer=[child for item in value if (child:=audit_view(item)) is not omitted]
                return answer if answer else omitted
            if value is None or isinstance(value,float) and not math.isfinite(value):
                return value
            return omitted
        diagnostics=[{} if (v:=audit_view(diag)) is omitted else v for diag in result['diagnostics']]
        if diagnostic_counts(diagnostics)!=diagnostic_counts(result['diagnostics']):
            raise ValueError('Diagnostic compaction must preserve ALL event/path counts')
        return {**result,'diagnostics':diagnostics,
            'diagnostics_storage':{'mode':'audit-tree-and-raw-reference-lossless',
                'raw_checkpoint':reference,'raw_result_digest':content_digest(result),
                'diagnostic_event_counts_preserved_exactly':True,
                'notice':'ALL detailed fits, utilities, parameters and histories are in the checksummed compressed checkpoint'}}
    def reduce(value):
        if isinstance(value,dict):
            return {key:reduce(item) for key,item in value.items()}
        if isinstance(value,list):
            if len(value)>32:
                return {'stored_in_raw_checkpoint':reference,'length':len(value),'sha256':content_digest(value)}
            return [reduce(item) for item in value]
        return value
    return {**result,'diagnostics':[reduce(diag) for diag in result['diagnostics']],
            'diagnostics_storage':{'mode':'long-arrays-external-lossless','raw_checkpoint':reference,
                'raw_result_digest':content_digest(result)}}


def load_case_checkpoints(directory,settings,transform=None):
    results=[]
    expected=content_digest(settings)
    paths=sorted([*Path(directory).glob('case-*.json'),*Path(directory).glob('case-*.json.gz')])
    for path in paths:
        if path.suffix=='.gz':
            with gzip.open(path,'rt',encoding='utf-8') as stream:
                item=json.load(stream)
        else:
            item=json.loads(path.read_text(encoding='utf-8'))
        index=item['case_index']
        if (item['settings_digest']!=expected or item['result_digest']!=content_digest(item['result'])
                or not isinstance(index,int) or not 0<=index<len(settings['cases'])
                or item['result']['case']!=settings['cases'][index]
                or item['result']['repetitions']!=settings['repetitions']):
            raise ValueError(f'Checkpoint mismatch: {path}')
        results.append((index,transform(item['result'],path) if transform else item['result']))
    if len({i for i,_ in results})!=len(results):
        raise ValueError('Duplicate checkpoint indices')
    return results
