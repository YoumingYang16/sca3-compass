import json
from pathlib import Path
import pytest
from sca3_compass import robustness_io as io


def test_windows_sharing_retry_preserves_original_and_recovers(tmp_path,monkeypatch):
    path=tmp_path/'result.json'
    io.write_json(path,{'old':1})
    real=io.os.replace
    calls=[]
    def locked_then_open(source,target):
        calls.append(1)
        if len(calls)<3:
            assert json.loads(path.read_text())=={'old':1}
            raise PermissionError('simulated Windows reader lock')
        real(source,target)
    monkeypatch.setattr(io.os,'replace',locked_then_open)
    monkeypatch.setattr(io.time,'sleep',lambda duration:None)
    io.write_json(path,{'new':2})
    assert len(calls)==3 and json.loads(path.read_text())=={'new':2}
    assert not list(tmp_path.glob('*.tmp'))


def test_checkpoint_source_and_payload_integrity(tmp_path):
    settings={'cases':[{'name':'case'}],'repetitions':10,'source_sha256':{'x':'abc'}}
    result={'case':settings['cases'][0],'repetitions':10,'power':[.2,.3]}
    checkpoint={'case_index':0,'settings_digest':io.content_digest(settings),
        'result_digest':io.content_digest(result),'result':result}
    io.write_json(tmp_path/'case-0000.json',checkpoint)
    assert io.load_case_checkpoints(tmp_path,settings)==[(0,result)]
    with pytest.raises(ValueError,match='Checkpoint mismatch'):
        io.load_case_checkpoints(tmp_path,{**settings,'repetitions':20})
    checkpoint['result']['power'][0]=.99
    io.write_json(tmp_path/'case-0000.json',checkpoint)
    with pytest.raises(ValueError,match='Checkpoint mismatch'):
        io.load_case_checkpoints(tmp_path,settings)


def test_compressed_checkpoint_is_lossless_and_compact_diagnostics_keep_failures(tmp_path):
    settings={'cases':[{'name':'case'}],'repetitions':2}
    result={'case':settings['cases'][0],'repetitions':2,
        'rows':[{'fdp_by_repetition':[0,.01],'power_by_repetition':[.1,.9]}],
        'diagnostics':[{'fit':{'converged':False,'iterations':500,'weights':list(range(241))}},
                       {'fit':{'converged':True,'matrix':[[1,0],[0,1]]}}]}
    raw=json.loads(json.dumps(result))
    compact=io.compact_case_result(result,'case-0000.json.gz')
    assert result==raw and compact['rows']==result['rows']
    assert compact['diagnostics'][0]['fit']['converged'] is False
    assert compact['diagnostics'][0]['fit']['weights']['length']==241
    checkpoint={'case_index':0,'settings_digest':io.content_digest(settings),
        'result_digest':io.content_digest(result),'result':result}
    io.write_json_gzip(tmp_path/'case-0000.json.gz',checkpoint)
    assert io.load_case_checkpoints(tmp_path,settings)==[(0,result)]
    transformed=io.load_case_checkpoints(tmp_path,settings,
        transform=lambda value,path:io.compact_case_result(value,path.name))
    assert transformed==[(0,compact)]


def test_audit_tree_compaction_keeps_all_paths_counts_failures_and_metrics():
    from sca3_compass.robustness_analysis import diagnostic_counts
    diagnostics=[{'fit':{'converged':False,'numerical_failure_count':2,'scale':None},
        'many':{'utilities':[{'tp':list(range(16)),'weights':[.5,1.5]} for _ in range(25)]},
        'folds':[{'converged':True,'value':None},{'not_run':True,'converged':False}]},
        {'fit':{'converged':True,'numerical_failure_count':0}}]
    result={'rows':[{'power_by_repetition':[.2,.3]}],'diagnostics':diagnostics}
    copy=json.loads(json.dumps(result))
    view=io.compact_case_result(result,'case.json.gz',audit_only=True)
    assert diagnostic_counts(view['diagnostics'])==diagnostic_counts(diagnostics)
    assert view['rows']==result['rows'] and result==copy
    assert 'many' not in view['diagnostics'][0]
    assert view['diagnostics'][0]['fit']['converged'] is False
    assert view['diagnostics_storage']['raw_result_digest']==io.content_digest(result)


def test_buffered_compression_exact_json_bytes_and_deterministic_gzip(tmp_path):
    import gzip
    import math
    value={'unicode':'分子可靠性🧬','negative_zero':-0.,'large':'x'*1100000,
           'rows':[{str(i):[math.sin(i),None,True,False]} for i in range(200)]}
    paths=[tmp_path/'a.gz',tmp_path/'b.gz']
    for path in paths:io.write_json_gzip(path,value)
    expected=json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
    assert gzip.decompress(paths[0].read_bytes())==expected
    assert paths[0].read_bytes()==paths[1].read_bytes()


def test_buffered_failure_preserves_old_target_and_failed_temp(tmp_path):
    path=tmp_path/'result.gz'
    io.write_json_gzip(path,{'old':1})
    old=path.read_bytes()
    with pytest.raises(ValueError):io.write_json_gzip(path,{'bad':float('nan')})
    assert path.read_bytes()==old
    assert len(list(tmp_path.glob('*.tmp')))==1
