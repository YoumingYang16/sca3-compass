"""Memory-bounded reference path must retain all integrity checks."""
import gzip
import json
import pytest
from test_robustness_replay import replay,archive,put


def prepare(archive,compact=False):
    project,folder,snapshot,protocol,results=archive
    protocol['compact']=compact
    put(folder/'EXPERIMENT_REGISTRY.json',{'experiments':[{'id':'RTEST',
        'status':'completed','settings':protocol}]})
    for i,result in enumerate(results):
        path=folder/f'RTEST-cases/case-{i:04}.json'
        item={'case_index':i,'settings_digest':replay.content_digest(protocol),
            'result_digest':replay.content_digest(result),'result':result}
        if compact:
            path.parent.mkdir(parents=True,exist_ok=True)
            with gzip.open(str(path)+'.gz','wt',encoding='utf-8') as stream:
                json.dump(item,stream)
        else:
            put(path,item)
    return folder,protocol,results


@pytest.mark.parametrize('compact',[False,True])
def test_complete_reference_does_not_parse_duplicated_aggregate(archive,monkeypatch,compact):
    folder,protocol,results=prepare(archive,compact)
    original=replay.read_json
    def restricted(path):
        assert str(path)!=str(folder/'RTEST-screen.json')
        return original(path)
    monkeypatch.setattr(replay,'read_json',restricted)
    actual=replay.reference_results(folder,protocol)
    assert actual['source_status']=='completed'
    assert 'aggregate_not_parsed' in actual['reference_storage']
    assert actual['cases']=={str(i):replay.raw_case(r) for i,r in enumerate(results)}
    assert len(actual['files'])==2


@pytest.mark.parametrize('bad',['missing','extra','digest','settings','index','case','repetitions','registry_status','registry_settings'])
def test_broken_checkpoint_does_not_fallback_to_good_aggregate(archive,bad):
    folder,protocol,results=prepare(archive)
    path=folder/'RTEST-cases/case-0000.json'
    item=replay.read_json(path)
    if bad=='missing':
        path.unlink()
    elif bad=='extra':
        put(folder/'RTEST-cases/case-0099.json',item)
    elif bad.startswith('registry_'):
        entry={'id':'RTEST','status':'running' if bad=='registry_status' else 'completed',
            'settings':{} if bad=='registry_settings' else protocol}
        put(folder/'EXPERIMENT_REGISTRY.json',{'experiments':[entry]})
    else:
        if bad=='digest':
            item['result_digest']='0'*64
        elif bad=='settings':
            item['settings_digest']='0'*64
        elif bad=='index':
            item['case_index']=1
        elif bad=='case':
            item['result']['case']=results[1]['case']
        else:
            item['result']['repetitions']=2
        put(path,item)
    with pytest.raises(replay.ReplayError):
        replay.reference_results(folder,protocol)
