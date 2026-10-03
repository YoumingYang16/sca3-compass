"""TEST_FIXTURE-only IO checks; never enter scientific experiment registry."""
import gzip,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def test_streaming_analysis_computes_expected_functional(tmp_path,monkeypatch):
    script=ROOT/'scripts/robustness_r1_confirm.py'
    spec=importlib.util.spec_from_file_location('fixture_confirm',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # This test isolates streaming/scoring; strict54-case membership is tested separately.
    monkeypatch.setattr(module,'validate_plan',lambda *args:None)
    folder=tmp_path/'artifacts/robustness';folder.mkdir(parents=True)
    names=['candidate','base1','base2'];receipts=[]
    values=[[[.9,.7,.8],[.7,.5,.4]],[[.8,.7,.7],[.6,.5,.5]]]
    for case in range(2):
        for rep in range(2):
            artifact=folder/f'fixture-{case}-{rep}.npz'
            arrays={name:np.r_[np.full(round(values[case][rep][j]*10),1000.),np.zeros(10-round(values[case][rep][j]*10))] for j,name in enumerate(names)}
            np.savez(artifact,truth=np.ones(10,bool),**arrays)
            record={'status':'completed','case_index':case,'rep':rep,'n_true_signed':10,'provenance':'TEST_FIXTURE',
                'input_path':str(artifact),'input_sha256':sha(artifact),'evidence_path':str(artifact),'evidence_sha256':sha(artifact),
                'diagnostics_R1':{'folds':[{'fallback':False}]},
                'metrics':{name:{'power':values[case][rep][j],'fdp':0.,'tp':round(values[case][rep][j]*10),'fp':0,'discoveries':round(values[case][rep][j]*10)} for j,name in enumerate(names)}}
            path=folder/f'fixture-{case}-{rep}.json.gz'
            with gzip.open(path,'wt',encoding='utf-8') as s:json.dump(record,s)
            receipts.append({'status':'completed','case_index':case,'rep':rep,'path':str(path),'sha256':sha(path)})
    plan={'reported_methods':names,'candidate':'candidate','reporting_error_budget':.025,
        'error_allocation':{'envelopes':.012,'FDR_candidate':.006,'FDR_other':.004,'contributions':.003},
        'strata':{'core54':[0,1]},'families':{'F':['base1','base2']},'development_selected_baselines':{'F':{'0':'base1','1':'base1'}},
        'contribution_pairs':{'difference':['candidate','base1']},'validity_scope_indices':[0,1]}
    dev=folder/'DEVFIXTURE-summary.json';dev.write_text('TEST_FIXTURE_ONLY')
    plan.update(development_run='DEVFIXTURE',development_summary_sha256=sha(dev))
    protocol={'phase':'C1','seed':42,'method_version':'TEST_FIXTURE_ONLY','cases':[{'name':'fixture0'},{'name':'fixture1'}],'repetition_counts':[2,2],
        'analysis_plan':plan,'source_sha256':{'scripts/robustness_r1_confirm.py':sha(script)}}
    for receipt in receipts:
        path=Path(receipt['path'])
        with gzip.open(path,'rt') as s:record=json.load(s)
        record.update(run_id='RTEST',phase='C1',method_version='TEST_FIXTURE_ONLY',protocol_digest=module.content_digest(protocol),seed_sequence=[42,receipt['case_index'],receipt['rep']])
        with gzip.open(path,'wt') as s:json.dump(record,s)
        receipt['sha256']=sha(path)
    (folder/'RTEST-screen.protocol.json').write_text(json.dumps(protocol))
    (folder/'RTEST-results-index.json').write_text(json.dumps({'complete':True,'settings':protocol,'receipts':receipts}))
    monkeypatch.setattr(sys,'argv',['fixture','--run-id','RTEST','--project-root',str(tmp_path)])
    module.main()
    actual=json.loads((folder/'RTEST-confirmation-analysis.json').read_text())
    assert abs(actual['comparisons']['core54/F']['sample_envelope_difference']-.15)<1e-12
    assert actual['whole_family_repetitions']==4
    assert actual['rows'][0]['methods']['candidate']['power']==.8
    assert actual['input_evidence_bytes_verified']==8*artifact.stat().st_size
