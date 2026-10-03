"""Engineering checks on a single already exposed DEV input, no new samples."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('r2_runner_fixture',SCRIPTS/'robustness_r2_development.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def prepare(tmp_path):
    path=SCRIPTS.parent/'artifacts/robustness/R0072-repetitions/case-0000/rep-000000.json.gz'
    old=mod.read(path)
    protocol={'run_id':'RFIXTURE','seed':7205107,'kind':'cached','method_version':'unit_fixture',
              'development_protocol_digest':old['protocol_digest']}
    mod.CONTEXT.clear()
    mod.CONTEXT.update(root=tmp_path,protocol=protocol,digest=mod.content_digest(protocol),
                       old={(0,0):{'path':str(path),'sha256':mod.sha(path)}})


def test_cached_family_exact_parity_and_safe_resume(tmp_path):
    prepare(tmp_path)
    first=mod.one(0,0)
    assert first['status']=='completed'
    record=mod.read(first['path'])
    assert len(record['metrics'])==160 and record['exact_plugin_replay']
    with np.load(record['evidence_path']) as archive:
        assert len(archive.files)==88  #72 new evidence labels +16 p-components
    repeated=mod.one(0,0)
    assert repeated['sha256']==first['sha256'] and repeated['reused']


def test_orphan_arrays_are_preserved_and_failure_is_recorded(tmp_path):
    prepare(tmp_path)
    folder=tmp_path/'artifacts/robustness/RFIXTURE-repetitions/case-0000'
    folder.mkdir(parents=True)
    target=folder/'rep-000000-evidence.npz'
    target.write_bytes(b'Preserve prior incomplete work')
    receipt=mod.one(0,0)
    record=mod.read(receipt['path'])
    assert record['status']=='failed' and 'Orphan evidence' in record['error']
    assert target.read_bytes()==b'Preserve prior incomplete work'


def test_family_scoring_signed_denominator():
    e=np.array([[100.,100.],[0.,0.]])
    truth=np.array([[True,False],[True,False]])
    assert mod.score(e,truth)=={'discoveries':2,'tp':1,'fp':1,'fdp':.5,'power':.5,'power_defined':True}
