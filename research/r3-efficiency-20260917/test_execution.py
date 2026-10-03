"""No confirmatory data: execution identity and protocol arithmetic tests."""
import gzip,json
from pathlib import Path
import pytest
import numpy as np
import r3_experiment
from r3_experiment import checkpoint_row,seed_for,one,check_input_schema,check_evidence_schema
from r3_common import sha,write,read,R2

def test_checkpoint_identity(tmp_path):
    write(tmp_path/'freeze.json',{'test_fixture':True})
    assert checkpoint_row(tmp_path,0,0) is None
    path=tmp_path/'raw/case-00/rep-00000.json.gz';path.parent.mkdir(parents=True)
    row={'case':0,'rep':0,'freeze_sha256':sha(tmp_path/'freeze.json'),'status':'hard_failure'}
    with gzip.open(path,'xt',encoding='utf8') as f:json.dump(row,f)
    assert checkpoint_row(tmp_path,0,0)['status']=='hard_failure'
    with pytest.raises(ValueError):_bad_row(tmp_path,path,row)

def _bad_row(root,path,row):
    # Controlled pytest temporary fixture only, never any research record.
    row['rep']=2
    with gzip.open(path,'wt',encoding='utf8') as f:json.dump(row,f)
    checkpoint_row(root,0,0)

def test_orphan_rejected(tmp_path):
    write(tmp_path/'freeze.json',{})
    p=tmp_path/'raw/case-00/rep-00000-input.npz';p.parent.mkdir(parents=True);p.touch()
    with pytest.raises(ValueError,match='orphaned'):checkpoint_row(tmp_path,0,0)

def test_seed_injective_and_interval_roster():
    p=json.loads((Path(__file__).parent/'protocol.json').read_text()) if (Path(__file__).parent/'protocol.json').exists() else json.loads((Path(__file__).parent/'protocol-confirm.json').read_text())
    assert len({seed_for(p,c,r) for c in range(15) for r in range(1024)})==15360
    # FDP all15, Power12non-null;5pairedPower;8Delta means;4boundary;1core.
    assert 8*15+8*12+5*12+8+4+1==p['planned_two_sided_intervals']<=p['interval_cap']

@pytest.mark.parametrize('case',[0,14])
def test_new_mode_wiring_on_reused_old_inputs(tmp_path,monkeypatch,case):
    """No new observations or Power screening: normal/drift executor coverage."""
    source=R2/f'C001/raw/case-{case:02}/rep-00000-input.npz'
    with np.load(source,allow_pickle=False) as f:
        z,cal,truth,h,k=(f[x] for x in ['z','calibration','truth','shape','kappa'])
    monkeypatch.setattr(r3_experiment,'generate',lambda p,c,r:(z,cal,truth,h,k))
    base=Path(__file__).parent
    p=read(base/('protocol.json' if (base/'protocol.json').exists() else 'protocol-confirm.json'))
    p['stage']='UNIT_WIRING_REUSED_R2_INPUT';write(tmp_path/'protocol.json',p)
    write(tmp_path/'freeze.json',{'fixture':True})
    item=one((str(tmp_path),case,0))
    with gzip.open(tmp_path/item['path'],'rt',encoding='utf8') as f:row=json.load(f)
    assert row['status']=='completed',row.get('traceback')
    expected=set(p['methods'])|({'R3_Delta5','R2_Delta5'} if case==14 else set())
    assert set(row['metrics'])==expected
    assert row['old_times']['r2_status']=='completed'
    assert checkpoint_row(tmp_path,case,0)['sha256']==item['sha256']
    with np.load(tmp_path/row['input_path'],allow_pickle=False) as a:check_input_schema(a,p,case)
    with np.load(tmp_path/row['evidence_path'],allow_pickle=False) as a:
        check_evidence_schema(a,row,p)
        class Missing:
            files=[k for k in a.files if k!='e_R3_main']
            def __getitem__(self,k):return a[k]
        with pytest.raises(ValueError,match='schema keys'):check_evidence_schema(Missing(),row,p)
