from pathlib import Path
from types import SimpleNamespace
import json
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import molecular_v1_finalize as finalizer


def test_failed_final_cli_cannot_mark_canonical_package_complete(tmp_path,monkeypatch):
    package=tmp_path/'package';(package/'evidence').mkdir(parents=True)
    def put(name,value):
        (package/name).write_text(json.dumps(value),encoding='utf-8')
    put('evidence/confirmation.json',{})
    put('evidence/replay.json',{})
    put('verification.json',{'scientific_pass':True})
    put('actual-command-verification.json',{'passed':True})
    put('release.json',{'packaging_status':'PENDING_ACTUAL_CLI_AND_FINAL_REVIEW'})
    put('actual-command.json',{'command':['python','entry.py','--output','old.json',
        '--validation-receipt',str(package/'release.json'),'--seed','1']})
    review=tmp_path/'review.json'
    review.write_text(json.dumps({'scientific_review_passed':True,
        'confirmation_sha256':finalizer.sha(package/'evidence/confirmation.json'),
        'replay_sha256':finalizer.sha(package/'evidence/replay.json')}),encoding='utf-8')
    old=(package/'release.json').read_bytes()
    monkeypatch.setattr(sys,'argv',['finalize','--package',str(package),'--review',str(review)])
    def fail(command,**kwargs):
        assert Path(command[command.index('--validation-receipt')+1]).name=='release-final-candidate.json'
        assert (package/'release.json').read_bytes()==old
        return SimpleNamespace(returncode=2,stdout='',stderr='controlled CLI failure')
    monkeypatch.setattr(finalizer.subprocess,'run',fail)
    with pytest.raises(RuntimeError,match='Final receipt-bound CLI failed'):finalizer.main()
    assert (package/'release.json').read_bytes()==old
    assert not (package/'MANIFEST.json').exists()
