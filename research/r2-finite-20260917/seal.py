"""Final artifact hash manifest after genuine completion; never edits V1."""
from pathlib import Path
from datetime import datetime,timezone
import sys
PHASE=Path(__file__).resolve().parent
sys.path.insert(0,str(PHASE/'C001'))
from experiment import sha,read,write
from provenance import verify_freeze


def main():
    target=PHASE/'DELIVERY_MANIFEST.json'
    if target.exists(): raise FileExistsError('preserve sealed delivery')
    verify_freeze(PHASE/'C001')
    summary=read(PHASE/'C001/summary.json')
    if summary['total_families']!=7680: raise ValueError('incomplete evidence')
    for name in ['RESEARCH_FINDINGS_ZH.md','CLAIM_EVIDENCE.md','PAPER_CORE.md','README.md',
                 'confirmation-replay.json','saved-decision-audit.json','mechanism-summary.json',
                 'figures/RESULTS.md','actual-invocation.json','FINAL_STATUS.json']:
        if not (PHASE/name).is_file(): raise ValueError('missing deliverable: '+name)
    status=read(PHASE/'FINAL_STATUS.json')
    if status['phase_status']!='COMPLETE_WITH_LIMITATIONS': raise ValueError('not at completed research handoff')
    files={}
    for path in sorted(PHASE.rglob('*')):
        if not path.is_file() or '__pycache__' in path.parts or '.pytest_cache' in path.parts or path==target: continue
        files[path.relative_to(PHASE).as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size}
    write(target,{'created_utc':datetime.now(timezone.utc).isoformat(),
          'scope':'completed R2 research evidence, NOT global algorithm victory or clinical release',
          'scientific_status':status['scientific_status'],'novelty_status':status['novelty_status'],
          'v1_unchanged_zip_sha256':read(PHASE/'C001/freeze.json')['v1_zip_sha256'],
          'files':files,'file_count':len(files),'total_bytes':sum(v['bytes'] for v in files.values())})
    print({'manifest':str(target),'files':len(files),'sha256':sha(target)},flush=True)


if __name__=='__main__': main()
