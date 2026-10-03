"""Seal completed R4 evidence without running inference or changing old files.

The default output is a hash-bound research directory. Optional --zip creates
one local evidence archive including the frozen local dependency closure. This
is not a certification of arbitrary-machine portability or mathematical truth.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, zipfile

PHASE=Path(__file__).resolve().parent
ROOT=PHASE.parent.parent
TRUSTED_C001='0fa6e578932db51caba13a92ba2e474e234392f6e8771af5ecf1e3b4b517a6a5'
ANCHORS={
 'releases/K-NR-1.0.0.zip':'e4bd2e6061bc886796916013e30ae5c05b86c543ab3c4b129f8173afbdba8fe2',
 'research/r2-finite-20260917/DELIVERY_MANIFEST.json':'51ad4505b7d74d90ed9c0164a8d359c3bb6238db6b2eb08ba741d9f7d1b78073',
 'research/r3-efficiency-20260917/DELIVERY_MANIFEST.json':'64e0e5048b7b78434c69fba0c7ed3543a98cd4ffaecd9db0f9087752670661ae',
 'research/r3-novelty-20260918/DELIVERY_MANIFEST.json':'10bf73416b58512ecb6e8a0e5a6c554ac3c928875d57b01ae4520692b410660c',
}

def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def read(p): return json.loads(p.read_text(encoding='utf8'))

def write(p,v):
    with p.open('x',encoding='utf8') as f: json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--zip',action='store_true');args=ap.parse_args()
    manifest=PHASE/'DELIVERY_MANIFEST.json'
    if manifest.exists(): raise FileExistsError('sealed evidence must never be overwritten')
    status=read(PHASE/'FINAL_STATUS.json');audit=read(PHASE/'checks/final-result-audit.json')
    if status['execution_status']!='COMPLETE_SCOPED_R4_RESEARCH': raise ValueError('unfinished research package')
    if sha(PHASE/'C001/freeze.json')!=TRUSTED_C001: raise ValueError('confirmation freeze differs')
    if audit['status']!='PASS' or audit['families']!=5120 or audit['intervals_recomputed']!=133:
        raise ValueError('missing complete result audit')
    for key,p in [('source_sha256',PHASE/'audit_r4.py'),('C001_index_sha256',PHASE/'C001/index.json'),
                  ('C001_summary_sha256',PHASE/'C001/summary.json'),('C001_freeze_sha256',PHASE/'C001/freeze.json')]:
        if audit[key]!=sha(p): raise ValueError('audit binding changed: '+key)
    for rel,digest in ANCHORS.items():
        if sha(ROOT/rel)!=digest: raise ValueError('predecessor anchor changed: '+rel)
    # Explicit dependency closure from the existing freezes, not the working app.
    dependencies={}
    for rel in ['research/r2-finite-20260917/C001','research/r3-efficiency-20260917/C001']:
        fr=read(ROOT/rel/'freeze.json')
        dependencies[rel+'/freeze.json']=sha(ROOT/rel/'freeze.json')
        for name,digest in fr['files'].items(): dependencies[rel+'/'+name]=digest
        for name,digest in fr.get('v1_dependencies',{}).get('files',{}).items(): dependencies[name]=digest
    for rel,digest in ANCHORS.items():
        # Original V1 ZIP is verified but not duplicated inside the new archive.
        if not rel.endswith('.zip'): dependencies[rel]=digest
    for rel,digest in dependencies.items():
        if sha(ROOT/rel)!=digest: raise ValueError('frozen dependency changed: '+rel)
    files={};excluded=[]
    for p in sorted(PHASE.rglob('*')):
        rel=p.relative_to(PHASE);parts=rel.parts
        if any(x in ['.r4-venv','__pycache__','.pytest_cache','delivery'] or x.startswith('test-temp-') for x in parts):
            continue
        if p.is_symlink(): raise ValueError('unexpected link in R4 package: '+str(rel))
        if not p.is_file(): continue
        if p.suffix=='.pyc' or p.name.startswith('.env'): excluded.append(rel.as_posix());continue
        files[rel.as_posix()]={'size':p.stat().st_size,'sha256':sha(p)}
    for name in ['R4_FINAL_REPORT_ZH.md','R4_RESULTS.md','R4_THEORY.md','R4_RELATED_WORK.md',
                 'R4_SPEC_AND_PLAN.md','REVIEW_RECORD.md','generated/GATES.json','generated/RESULT_TABLES.md']:
        if name not in files: raise ValueError('missing delivery material: '+name)
    write(manifest,{'utc':datetime.now(timezone.utc).isoformat(),'version':'R4-TCB-0.2.0',
        'execution_status':status['execution_status'],'scientific_grades':status['scientific_grades'],
        'files':files,'frozen_dependencies':dependencies,'verified_predecessor_anchors':ANCHORS,
        'exclusions':['.r4-venv','pytest fixture scratch','bytecode/cache','delivery artifacts',*excluded],
        'limitations':['Hashes certify current bytes, not historic truth or proof correctness.',
          'Predecessor V1 ZIP is verified locally, not duplicated; old full result archives not included.',
          'Some historical absolute path checks limit cross-machine replay; do not bypass identity checks.',
          'All new observations are labelled simulation; no protected or patient data read.'],
        'manifest_self_hash':'not self-referential; archive receipt binds this manifest'})
    for rel,v in files.items():
        if sha(PHASE/rel)!=v['sha256']: raise ValueError('file changed during seal: '+rel)
    if not args.zip:
        print('SEALED',len(files),'R4 files',len(dependencies),'frozen dependencies');return
    out=PHASE/'delivery';out.mkdir(exist_ok=True)
    target=out/'R4-TCB-0.2.0-research-evidence.zip'
    members={str((PHASE/rel).relative_to(ROOT).as_posix()):PHASE/rel for rel in files}
    members.update({rel:ROOT/rel for rel in dependencies})
    members[manifest.relative_to(ROOT).as_posix()]=manifest
    with zipfile.ZipFile(target,'x',allowZip64=True) as z:
        for rel,p in sorted(members.items()):
            if Path(rel).is_absolute() or '..' in Path(rel).parts: raise ValueError('unsafe archive path')
            z.write(p,rel,compress_type=zipfile.ZIP_STORED if p.suffix in ['.npz','.gz'] else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(target) as z:
        if len(z.namelist())!=len(set(z.namelist())) or set(z.namelist())!=set(members):
            raise ValueError('archive roster')
        if z.testzip() is not None: raise ValueError('archive CRC')
        for rel,p in members.items():
            with z.open(rel) as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
            if digest!=sha(p): raise ValueError('archive member differs: '+rel)
    write(out/'ARCHIVE_RECEIPT.json',{'utc':datetime.now(timezone.utc).isoformat(),
        'path':str(target),'size':target.stat().st_size,'sha256':sha(target),
        'members':len(members),'manifest_sha256':sha(manifest),'checks':'CRC, exact unique member roster, all member SHA256 match local files',
        'not_uploaded':True})
    print('SEALED_ARCHIVE',target,target.stat().st_size,len(members))

if __name__=='__main__': main()
