"""Seal this bounded R3 delivery only after actual analysis/replays/review.

Never labels a scoped result as full research success merely because files exist.
No old release writes, archive deletion or new statistical experiment.
"""
from pathlib import Path
from datetime import datetime,timezone
import sys
sys.dont_write_bytecode=True
BASE=Path(__file__).resolve().parent;OUT=BASE/'C001'
sys.path.insert(0,str(OUT))
from r3_common import read,write,sha,R2,ROOT,verify_r2
from r3_experiment import validate

REQUIRED=['EFFICIENCY_LOSS_LEDGER.md','EFFICIENCY_LITERATURE_MATRIX.md','R3_THEORY.md',
          'R3_ALGORITHM.md','R3_RESULTS.md','VALIDITY_EFFICIENCY_FRONTIER.md',
          'R3_CLAIM_EVIDENCE.md','R3_PAPER_CORE.md','RESEARCH_FINDINGS_ZH.md',
          'REVIEW_FINAL.md','FINAL_STATUS.json','REPRODUCTION_AUDIT.json','EXAMPLE_RECEIPT.json',
          'RECOMPUTED_SUMMARY.json','IMPLEMENTATION_DIAGNOSTICS.json',
          'frozen-tests.xml','figures/manifest.json']

def main():
    validate(OUT);verify_r2()
    frozen_identity=read(OUT/'freeze.json')
    for field,path in [('r2_manifest_sha256',R2/'DELIVERY_MANIFEST.json'),
                       ('r2_index_sha256',R2/'C001/index.json'),
                       ('r2_freeze_sha256',R2/'C001/freeze.json'),
                       ('r2_protocol_sha256',R2/'C001/protocol.json')]:
        if sha(path)!=frozen_identity[field]:raise ValueError('frozen R2 handoff identity changed:'+field)
    if sha(ROOT/'releases/K-NR-1.0.0.zip')!=read(R2/'C001/freeze.json')['v1_dependencies']['zip_sha256']:
        raise ValueError('V1 archive identity changed')
    for name in REQUIRED:
        if not (BASE/name).is_file() or (BASE/name).stat().st_size==0:raise ValueError('missing deliverable:'+name)
    status=read(BASE/'FINAL_STATUS.json');summary=read(OUT/'summary.json');p=read(OUT/'protocol.json')
    if not status.get('bounded_phase_complete'):raise ValueError('phase still incomplete')
    if not status.get('internal_final_review_integrated'):raise ValueError('review not integrated')
    if summary['families']!=15360 or summary['two_sided_intervals']!=289:raise ValueError('confirmation scale mismatch')
    if sha(BASE/'RECOMPUTED_SUMMARY.json')!=sha(OUT/'summary.json'):raise ValueError('independent re-analysis not identical')
    implementation=read(BASE/'IMPLEMENTATION_DIAGNOSTICS.json')
    if implementation['index_sha256']!=sha(OUT/'index.json') or implementation['freeze_sha256']!=sha(OUT/'freeze.json'):
        raise ValueError('implementation diagnostics identity changed')
    if set(implementation['rows'])!={str(c) for c in range(15)} or any(v['families']!=1024 for v in implementation['rows'].values()):
        raise ValueError('implementation diagnostics incomplete')
    replay=read(BASE/'REPRODUCTION_AUDIT.json')
    if replay['status']!='EXACT_REPLAY_PASS' or [(r['case'],r['rep']) for r in replay['checks']]!=[(c,0) for c in [0,1,5,10,14]]:raise ValueError('prespecified replay incomplete')
    if replay['freeze_sha256']!=sha(OUT/'freeze.json') or replay['index_sha256']!=sha(OUT/'index.json'):raise ValueError('replay identity changed')
    example=read(BASE/'EXAMPLE_RECEIPT.json')
    if example['version']!=p['version'] or example['status']!='completed' or example['freeze_sha256']!=sha(OUT/'freeze.json'):raise ValueError('actual invocation incomplete')
    if status.get('full_user_success'):
        gates=summary['empirical_gates']
        keys=['model_fdr_uppers_at_most_05','core_improvement_lower_positive','all_M0_nonnull_no_important_regression','drift_nonincrease_demonstrated_in_both_scenarios']
        if not all(gates[k] for k in keys) or gates['full_user_success_excluded_by_drift'] or not status.get('novelty_established'):
            raise ValueError('full success contradicts evidence/criteria')
    for directory in ['D001','D002','D003','C001']:
        frozen=read(BASE/directory/'freeze.json')
        for name,digest in frozen['files'].items():
            if sha(BASE/directory/name)!=digest:raise ValueError('historical frozen content changed:'+directory+'/'+name)
    figures=read(BASE/'figures/manifest.json')
    for name,digest in figures['outputs'].items():
        if sha(BASE/'figures'/name)!=digest:raise ValueError('figure artifact changed')
    if figures['source_sha256']!=sha(OUT/'summary.json') or figures['generator_sha256']!=sha(BASE/'make_figures.py'):raise ValueError('figure provenance changed')
    import xml.etree.ElementTree as ET
    tests=ET.parse(BASE/'frozen-tests.xml').getroot()
    suites=list(tests.iter('testsuite'))
    if sum(int(x.attrib.get('tests',0)) for x in suites)!=15 or any(int(x.attrib.get(k,0)) for x in suites for k in ['errors','failures','skipped']):raise ValueError('frozen tests not passed')
    files={}
    for path in sorted(BASE.rglob('*')):
        rel=path.relative_to(BASE)
        if not path.is_file() or path.name=='DELIVERY_MANIFEST.json' or any(part in ['__pycache__','.pytest_cache'] or part.startswith('test-tmp-') for part in rel.parts):continue
        files[rel.as_posix()]={'sha256':sha(path),'bytes':path.stat().st_size}
    write(BASE/'DELIVERY_MANIFEST.json',{'sealed_utc':datetime.now(timezone.utc).isoformat(),
       'version':p['version'],'bounded_phase_complete':status['bounded_phase_complete'],
       'full_user_success':status['full_user_success'],'files':files,
       'dependencies':{'R2_DELIVERY_MANIFEST_sha256':sha(R2/'DELIVERY_MANIFEST.json'),
                       'R2_C001_freeze_sha256':sha(R2/'C001/freeze.json'),
                       'V1_zip_sha256':sha(ROOT/'releases/K-NR-1.0.0.zip')},
       'exclusions':['test-tmp-* fixtures','__pycache__','.pytest_cache','this manifest itself'],
       'portability':'Requires preserved R2 C001/V1 trees and original absolute import paths described in README; relocation needs a separate audited entrypoint/manifest. Not a standalone wheel.'})
    print({'sealed_files':len(files),'bytes':sum(x['bytes'] for x in files.values()),'manifest_sha256':sha(BASE/'DELIVERY_MANIFEST.json')})

if __name__=='__main__':main()
