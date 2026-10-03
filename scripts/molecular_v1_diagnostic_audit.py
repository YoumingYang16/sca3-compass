"""Descriptive diagnostic counts from existing frozen records; no new inference."""
from pathlib import Path
import argparse
import hashlib
import gzip
import json
import sys
import time
import shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sca3_compass.robustness_io import write_json
from research_window import wait_start_gate,cooperative_stop


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as stream:return json.load(stream)
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    wait_start_gate()
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if out.exists():raise FileExistsError('Preserve previous audits')
    base=ROOT/'artifacts/robustness';index_path=base/'R0078-v1-index.json'
    index=read_json(index_path);p=index['protocol']
    if not index['complete']:raise ValueError('Complete original batch required')
    registry=read_json(base/'EXPERIMENT_REGISTRY.json')
    entry=next(x for x in registry['experiments'] if x['id']=='R0078')
    if sha(index_path)!=entry['sha256']:raise ValueError('Index integrity')
    begin=time.perf_counter();rows={};pattern_families=[];guard_families=[];seen=set()
    for ref in index['receipts']:
        if cooperative_stop():raise InterruptedError('Preserve all original records')
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Record integrity')
        r=read_json(ref['path']);i,j=r['case_index'],r['rep']
        if (i,j) in seen or r['status']!='completed':raise ValueError('Invalid family')
        seen.add((i,j))
        c=rows.setdefault(str(i),{'families':0,'folds':0,'pattern_failure_folds':0,
            'calibration_guard_failure_folds':0,'gamma_zero_direction_folds':0,
            'gamma_one_direction_folds':0,'gamma_interior_direction_folds':0})
        c['families']+=1;pattern=False;guard=False
        for fold in r['diagnostics']['folds']:
            c['folds']+=1
            if fold['pattern']['fallback']:
                pattern=True;c['pattern_failure_folds']+=1
            if not fold['guard_success']:
                guard=True;c['calibration_guard_failure_folds']+=1
            for gamma in fold['gamma']:
                if not 0<=gamma<=1:raise ValueError('Gamma outside convex range')
                c['gamma_zero_direction_folds' if gamma==0 else 'gamma_one_direction_folds' if gamma==1 else 'gamma_interior_direction_folds']+=1
        if pattern:pattern_families.append([i,j])
        if guard:guard_families.append([i,j])
    expected={(i,j) for i,n in enumerate(p['counts']) for j in range(n)}
    if seen!=expected:raise ValueError('Family membership')
    totals={k:sum(c[k] for c in rows.values()) for k in next(iter(rows.values()))}
    out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/'source.py')
    write_json(out/'summary.json',{'run_id':'R0078','kind':'DESCRIPTIVE_EXISTING_DIAGNOSTICS_NO_NEW_DATA',
        'complete':True,'index_sha256':sha(index_path),'source_sha256':sha(__file__),
        'totals':totals,'per_case':rows,'pattern_fallback_families':pattern_families,
        'calibration_guard_fallback_families':guard_families,'seconds':time.perf_counter()-begin,
        'notice':'Gamma zero is ordinary-component gating, not necessarily failure; counts are fold-direction units, NOT independent statistical replications. No estimate of its causal contribution.'})
    print(json.dumps({'complete':True,'totals':totals,'pattern_families':len(pattern_families),'guard_families':len(guard_families)}))


if __name__=='__main__':main()
