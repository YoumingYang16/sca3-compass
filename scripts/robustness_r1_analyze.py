"""Recompute whole-family paired development summaries from immutable records."""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def read(p):
    with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else Path(p).open(encoding='utf-8')) as s:return json.load(s)

def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def summarize(run,verify=False):
    index=read(ROOT/'artifacts/robustness'/f'{run}-results-index.json');protocol=index['settings']
    grouped={i:[] for i in range(len(protocol['cases']))}
    for receipt in index['receipts']:
        if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Record hash mismatch')
        r=read(receipt['path'])
        if verify:
            for field in ['input','evidence']:
                if sha(r[field+'_path'])!=r[field+'_sha256']:raise ValueError(field+' hash mismatch')
        if r['status']!='completed':raise ValueError('Failure requires separate accounting, not omission')
        # Keep only paths globally; large nuisance-fit receipts are loaded one case at a time.
        grouped[r['case_index']].append(receipt)
    rows=[]
    for i,case_receipts in grouped.items():
        if not case_receipts:continue
        reps=[read(receipt['path']) for receipt in case_receipts]
        reps.sort(key=lambda r:r['rep']);names=list(reps[0]['metrics']);n=len(reps)
        if len({r['rep'] for r in reps})!=n:raise ValueError('Duplicate repetition')
        power=np.array([[r['metrics'][m]['power'] for m in names] for r in reps])
        fdp=np.array([[r['metrics'][m]['fdp'] for m in names] for r in reps])
        metrics={m:{'power':float(power[:,j].mean()) if reps[0]['n_true_signed'] else None,
            'fdp':float(fdp[:,j].mean()),'power_mcse':float(power[:,j].std(ddof=1)/np.sqrt(n)),
            'fdp_mcse':float(fdp[:,j].std(ddof=1)/np.sqrt(n)),
            'mean_tp':float(np.mean([r['metrics'][m]['tp'] for r in reps])),
            'mean_fp':float(np.mean([r['metrics'][m]['fp'] for r in reps]))} for j,m in enumerate(names)}
        prefix='R1B_guard' if protocol.get('backend')=='bootstrap_guard' else 'R1'
        families={'H':protocol['H_baselines'],
            'F_safe':[f'{prefix}_pilotc{c}_{m}_eBH' for c in protocol['F_multipliers'] for m in protocol['F_safe_kernels']],
            'F_empirical':[f'{prefix}_pilotc{c}_{m}_eBH' for c in protocol['F_multipliers'] for m in protocol['F_empirical_kernels']]}
        comparisons={};main=protocol['R1_main']
        for family,labels in families.items():
            b=max(labels,key=lambda m:metrics[m]['power'] or 0);delta=power[:,names.index(main)]-power[:,names.index(b)]
            comparisons[family]={'sample_envelope_method':b,'delta':float(delta.mean()) if reps[0]['n_true_signed'] else None,
                'paired_mcse_at_sample_winner':float(delta.std(ddof=1)/np.sqrt(n)),
                'warning':'Development sample envelope and unadjusted MCSE, NOT selection-adjusted confidence bounds'}
        folds=[f for r in reps for f in r['diagnostics_R1']['folds']]
        rows.append({'case_index':i,'case':protocol['cases'][i],'n':n,'methods':metrics,'comparisons':comparisons,
            'fallback_folds':sum(f['fallback'] for f in folds),'total_folds':len(folds),
            'gamma_mean':float(np.mean([f['gamma'] for f in folds])),
            'seconds_mean':float(np.mean([r['elapsed_seconds'] for r in reps])),
            'cpu_seconds_total':float(sum(r['cpu_seconds'] for r in reps)),
            'tail_diagnostics':reps[0]['diagnostics_R1']['tail_table'],
            'null_input':{m:{q:float(np.mean([r['null_input'][m]['below_threshold'][q]/r['null_input'][m]['null_claims'] for r in reps])) for q in ['0.0001','0.001','0.01','0.05']} for m in reps[0]['null_input']}})
    result={'run_id':run,'complete':index['complete'],'phase':protocol['phase'],'index_sha256':sha(ROOT/'artifacts/robustness'/f'{run}-results-index.json'),
        'records_verified':sum(len(v) for v in grouped.values()),'full_arrays_verified':verify,'rows':rows}
    dest=ROOT/'artifacts/robustness'/f'{run}-summary.json';dest.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('case | R0 P/FDR | R1 P/FDR | Fsafe P/FDR | Femp P/FDR | H delta pp | Fsafe delta pp | gamma | fallback')
    for row in rows:
        def fmt(m):
            v=row['methods'][m];return f"{100*v['power']:.2f}/{100*v['fdp']:.2f}" if v['power'] is not None else f"NA/{100*v['fdp']:.2f}"
        b=row['comparisons'];d=lambda k:'NA' if b[k]['delta'] is None else f"{100*b[k]['delta']:+.2f}"
        print(row['case']['name'],'|',fmt(protocol['R0_main']),'|',fmt(protocol['R1_main']),'|',fmt(b['F_safe']['sample_envelope_method']),'|',fmt(b['F_empirical']['sample_envelope_method']),'|',d('H'),'|',d('F_safe'),'|',round(row['gamma_mean'],3),'|',row['fallback_folds'])
    print('summary',dest,'repetitions',result['records_verified'],'cpu seconds',sum(r['cpu_seconds_total'] for r in rows))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--verify-arrays',action='store_true');a=p.parse_args();summarize(a.run_id,a.verify_arrays)
