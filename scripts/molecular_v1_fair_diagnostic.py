"""Rescore EXISTING DEV ordinary K component; no new data or fitting."""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
import sys
import shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from molecular_v1_validation import read,sha,score
from sca3_compass.robustness_io import write_json
from sca3_compass.robustness_pilot_selection import apply_pilot_multiplier
from sca3_compass.molecular_methods import ebh,fdr_adjust


def main():
    base=ROOT/'artifacts/robustness';index=read(base/'R0077-v1-index.json')
    out=base/'R0077-fair-diagnostic';out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'source.py');cases={};records=[]
    for ref in index['receipts']:
        if sha(ref['path'])!=ref['sha256']:raise ValueError('Record hash')
        row=read(ref['path']);i,j=row['case_index'],row['rep']
        if row['status']!='completed':raise ValueError('Missing DEV data')
        for name in ['input','evidence']:
            if sha(row[name+'_path'])!=row[name+'_sha256']:raise ValueError('Array hash')
        with np.load(row['input_path'],allow_pickle=False) as inp,np.load(row['evidence_path'],allow_pickle=False) as arrays:
            p=arrays['p_K_ordinary_bonf'];truth=inp['truth'];e=np.empty_like(p)
            for f in row['diagnostics']['folds']:
                held=np.arange(len(p))%2==f['fold']
                e[held]=apply_pilot_multiplier(p[held],f['pilots']['ordinary_bonf'],2*len(p),.5)[0]
            metrics={'conditional_eBH':score(ebh(e,.05),truth),'conditional_BY':score(fdr_adjust(p,'BY')<=.05,truth)}
        record={'case':i,'rep':j,'source_sha256':ref['sha256'],'metrics':metrics,
                'candidate':row['metrics']['K_NR'],'raw_fair_plugin':row['metrics']['B_fair_plugin'],
                'original_calibration':row['diagnostics']['folds'][0]['original']['calibration_fit']}
        records.append(record);cases.setdefault(i,[]).append(record)
    summary=[]
    for i,rows in sorted(cases.items()):
        stats={}
        for m in ['conditional_eBH','conditional_BY']:
            stats[m]={'fdp':float(np.mean([r['metrics'][m]['fdp'] for r in rows])),
                      'power':None if rows[0]['metrics'][m]['power'] is None else float(np.mean([r['metrics'][m]['power'] for r in rows])),
                      'K_minus':None if rows[0]['metrics'][m]['power'] is None else float(np.mean([r['candidate']['power']-r['metrics'][m]['power'] for r in rows]))}
        summary.append({'case':i,'methods':stats})
    write_json(out/'result.json',{'kind':'DEV_ONLY_EXISTING_DATA_RESCORING_NO_NEW_SAMPLES','source_run':'R0077',
        'preselected_comparators':['conditional_eBH','conditional_BY'],'records':records,'scenes':summary})
    print('Completed rescore',len(records))


if __name__=='__main__':main()
