"""Read-only mechanism association in ALL pre-existing R0075 DEV records.

No refitting, p-values, new experiments or C2 outcomes. Known simulator rho is
used solely to describe estimation error, never for deployed decisions.
"""
from research_window import wait_start_gate,cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime,timezone
import sys
import shutil
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from robustness_r2_development import read,sha
from sca3_compass.robustness_io import content_digest,write_json


def kappa(rho,k=6):
    r=np.asarray(rho,float)
    if np.any(r<=-1/(k-1)) or np.any(r>=1) or not np.isfinite(r).all():raise ValueError('Admissible shape required')
    return (1+(k-1)*r)/(k*(1-r))


def describe(x):
    return {'mean':float(np.mean(x)),'q05':float(np.quantile(x,.05)),
            'median':float(np.median(x)),'q95':float(np.quantile(x,.95))}


def main():
    base=ROOT/'artifacts/robustness';index=read(base/'R0075-results-index.json')
    analysis=read(base/'R0075-analysis/summary.json');protocol=index['settings']
    if not index['complete'] or not analysis['complete'] or analysis['index_sha256']!=sha(base/'R0075-results-index.json'):
        raise ValueError('Completed audited original DEV required')
    if len(index['receipts'])!=4000 or len(protocol['cases'])!=8:raise ValueError('All4000 DEV families required')
    out=base/'R0075-scale-diagnostics';out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'generate_source.py')
    grouped={i:[] for i in range(8)}
    for receipt in index['receipts']:grouped[receipt['case_index']].append(receipt)
    rows=[];start=time.perf_counter()
    for case,receipts in grouped.items():
        if cooperative_stop():raise InterruptedError('Finite diagnostic stop')
        receipts.sort(key=lambda r:r['rep'])
        if [r['rep'] for r in receipts]!=list(range(500)):raise ValueError('Incomplete DEV case')
        values=[]
        for receipt in receipts:
            if sha(receipt['path'])!=receipt['sha256']:raise ValueError('DEV record changed')
            r=read(receipt['path'])
            if r['status']!='completed' or r['protocol_digest']!=content_digest(protocol):raise ValueError('Wrong completed DEV')
            folds=r['diagnostics_R2']['folds'];a,b=folds
            # Shared calibration estimate must not be counted twice as iid.
            if a['original']['rho']!=b['original']['rho']:raise ValueError('Unexpected differing shared calibrationrho')
            values.append({'rho':a['original']['rho'],
                'min_Krho':min(f['guarded']['rho'] for f in folds),
                'max_Krho':max(f['guarded']['rho'] for f in folds),
                'calibration_gaussian':a['original']['calibration_fit']['gaussian_bic_selected'],
                'target_gaussian_fraction':sum(f['original']['prior']['gaussian_bic_selected'] for f in folds)/2,
                'K_failed_folds':sum(not f['K_success'] for f in folds),
                'P_fdp':r['metrics']['R2P_pilotc0.5_projection_gate_eBH']['fdp'],
                'K_fdp':r['metrics']['R2K_pilotc0.5_projection_gate_eBH']['fdp']})
        truth=protocol['cases'][case]['rho'];rho=np.array([v['rho'] for v in values])
        minrho=np.array([v['min_Krho'] for v in values]);maxrho=np.array([v['max_Krho'] for v in values])
        row={'case_index':case,'case':protocol['cases'][case],'n_independent_DEV_families':500,
            'true_rho_simulator_diagnostic_only':truth,'rho_hat':describe(rho),
            'kappa_plugin_over_true':describe(kappa(rho)/kappa(truth)),
            'kappa_min_K_over_true':describe(kappa(minrho)/kappa(truth)),
            'rho_plugin_below_true_fraction':float(np.mean(rho<truth)),
            'at_least_one_Kfold_rho_below_true_fraction':float(np.mean(minrho<truth)),
            'both_Kfold_rho_below_true_fraction':float(np.mean(maxrho<truth)),
            'calibration_Gaussian_selection_fraction':float(np.mean([v['calibration_gaussian'] for v in values])),
            'target_Gaussian_fold_fraction':float(np.mean([v['target_gaussian_fraction'] for v in values])),
            'K_failed_folds':sum(v['K_failed_folds'] for v in values),'prespecified_rho_error_strata':{}}
        for label,mask in [('under',rho<truth),('not_under',rho>=truth)]:
            group=[v for v,yes in zip(values,mask) if yes]
            row['prespecified_rho_error_strata'][label]={'families':len(group),
                'P_any_false_discovery_fraction':float(np.mean([v['P_fdp']>0 for v in group])) if group else None,
                'P_mean_fdp':float(np.mean([v['P_fdp'] for v in group])) if group else None,
                'K_mean_fdp':float(np.mean([v['K_fdp'] for v in group])) if group else None}
        rows.append(row)
        print('DEV only scale audit',case,flush=True)
    result={'run_id':'R0075','complete':True,'kind':'DESCRIPTIVE_EXISTING_DEV_MECHANISM_AUDIT',
        'index_sha256':sha(base/'R0075-results-index.json'),'analysis_sha256':sha(base/'R0075-analysis/summary.json'),
        'script_sha256':sha(__file__),'rows':rows,'elapsed_seconds':time.perf_counter()-start,
        'finished_utc':datetime.now(timezone.utc).isoformat(),'provenance':'SIMULATION_NOT_PATIENT_DATA',
        'limits':['No confidence intervals or new confirmation','Associations are not controlled causal comparisons',
                  'Knownrho is simulator-only diagnosis, not a deployed gate',
                  'Finitebootstrap maxrho undercoverage frequencies are descriptive, not a confidence certificate',
                  'No C2 outcome read and no frozen-method/threshold change']}
    write_json(out/'summary.json',result)
    print(out,flush=True)


if __name__=='__main__':main()
