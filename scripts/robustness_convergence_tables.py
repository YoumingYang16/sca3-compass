"""Regenerate diagnostic tables from audited results; never fit or simulate."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import numpy as np
import robustness_replay as rp
from robustness_convergence_auxiliary import compound_scale_reference

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    folder=ROOT/'artifacts/robustness/convergence-20260915'
    text=['# Bounded convergence: all-case numerical appendix',
        '', 'DEVELOPMENT REANALYSIS ONLY. No new random draws, model fitting, threshold tuning or acceptance decision.',
        'All power/FDR cells are percentages; differences are percentage points. Power higher is better; FDR is compared with nominal 5%, not optimized to zero.',
        'A=support-Simes/original package; B=support-Simes/candidate package; C=projection/original; D=projection/candidate.',
        'The package also recomputes weights, profiles and pilot. Raw and gated projection are different procedures; no cross-run ranking.',
        'The 95% t intervals below are pointwise Monte Carlo descriptions only. All paired bounded intervals and raw vectors remain in the input JSON files.', '']
    provenance={}
    mainline=rp.read_json(folder/'mainline-comparison.json')
    for run,data in mainline.items():
        text += [f'## {run}: all54 core cases','',
            'FDR bound uses a separate .05/54 family within this run; it is not the original 4374-interval family or cross-run confirmation.',
            '', '| Case | Main power | Mean signed FDP | FDR upper | Gap to observed best of15 |',
            '|---|---:|---:|---:|---:|']
        for row in data['per_case']:
            text.append(f"| {row['index']}: {row['case']['name']} | {100*row['main_power']:.4f} | {100*row['mean_fdp']:.4f} | {100*row['fdp_simultaneous']['upper']:.4f} | {100*row['paired_observed_best']['mean']:+.4f} |")
        text.append('')
    summaries={}
    for run in ['R0064','R0067']:
        file=folder/f'{run}-contribution-decomposition.json';data=rp.read_json(file)
        provenance[str(file.relative_to(ROOT))]=rp.sha(file)
        for package in dict.fromkeys(row['package'] for row in data):
            for mode in ['raw','gated']:
                text += [f'## {run}: {package}, {mode}','',
                    '| Scene | n | A power/FDR | B power/FDR | C power/FDR | D power/FDR | B-A | C-A | D-B | D-C | Interaction |',
                    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
                for row in data:
                    if (row['package'],row['projection_mode'])!=(package,mode):continue
                    cells=[]
                    for k in 'ABCD':
                        c=row['cells'][k]
                        power=f"{100*c['power']:.3f}" if row['power_defined'] else 'NA'
                        cells.append(f"{power}/{100*c['fdr']:.3f}")
                    effects=['NA']*5 if row['effects'] is None else [f"{100*v['mean']:+.3f}" for v in row['effects'].values()]
                    text.append('| '+' | '.join([row['case']['name'],str(row['repetitions']),*cells,*effects])+' |')
                text += ['', 'Paired pointwise approximate 95% intervals for B-A and D-B:', '',
                    '| Scene | B-A interval | D-B interval |','|---|---:|---:|']
                for row in data:
                    if (row['package'],row['projection_mode'])!=(package,mode) or row['effects'] is None:continue
                    cis=['['+', '.join(f'{100*x:+.3f}' for x in row['effects'][key]['approximate_t_interval'])+']'
                         for key in ['package_simple_B-A','projection_candidate_D-B']]
                    text.append('| '+' | '.join([row['case']['name'],*cis])+' |')
                text.append('')
        mechanisms=rp.read_json(folder/f'{run}-scale-mechanism-records.json')
        summaries[run]=[]
        for case in mechanisms:
            case_summary={'case':case['case'],'packages':{}}
            c=case['case']
            case_summary['known_compound_scale_reference']=compound_scale_reference(c['rho'],c.get('calibration_rho',c['rho']))
            case_summary['reference_notice']='Known-model ratio only, not substituted into any method; fitted nuisances and non-IG tails can invalidate exact interpretation.'
            for package in dict.fromkeys(r['package'] for r in case['fold_records']):
                records=[r for r in case['fold_records'] if r['package']==package]
                values=np.array([r['selected_multiplier'] for r in records])
                gamma=np.array([r['gamma'] for r in records])
                case_summary['packages'][package]={'fold_count':len(records),'scale_min_median_max':np.quantile(values,[0,.5,1]).tolist(),
                    'gamma_mean_by_sign':gamma.mean(axis=0).tolist(),'gamma_zero_entries':int((gamma==0).sum()),
                    'gamma_point9_entries':int((gamma==.9).sum()),'assumed_floor_active_folds':sum(r.get('floor_active',False) for r in records),
                    'scale_fallback_folds':sum(r['scale_fallback'] for r in records),
                    'selected_scale_nonconverged_folds':sum(not r['scale_fit_converged'] for r in records),
                    'projection_pilot_zero_folds':sum(r['pilot_projection']['zero_count_fallback'] for r in records)}
            summaries[run].append(case_summary)
    for p in [folder/'mainline-comparison.json',folder/'R0064-scale-mechanism-records.json',folder/'R0067-scale-mechanism-records.json']:
        provenance[str(p.relative_to(ROOT))]=rp.sha(p)
    (args.output/'all-case-tables.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    rp.write_new(args.output/'mechanism-summary.json',summaries)
    rp.write_new(args.output/'receipt.json',{'utc':datetime.now(timezone.utc).isoformat(),'source_sha256':rp.sha(__file__),
        'auxiliary_source_sha256':rp.sha(ROOT/'scripts/robustness_convergence_auxiliary.py'),'inputs':provenance,
        'outputs':{p.name:rp.sha(p) for p in args.output.iterdir() if p.is_file()},'new_simulations':0,
        'confirmation':False,'phase':'bounded development diagnosis'})
    print('TABLES COMPLETE',args.output)


if __name__=='__main__':main()
