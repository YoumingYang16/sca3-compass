"""Descriptive receipts only, no extra tests or candidate selection."""
from pathlib import Path
import gzip,json
import numpy as np
from experiment import read,write,sha
PHASE=Path(__file__).resolve().parent


def main():
    out=PHASE/'C001'; index=read(out/'index.json'); p=read(out/'protocol.json'); rows=[]
    for case in range(len(p['cases'])):
        records=[]
        for entry in index['rows']:
            if entry['case']!=case: continue
            path=out/entry['path']
            if sha(path)!=entry['sha256']: raise ValueError('record changed')
            with gzip.open(path,'rt',encoding='utf-8') as stream: records.append(json.load(stream))
        folds=[f for r in records for f in r['folds']]
        gamma=np.asarray([f['gamma_from_pilot'] for f in folds])
        normalizers=np.array([[x['normalizer'] for x in f['grid_calibrators']] for f in folds])
        ratios=[r['kappa_hat']/r['kappa_true_cal'] for r in records if r['kappa_hat'] is not None]
        rows.append({'case':case,'name':p['cases'][case]['name'],'whole_families':len(records),
                     'fold_receipts_dependent_not_repetitions':len(folds),
                     'gamma_zero_fraction':float(np.mean(gamma==0)),
                     'all_gammas_zero_family_fraction':float(np.mean([bool(r['folds']) and all(all(x==0 for x in f['gamma_from_pilot']) for f in r['folds']) for r in records])),
                     'training_pattern_nonconvergence_fraction':float(np.mean([not f['training_pattern_converged'] for f in folds])),
                     'pilot_pattern_nonconvergence_fraction':float(np.mean([not f['pilot_pattern_converged'] for f in folds])),
                     'normalizer_median_ordinary_projection':np.median(normalizers,axis=0).tolist(),
                     'kappa_hat_over_true_cal_q10_q50_q90':np.quantile(ratios,[.1,.5,.9]).tolist(),
                     'interpretation':'post-confirmation descriptive mechanism receipts, no new inferential comparison or causal attribution'})
    write(PHASE/'mechanism-summary.json',{'rows':rows,'index_sha256':sha(out/'index.json'),
          'warning':'folds/signs are dependent subunits, not independent repetitions; no confidence/significance claim'})
    print('descriptive receipts summarized for',len(rows),'scenes',flush=True)


if __name__=='__main__': main()
