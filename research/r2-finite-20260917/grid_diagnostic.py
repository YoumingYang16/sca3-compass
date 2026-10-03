"""Fixed256-family saved-p rescore. No new data/reference or model selection."""
from pathlib import Path
import gzip
import json
import shutil
from datetime import datetime, timezone
import numpy as np
from experiment import read, write, sha, score, js
from finite_calibration import ebh
from grid_calibration import grid_calibrate

PHASE = Path(__file__).resolve().parent


def run():
    out = PHASE/'D003'
    if out.exists(): raise FileExistsError('preserve completed/partial diagnostic')
    out.mkdir()
    protocol = {'id':'R2FC-D003', 'stage':'DEVELOPMENT_SAME_P_NO_NEW_DATA',
                'source_index_sha256':sha(PHASE/'D002/index.json'), 'families':256,
                'method':'discrete-support normalization only; all pilot receipts/gamma/ranks fixed',
                'stop':'one pass over256 saved families, no tuning or new references'}
    write(out/'protocol.json', protocol)
    for f in ['grid_calibration.py', 'grid_diagnostic.py']: shutil.copy2(PHASE/f, out/f)
    write(out/'freeze.json', {'utc':datetime.now(timezone.utc).isoformat(),
                             'files':{f:sha(out/f) for f in ['protocol.json','grid_calibration.py','grid_diagnostic.py']}})
    index=read(PHASE/'D002/index.json'); rows=[]
    for entry in index['rows']:
        rp=PHASE/'D002'/entry['path']
        if sha(rp)!=entry['sha256']: raise ValueError('source record changed')
        with gzip.open(rp,'rt',encoding='utf-8') as stream: old=json.load(stream)
        ep=PHASE/'D002'/old['evidence_path']
        if sha(ep)!=old['evidence_sha256'] or sha(old['input_path'])!=old['input_sha256']:
            raise ValueError('source content changed')
        with np.load(ep) as a: p=a['p']; draws=len(a['reference']); olde=a['e_PB_main']
        with np.load(old['input_path']) as a: truth=a['truth']
        main=np.zeros((len(p),2)); ordinary=main.copy(); norm=[]
        for f in old['folds']:
            held=np.arange(len(p))%4==f['fold']; ec=[]
            for c in range(2):
                val,n=grid_calibrate(p[held,:,c], f['pilots'][c], 2*len(p), draws, c)
                ec.append(val); norm.append(n)
            gamma=np.asarray(f['gamma_from_pilot'])
            main[held]=(1-gamma)*ec[0]+gamma*ec[1]; ordinary[held]=ec[0]
        if np.any(main+1e-9 < olde): raise ArithmeticError('unexpected loss of evidence')
        metrics={**old['metrics'], 'PB_grid':score(ebh(main,.05),truth),
                 'PB_grid_ordinary':score(ebh(ordinary,.05),truth)}
        rows.append({'case':entry['case'],'rep':entry['rep'],'metrics':metrics,'normalizers':norm,
                     'source_record_sha256':entry['sha256']})
    write(out/'records.json',rows)
    summary=[]
    for case in sorted(set(r['case'] for r in rows)):
        sub=[r for r in rows if r['case']==case]; metrics={}
        for key in sub[0]['metrics']:
            metrics[key]={name:None if sub[0]['metrics'][key][name] is None else
                          float(np.mean([r['metrics'][key][name] for r in sub])) for name in ['power','fdp']}
        summary.append({'case':case,'reps':len(sub),'metrics':metrics})
        print(case,json.dumps({k:v for k,v in metrics.items() if k.startswith('PB') or k=='B_strong'}),flush=True)
    write(out/'summary.json',{'rows':summary,'records_sha256':sha(out/'records.json')})


if __name__=='__main__': run()
