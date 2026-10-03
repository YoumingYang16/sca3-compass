"""Bounded DEV replay/80-digit calibration check, not universal certification."""
from pathlib import Path
import gzip
import json
import platform
import sys
import time
import numpy as np
import mpmath as mp
import scipy
from threadpoolctl import threadpool_limits
from experiment import sha,read,write
from predictive_bridge import evaluate,kappa_estimator

PHASE=Path(__file__).resolve().parent


def high_kappa(cal):
    mp.mp.dps=80
    h=mp.matrix(6,5)
    for k in range(1,6):
        den=mp.sqrt(k*(k+1))
        for row in range(k): h[row,k-1]=1/den
        h[k,k-1]=-k/den
    values=[]
    for c in cal:
        matrix=mp.matrix(c.tolist()); y=matrix*h
        x=matrix*mp.matrix([mp.mpf(1)/6]*6)
        values.append((x.T*(y*y.T)**-1*x)[0]/2)
    values.sort(); n=len(values)
    median=values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2
    return median/((1+mp.sqrt(2))/2)


def main():
    start=time.perf_counter(); rows=[]
    with threadpool_limits(1):
        for case in [0,5,10]:
            rp=PHASE/f'D002/raw/case-{case:02}/rep-00000.json.gz'
            with gzip.open(rp,'rt',encoding='utf-8') as stream: old=json.load(stream)
            with np.load(old['input_path']) as a: z,cal=a['z'],a['calibration']
            with np.load(PHASE/'D002'/old['evidence_path']) as a: oldp,olde=a['p'],a['e_PB_main']
            result=evaluate(z,cal,seed=old['algorithm_seed'],reference_draws=4095)
            assert result['status']=='completed'
            assert np.array_equal(result['p'],oldp)
            assert np.array_equal(result['evidence']['PB_main'],olde)
            assert np.all(result['evidence']['PB_grid']+1e-9>=olde)
            high=high_kappa(cal); low=kappa_estimator(cal)
            relative=float(abs(mp.mpf(low)/high-1))
            if relative>1e-10: raise ArithmeticError('calibration numerical discrepancy')
            rows.append({'case':case,'rep':0,'source_record_sha256':sha(rp),
                         'p_and_continuous_e_exact_replay':True,
                         'kappa_float':low,'kappa_80digit':str(high),'relative_error':relative})
    write(PHASE/'numerical-audit.json',{'rows':rows,'seconds':time.perf_counter()-start,
          'scope':'3 specified DEV replay cases plus80-digit kappa, NOT full numerical/FDR certification',
          'environment':{'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'mpmath':mp.__version__}})
    print(json.dumps(rows),flush=True)


if __name__=='__main__': main()
