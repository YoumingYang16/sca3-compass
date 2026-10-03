"""Deterministic critical-value diagnostic; no draws, no C1 outcome access."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.stats import norm,t
from sca3_compass.robustness_pivotal import predictive_table
from threadpoolctl import threadpool_limits
def main():
    rows=[]
    harmonic=float(np.sum(1/np.arange(1,65)));levels=[.01,.001,.05/(512*harmonic)]
    with threadpool_limits(1):
        for alpha in levels:
            row={'tail_probability':alpha,'normal_critical':float(norm.isf(alpha)),
                't5_critical':float(t.isf(alpha,5)),'t25_critical':float(t.isf(alpha,25))}
            for n in [32,128,512]:
                table=predictive_table(n,5);ids=np.flatnonzero(table['upper']<=alpha)
                row[f'pivotal_validation_n{n}_critical_upper']=float(table['z'][ids[0]]) if len(ids) else None
            rows.append(row)
    out=ROOT/'artifacts/robustness/R1R2-20260916/pivotal-efficiency-deterministic.json'
    with out.open('x',encoding='utf-8') as s:json.dump({'provenance':'DETERMINISTIC_MATHEMATICAL_DIAGNOSTIC_NOT_POWER_EXPERIMENT','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'rows':rows,'warning':'Critical values are dimensionless at each model standardization; not same-data power estimates, not an oracle comparison and not interval-certified numerical bounds. Explains part ofR0070loss, not a quantitative causal decomposition oftotalPower.'},s,indent=2)
    print(json.dumps(rows,indent=2));print(out)
if __name__=='__main__':main()
