"""M005 predetermined deterministic two-point information bound, no simulation."""
import shutil
from datetime import datetime,timezone
import numpy as np
from scipy.stats import chi2
from r5_common import PHASE,sha,write

def bound(nt,r,q=.05):
    if nt<1 or not 0<r<=1 or not 0<q<1:raise ValueError('invalid binary experiment')
    return float(chi2.cdf(chi2.ppf(q,4*nt)/r,4*nt))

def main():
    out=PHASE/'M005';out.mkdir(exist_ok=False)
    files=['check_information_boundary.py','FINITE_INFORMATION_BOUNDARY.md','r5_common.py']
    for f in files:shutil.copy2(PHASE/f,out/f)
    write(out/'protocol.json',{'utc':datetime.now(timezone.utc).isoformat(),'kind':'DETERMINISTIC_INFORMATION_BOUND_EVALUATION',
        'Nt':[4,12,32],'r':[.95,.8,.5],'q':.05,'new_data':False,'formal_confirmation':False,
        'scope':'Gaussian full-input submodel, prior-averagePower upper bound; NOT stored C1fixed-effect Power bound'})
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in files+['protocol.json']}})
    rows=[]
    for n in [4,12,32]:
        assert np.isclose(bound(n,1.),.05,rtol=1e-10)
        vals=[]
        for r in [.95,.8,.5]:
            d=4*n;cut=d*np.log(1/r)/(1-r)
            tv=float(chi2.cdf(cut,d)-chi2.cdf(r*cut,d))
            v=bound(n,r);vals.append(v)
            row={'Nt':n,'ratio_kappa1_over_kappa0':r,'d':d,'nominal_q':.05,
                 'NP_upper_any_discovery_and_prior_Power':v,'total_variation':tv,
                 'looser_q_plus_TV_bound':min(1.,.05+tv),
                 'KL_P1_P0':float(2*n*(r-1-np.log(r)))}
            assert v<=.05+tv+1e-12
            rows.append(row);print(row,flush=True)
        assert np.all(np.diff(vals)>=0)
    write(out/'result.json',{'status':'DETERMINISTIC_BOUND_EVALUATED_NOT_METHOD_SUCCESS','rows':rows,
        'scope_note':'Exact NP binary upper bound; boundary-only cannot meet R5G3/G4',
        'proof_status':'INTERNAL_CHALLENGE_NOT_PRIORITY_CERTIFIED','EXTERNAL_REVIEW':'NOT_CONDUCTED'})

if __name__=='__main__':main()
