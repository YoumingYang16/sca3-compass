"""Deterministic quadrature diagnostic, NOT a Monte Carlo confirmation."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit
from r5_common import PHASE,write,sha

def moments(w):
    w=np.array(w,float);w=w-w.mean();n=len(w)
    derivative=lambda y:np.sum(2-3*expit(y+w+np.log(2)))
    ld=lambda y:np.sum(np.log(8)+2*(y+w)-3*np.logaddexp(0.,y+w+np.log(2)))
    mode=brentq(derivative,-w.max()-10,-w.min()+10);shift=ld(mode)
    bounds=sorted(set([-w.max()-60,-w.min()+60,mode,*(-w).tolist()]))
    def integrate(power):
        vals=[quad(lambda y:(y-mode)**power*np.exp(ld(y)-shift),l,u,epsabs=1e-9,epsrel=1e-10,limit=200)
              for l,u in zip(bounds[:-1],bounds[1:])]
        return sum(x[0] for x in vals),sum(x[1] for x in vals)
    z,ez=integrate(0);m,em=integrate(1);v,ev=integrate(2)
    return {'mode':float(mode),'mean':float(mode+m/z),'variance':float(v/z-(m/z)**2),
            'quadrature_errors':[ez,em,ev],'finite_integration_limits':[bounds[0],bounds[-1]],
            'interpretation':'numeric diagnostic with finite tail cutoff60, not interval-arithmetic proof'}

def main():
    out=PHASE/'M001';out.mkdir(exist_ok=False)
    write(out/'protocol.json',{'kind':'DETERMINISTIC_QUADRATURE_CHECK','code_sha':sha(__file__),
        'n':[4,5,6,12,32],'L':[2,5,10,20,40],'stop':'exactly25finitepatterns, no model fitting or family simulation',
        'target':'proposed conditional precision boundary, not Power/FDR nor prevalence'})
    rows=[]
    for n in [4,5,6,12,32]:
        for L in [2,5,10,20,40]:
            if n%3==0:w=[-2*L]*(n//3)+[L]*(2*n//3)
            else:
                k=(2*n)//3;w=[L]*k+[-L]*(n-k-1)+[0]
            result=moments(w);row={'n':n,'L':L,**result};rows.append(row)
            print(n,L,round(result['variance'],6),flush=True)
    write(out/'result.json',{'rows':rows,'asymptotic_reference_nonmultiple':float(np.pi**2/3-1),
                             'asymptotic_variance_over_L2_multiple':.75,'gates_passed':[]})
if __name__=='__main__':main()
