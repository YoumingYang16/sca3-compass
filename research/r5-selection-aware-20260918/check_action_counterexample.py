"""M007 deterministic K2 check; neither an R5 simulation nor FDR evidence."""
import shutil
import numpy as np
from scipy.special import roots_hermitenorm
from r5_common import PHASE,write,sha


def main():
    out=PHASE/'M007';out.mkdir(exist_ok=False)
    names=['check_action_counterexample.py','ACTION_KERNEL_THEOREM.md','r5_common.py']
    for name in names:shutil.copy2(PHASE/name,out/name)
    write(out/'freeze.json',{'files':{n:sha(out/n) for n in names}})
    write(out/'protocol.json',{'epsilon':.15,'tau':.6,'slacks':[0,.25,1,2],
        'quadrature_orders':[64,128],'role':'DIAGNOSTIC_ONLY abstract Gaussian error model;no FDR claim'})
    rows=[];ep=.15;tau=.6;d=1+2*(ep**2+tau**2)
    for s in [0,.25,1,2]:
        c=np.exp(-s*s/d)/np.sqrt(d);prob=.5-.25*c;mean=-tau*tau*s*c/(2*d)/prob
        quadrature=[]
        for n in [64,128]:
            z,w=roots_hermitenorm(n);w=w/np.sqrt(2*np.pi);x=s+ep*z[:,None];y=tau*z[None,:]
            gate=.5-.25*np.exp(-(x-y)**2);weights=w[:,None]*w[None,:]
            p=float(np.sum(weights*gate));m=float(np.sum(weights*gate*y)/p)
            quadrature.append({'n':n,'prob':p,'mean':m,'mean_minus_formula':m-mean})
        rows.append({'slack':s,'prob':float(prob),'mean':float(mean),'quadrature':quadrature})
    write(out/'result.json',{'status':'completed','rows':rows,'conclusion':'s0 target mean0, s>0 negative: stochastic order fails for this weight, NOT FDR proof'})
    print(rows)


if __name__=='__main__':main()
