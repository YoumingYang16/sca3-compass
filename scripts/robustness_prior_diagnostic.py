"""Recreate the R0043 single energy-prior nonconvergence; numeric audit only."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma
from threadpoolctl import threadpool_limits
from robustness_screen import data
from sca3_compass.molecular_data import PROJECT_ROOT,digest
from sca3_compass.robustness_methods import evaluate_candidates,contrasts
from sca3_compass.robustness_prior import _student_logpdf,fit_energy_prior
from sca3_compass.robustness_io import write_json


def main():
    target=PROJECT_ROOT/'artifacts/robustness/prior-r43-case20-rep58-diagnostic.json'
    if target.exists():
        raise SystemExit('Preserve the existing diagnostic')
    protocol=json.loads((PROJECT_ROOT/'artifacts/robustness/R0043-screen.protocol.json').read_text(encoding='utf-8'))
    rng=np.random.default_rng(np.random.SeedSequence([protocol['seed'],20]))
    with threadpool_limits(limits=1):
        for _ in range(59):
            z,x,_=data(rng,protocol['cases'][20])
        _,info=evaluate_candidates(z,x,'frontier')
        shape=np.asarray(info['shape'][1]['matrix']);rho=info['fit']['rho']
        residual=(z-z.mean(-1)[...,None])@contrasts(z.shape[-1])
        q=np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(shape),residual)/(1-rho)
        q=q[::2];d=20
        def objective(params):
            ls,lnu=params
            density,kernel,ratio=_student_logpdf(np.log(q),d/2,ls,lnu)
            halfnu=np.exp(lnu)/2
            scale=-d/2+(d/2+halfnu)*ratio
            tail=halfnu*(digamma(d/2+halfnu)-digamma(halfnu))-d/2-halfnu*kernel+(d/2+halfnu)*ratio
            return -density.mean(),-np.array([scale.mean(),tail.mean()])
        results=[]
        for method in ['L-BFGS-B','SLSQP']:
            for start in [5.,30.]:
                fit=minimize(objective,[float(np.median(np.log(q/d))),np.log(start)],jac=True,method=method,
                    bounds=[(-9,9),(np.log(1.05),np.log(200))],
                    options={'maxiter':200,'ftol':1e-12,**({'gtol':1e-7} if method=='L-BFGS-B' else {})})
                results.append({'method':method,'start':start,'success':bool(fit.success),'message':str(fit.message),
                    'iterations':int(fit.nit),'objective':float(fit.fun),'parameters':fit.x.tolist(),
                    'gradient':objective(fit.x)[1].tolist()})
        current=fit_energy_prior(q,d)
    artifact={'phase':'NUMERICAL_DEVELOPMENT_DIAGNOSTIC','run_reference':'R0043','case_index':20,'repetition':58,'fold':1,
        'seed':protocol['seed'],'training_q':q.tolist(),'shape':shape.tolist(),'rho':rho,
        'current_fit':current,'optimizer_checks':results,
        'source_sha256':{str(p.relative_to(PROJECT_ROOT)):digest(p) for p in
            [Path(__file__),PROJECT_ROOT/'scripts/robustness_screen.py',PROJECT_ROOT/'src/sca3_compass/robustness_prior.py',PROJECT_ROOT/'src/sca3_compass/robustness_methods.py']}}
    write_json(target,artifact)
    print(json.dumps(results))


if __name__=='__main__':
    main()
