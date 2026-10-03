"""M006 fixed law check, no family Power/FDR and no acceptance decisions."""
from pathlib import Path
import sys,shutil,time
import numpy as np
from scipy.stats import beta,f,kstest
from scipy.integrate import quad
from triangular_calibration import ConditionalTriangular
from r5_common import PHASE,write,sha


def main():
    out=PHASE/'M006';out.mkdir(exist_ok=False);started=time.perf_counter()
    files=['check_triangular_information.py','triangular_calibration.py','ancillary_calibration.py',
           'selection_profile.py','r5_common.py','TRIANGULAR_CALIBRATION.md']
    for name in files:shutil.copy2(PHASE/name,out/name)
    write(out/'freeze.json',{'files':{name:sha(out/name) for name in files}})
    write(out/'protocol.json',{'purpose':'DIAGNOSTIC_ONLY kernel law/FI, not FDR/Power',
        'gaussian_tuples':32768,'seed':605018,'conditional_draws_each':8192,
        'fixtures':['zero residuals N4 angles(.04,.3,.65)','w(-3,-1,1,3) sameangles',
                    'zero residuals N4 angles(1e-7,1e-4,.02)'],
        'no_extension':True,'independent_R5_confirmation':False})
    n=32768;rng=np.random.default_rng(605018);e=rng.normal(size=(n,4,5));z=rng.normal(size=(n,4))
    _,r=np.linalg.qr(e.swapaxes(1,2),mode='reduced')
    r*=np.sign(np.diagonal(r,axis1=1,axis2=2))[:,:,None]
    t=np.linalg.solve(r.swapaxes(1,2),z[:,:,None])[:,:,0]
    q=np.cumsum(t*t,axis=1);prev=np.c_[np.zeros(n),q[:,:3]]
    transformed=beta.cdf(t*t/(1+q),.5,np.array([5,4,3,2])/2)
    score=2.5-np.sum(1/(1+q[:,:3]),axis=1)-1.5/(1+q[:,3])
    radial_score=-2+3*q[:,3]/(1+q[:,3])
    results={'n':n,'score_mean':float(score.mean()),'score_mean_se':float(score.std(ddof=1)/np.sqrt(n)),
        'information_mean':float(np.mean(score**2)),'information_se':float(np.std(score**2,ddof=1)/np.sqrt(n)),
        'information_theory':17/24,'radial_information_mean':float(np.mean(radial_score**2)),
        'radial_information_theory':.5,'innovation_KS':[{'statistic':float(kstest(transformed[:,d],'uniform').statistic),
            'pvalue':float(kstest(transformed[:,d],'uniform').pvalue)} for d in range(4)],
        'innovation_correlation':np.corrcoef(transformed.T).tolist(),
        'radius_F42_KS':{'statistic':float(kstest(q[:,-1]/2,f(4,2).cdf).statistic),
                         'pvalue':float(kstest(q[:,-1]/2,f(4,2).cdf).pvalue)}}
    b=np.array([[2,0,0,0],[.7,1,0,0],[-.3,.2,.5,0],[.2,-.8,.5,3.]])
    radii=np.exp(np.linspace(-20,20,256));ee=np.einsum('ij,njk->nik',b,e[:256])*radii[:,None,None]
    zz=(z[:256]@b.T)*radii[:,None]
    _,rr=np.linalg.qr(ee.swapaxes(1,2),mode='reduced');rr*=np.sign(np.diagonal(rr,axis1=1,axis2=2))[:,:,None]
    tt=np.linalg.solve(rr.swapaxes(1,2),zz[:,:,None])[:,:,0]
    results['lower_transform_relative_max_error']=float(np.max(np.abs(tt-t[:256])/(1+np.abs(t[:256]))))
    fixtures=[(np.zeros(4),[.04,.3,.65]),(np.array([-3.,-1.,1.,3.]),[.04,.3,.65]),
              (np.zeros(4),[1e-7,1e-4,.02])]
    checks=[];raw={}
    for i,(w,a) in enumerate(fixtures):
        law=ConditionalTriangular(w,np.tile(a,(4,1)));fun=lambda y:np.exp(law.log_density(y)-law.shift)
        norm=quad(fun,-100,100,epsabs=1e-11)[0]
        mean=quad(lambda y:y*fun(y),-100,100,epsabs=1e-11)[0]/norm-1
        samples,receipt=law.draw(np.random.default_rng(605019+i),8192);raw[f'conditional_{i}']=samples
        checks.append({'fixture':i,'mean_expected':mean,'mean_sample':float(samples.mean()),
            'sample_mean_se':float(samples.std(ddof=1)/np.sqrt(len(samples))),'receipt':receipt})
    results['conditional_checks']=checks;results['seconds']=time.perf_counter()-started
    results['status']='completed';results['acceptance_gate_conclusion']='NONE; diagnostic only'
    with (out/'raw.npz').open('xb') as file:np.savez_compressed(file,t=t,q=q,score=score,**raw)
    write(out/'result.json',results);print(results)


if __name__=='__main__':main()
