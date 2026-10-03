"""Bounded characteristic-moment scale exploration on TRAIN fixtures only.

No integration, identification theorem, FDR claim, or confirmation. Uses
E[exp(i*w*a'M)|v]=c(a,w)*phi_t(w*sqrt(tau*v)) under effects independent of v.
Unlike squared-response regression the observed moments have modulus one.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import minimize_scalar,nnls
from scipy.special import kve,gammaln
from sca3_compass.robustness_io import write_json
from sca3_compass.robustness_continuous_scale import _gamma_rule


def student_cf(argument,df):
    argument=np.abs(np.asarray(argument,float))
    if not np.isfinite(argument).all() or df<3 or np.isnan(df):
        raise ValueError('Finite arguments and df>=3 needed for this diagnostic')
    if np.isinf(df):return np.exp(-argument**2/2)
    a=df/2;y=np.sqrt(df)*argument
    small=argument<1e-5
    result=np.empty_like(y)
    result[small]=1-df/(df-2)*argument[small]**2/2
    q=y[~small]
    with np.errstate(divide='ignore'):
        result[~small]=np.exp((1-a)*np.log(2)-gammaln(a)+a*np.log(q)+np.log(kve(a,q))-q)
    bad=~np.isfinite(result)|(result>1+1e-12)
    if np.any(bad):
        # Large-order K_a(x) can overflow even when the CF itself is near1.
        # Compute the SAME Gamma-mixture integral, not a Gaussian substitute.
        z=argument[bad]
        estimates=[]
        for order in [64,128]:
            nodes,weights=_gamma_rule(order,a)
            estimates.append(np.exp(-a*z[:,None]**2/(2*nodes))@weights)
        if np.max(np.abs(estimates[0]-estimates[1]))>2e-12:
            raise FloatingPointError('Gamma characteristic integral failed refinement')
        result[bad]=estimates[1]
    if not np.isfinite(result).all() or np.any(result<0) or np.any(result>1+1e-12):
        raise FloatingPointError('Unrepresentable characteristic function')
    return np.minimum(result,1.)


def cf_fit(x,v,shape,df,frequency_mode='base'):
    x=np.asarray(x);v=np.asarray(v);shape=np.asarray(shape)
    spread=float(np.std(np.log(v)))
    if spread<1e-5:
        return {'status':'NO_VARIANCE_CONTRAST_NOT_IDENTIFIED','log_variance_spread':spread}
    directions=[*np.eye(4),np.ones(4)]
    for i in range(4):
        for j in range(i):
            d=np.zeros(4);d[i]=1;d[j]=-1;directions.append(d)
    directions=np.array(directions)
    directions/=np.sqrt(np.einsum('nd,dk,nk->n',directions,shape,directions))[:,None]
    if frequency_mode not in ('base','observed_scale'):
        raise ValueError('Unknown frequency scaling')
    frequency_scale=float(np.median(v))
    if frequency_mode=='observed_scale':
        # Calibration drift can make base v tens of times too small. Then all
        # Fourier moments vanish and the scale is weakly identified. A robust
        # TRAIN-only observation scale changes frequencies, never observations.
        energy=np.einsum('ni,ij,nj->n',x,np.linalg.inv(shape),x)
        frequency_scale=max(frequency_scale,float(np.median(energy)/4))
    frequencies=np.array([.25,.5,1.,2.])/np.sqrt(frequency_scale)
    observation=np.exp(1j*(x@directions.T)[:,:,None]*frequencies).reshape(len(x),-1)
    def objective(eta,receipt=False):
        kernel=student_cf(np.sqrt(np.exp(eta)*v[:,None])*frequencies,df)
        kernel=np.tile(kernel,(1,len(directions)))
        denominator=np.sum(kernel*kernel,axis=0)
        coefficient=np.divide(np.sum(observation*kernel,axis=0),denominator,
            out=np.zeros(observation.shape[1],complex),where=denominator>1e-300)
        coefficient/=np.maximum(1,np.abs(coefficient))
        residual=observation-kernel*coefficient
        value=float(np.mean(np.abs(residual)**2))
        if receipt:return {'objective':value,'effect_cf_real':coefficient.real.tolist(),
            'effect_cf_imag':coefficient.imag.tolist(),'zero_kernel_count':int(np.sum(denominator<=1e-300))}
        return value
    grid=np.linspace(np.log(.001),np.log(1000),121)
    values=np.array([objective(eta) for eta in grid])
    candidates=[(values[0],grid[0]),(values[-1],grid[-1])]
    fits=[]
    for j in range(1,len(grid)-1):
        if values[j]<=values[j-1] and values[j]<=values[j+1]:
            result=minimize_scalar(objective,bounds=(grid[j-1],grid[j+1]),method='bounded',
                options={'xatol':1e-9,'maxiter':200})
            fits.append({'success':bool(result.success),'eta':float(result.x),'objective':float(result.fun)})
            candidates.append((result.fun,result.x))
    value,eta=min(candidates)
    return {'status':'DEVELOPMENT_PROFILE_OPTIMUM_NOT_IDENTIFICATION','tau':float(np.exp(eta)),
        'log_variance_spread':spread,'grid_tau':np.exp(grid).tolist(),'grid_objective':values.tolist(),
        'local_fits':fits,'directions':directions.tolist(),'frequencies':frequencies.tolist(),
        'frequency_mode':frequency_mode,'frequency_scale':frequency_scale,
        'active_boundary':eta in (grid[0],grid[-1]),**objective(eta,True)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=Path(__file__);shutil.copy2(source,args.output/'source.py')
    root=source.resolve().parents[1]
    for relative in ['src/sca3_compass/robustness_io.py','src/sca3_compass/robustness_continuous_scale.py']:
        target=args.output/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/relative,target)
    paths=sorted((root/'artifacts/robustness/continuous-floor-generator-dev1').glob('case-*-inputs.json'))
    write_json(args.output/'protocol.json',{'phase':'DEVELOPMENT_SCALE_DIAGNOSTIC',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'no_held_truth':True,'key_extra_assumption':'effect vector independent of conditional noise variance',
        'frequency_modes':['base','observed_scale'],
        'scope':'weak/no identification must be reported; Gaussian constant variance not identifiable'})
    for path in paths:
        d=json.loads(path.read_text(encoding='utf-8'));x=np.array(d['means']);v=np.array(d['base_variance'])
        shape=np.array(d['shape']);df=np.inf if d['df'] is None else d['df']
        energy=np.einsum('ni,ij,nj->n',x,np.linalg.inv(shape),x)
        c=4 if np.isinf(df) else 4*df/(df-2);b=v+np.median(v)
        design=np.column_stack([c*v/b,1/b])
        started=time.perf_counter();result=cf_fit(x,v,shape,df)
        result['observed_scale_candidate']=cf_fit(x,v,shape,df,frequency_mode='observed_scale')
        result.update(seconds=time.perf_counter()-started,
            rejected_square_moment_diagnostic={'ols_coefficients':np.linalg.lstsq(design,energy/b,rcond=None)[0].tolist(),
                'nnls_coefficients':nnls(design,energy/b)[0].tolist(),'condition_number':float(np.linalg.cond(design))})
        write_json(args.output/path.name.replace('inputs','results'),result)
        print(path.name,result['status'],result.get('tau'),
            'observed_frequency_tau',result['observed_scale_candidate'].get('tau'),flush=True)


if __name__=='__main__':main()
