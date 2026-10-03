"""Bounded actual-generator SCALE diagnostic; not a downstream FDR benchmark.

Stores exact fold inputs, all fits, source copies and failure receipts. Each
listed case uses repetition zero and outer fold zero only. No real cohort.
"""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from threadpoolctl import threadpool_limits
from robustness_screen import data
from sca3_compass.molecular_data import digest
from sca3_compass.robustness_methods import evaluate_candidates,contrasts
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass.robustness_joint_scale import fit_joint_pattern_scale
from sca3_compass.robustness_continuous_scale import fit_continuous_scale
from sca3_compass.robustness_domain import domain_variance
from sca3_compass.robustness_io import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=6604801)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=False)
    paths=[Path(__file__),root/'scripts/robustness_screen.py',*sorted((root/'src/sca3_compass').glob('*.py'))]
    hashes={str(path.relative_to(root)):digest(path) for path in paths}
    for path in paths:
        copied=args.output/'source'/path.relative_to(root)
        copied.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,copied)
        if digest(copied)!=hashes[str(path.relative_to(root))]:
            raise RuntimeError('Source changed during diagnostic snapshot')
    cases=json.loads((root/'configs/robustness_central_anchor_development.json').read_text(encoding='utf-8'))['cases']
    selected=[0,2,3,4,6,9,10]
    protocol={'phase':'DEVELOPMENT_SCALE_DIAGNOSTIC_NOT_FDR_OR_CONFIRMATION',
        'seed':args.seed,'case_indices':selected,'repetitions_per_case':1,'outer_fold':0,
        'minimum_null_weight':.5,'source_sha256':hashes,'cases':cases,
        'additional_assumption':'at least half of the working effect mixture has exactly zero effect; assumed, not measured'}
    write_json(args.output/'protocol.json',protocol)
    with threadpool_limits(limits=1):
        for index in selected:
            started=time.perf_counter()
            z,calibration,_=data(np.random.default_rng(np.random.SeedSequence([args.seed,index])),cases[index])
            _,diagnostics=evaluate_candidates(z,calibration,'frontier')
            energy_prior_candidates(z,calibration,diagnostics)
            info=diagnostics['energy_prior']['folds'][0]
            shape=np.asarray(info['study_shape']);rho=diagnostics['fit']['rho']
            means=z.mean(-1);residual=(z-means[...,None])@contrasts(z.shape[-1])
            q=np.einsum('gsk,st,gtk->g',residual,np.linalg.inv(shape),residual)/(1-rho)
            train=np.arange(len(z))%2!=0;d=info['residual_dimension'];v=info['projection_variance']
            prior=info['target_only'];df=np.inf if prior['gaussian_bic_selected'] else prior['df']+d
            variance=np.full(train.sum(),v*prior['scatter']) if np.isinf(df) else v*(prior['df']*prior['scatter']+q[train])/df
            inputs={'means':means[train].tolist(),'base_variance':variance.tolist(),'shape':shape.tolist(),
                'df':None if np.isinf(df) else df,'q':q[train].tolist(),'residual_dimension':d,'projection_variance':v}
            write_json(args.output/f'case-{index:02}-inputs.json',inputs)
            result={'case_index':index,'case':cases[index],'inputs':inputs,'fits':{}}
            functions={
                'point_joint':lambda:fit_joint_pattern_scale(means[train],variance,shape,df),
                'free_central':lambda:domain_variance(means[train],q[train],d,v,'transport_fcentral',shape=shape),
                'continuous_unconstrained':lambda:fit_continuous_scale(means[train],variance,shape,df),
                'continuous_floor_half':lambda:fit_continuous_scale(means[train],variance,shape,df,minimum_null_weight=.5)}
            for label,call in functions.items():
                stamp=time.perf_counter()
                try:
                    output=call()
                    if label=='free_central':
                        output={'variance_multiplier':output[0]/v,'receipt':output[1]}
                    result['fits'][label]={'result':output,'elapsed_seconds':time.perf_counter()-stamp}
                except (ArithmeticError,ValueError,np.linalg.LinAlgError) as error:
                    result['fits'][label]={'failed':True,'error_type':type(error).__name__,
                        'error_message':str(error),'elapsed_seconds':time.perf_counter()-stamp}
                value=result['fits'][label].get('result',{})
                print(index,cases[index]['name'],label,value.get('variance_multiplier',value.get('tau')),
                    value.get('converged',value.get('diagnostics',{}).get('converged')),flush=True)
            result['elapsed_seconds']=time.perf_counter()-started
            # Existing point-joint receipts can contain NumPy scalars/arrays.
            from sca3_compass.robustness_pattern_test import _json_value
            write_json(args.output/f'case-{index:02}-results.json',_json_value(result))


if __name__=='__main__':
    main()
