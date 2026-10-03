"""Capture/recheck the four actual R0055 fixed-pattern stationarity failures.

This is numerical regression evidence, NOT a fresh scientific experiment.
The shrink branch has the same inputs and failures in these cases.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from robustness_screen import data
from robustness_replay import read_json,write_new,sha
from sca3_compass.robustness_methods import contrasts
from sca3_compass.robustness_patterns import fit_pattern_mixture
from sca3_compass.robustness_pattern_test import _json_value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['capture','recover'])
    args=parser.parse_args()
    folder=Path('artifacts/robustness')
    fixture=folder/'pattern-R0055-stationarity-inputs.json'
    target=fixture if args.mode=='capture' else folder/'pattern-R0055-newton-recovery.json'
    if target.exists():
        raise FileExistsError('Preserve numerical diagnostic evidence')
    with threadpool_limits(limits=1):
        if args.mode=='capture':
            protocol=read_json(folder/'R0055-screen.protocol.json')
            raw=read_json(folder/'R0055-cases/case-0007.json.gz')['result']
            failures={i:[f['fold'] for f in d['pattern_testing']['folds']
                if not f['fit']['diagnostics']['converged']] for i,d in enumerate(raw['diagnostics'])}
            failures={i:folds for i,folds in failures.items() if folds}
            rng=np.random.default_rng(np.random.SeedSequence([protocol['seed'],7]))
            records=[]
            for rep in range(max(failures)+1):
                z,_,_=data(rng,protocol['cases'][7])
                if rep not in failures:
                    continue
                diag=raw['diagnostics'][rep]
                for fold in failures[rep]:
                    info=diag['energy_prior']['folds'][fold]
                    shape=np.array(info['study_shape']); prior=info['target_only']
                    means=z.mean(-1); y=(z-means[...,None])@contrasts(z.shape[-1])
                    q=np.einsum('gsk,st,gtk->g',y,np.linalg.inv(shape),y)/(1-diag['fit']['rho'])
                    df=np.inf if prior['gaussian_bic_selected'] else prior['df']+20
                    v=info['projection_variance']
                    variance=np.full(len(z),v*prior['scatter']) if np.isinf(df) else v*(prior['df']*prior['scatter']+q)/df
                    train=np.arange(len(z))%2!=fold
                    fit=fit_pattern_mixture(means[train],variance[train],shape,df)
                    archived=diag['pattern_testing']['folds'][fold]['fit']
                    if not np.array_equal(fit['weights'],archived['weights']):
                        raise ValueError('Recreated weights do not exactly match the archived failure')
                    records.append({'repetition':rep,'fold':fold,'training_means':means[train],
                        'variance':variance[train],'shape':shape,'df':None if np.isinf(df) else df,
                        'before':fit['diagnostics'],'before_weights':fit['weights']})
        else:
            records=[]
            for old in read_json(fixture)['records']:
                fit=fit_pattern_mixture(old['training_means'],old['variance'],old['shape'],
                    np.inf if old['df'] is None else old['df'])
                records.append({'repetition':old['repetition'],'fold':old['fold'],
                    'before':old['before'],'after':fit['diagnostics'],
                    'objective_difference':fit['diagnostics']['objective']-old['before']['objective'],
                    'maximum_weight_difference':float(np.max(np.abs(fit['weights']-old['before_weights'])))})
    write_new(target,_json_value({'phase':'NUMERICAL_REGRESSION_NOT_CONFIRMATION','mode':args.mode,
        'records':records,'source_sha256':{str(p):sha(p) for p in
            [Path(__file__),Path('src/sca3_compass/robustness_patterns.py'),Path('scripts/robustness_screen.py')]}}))
    print(json.dumps({'output':str(target),'count':len(records)}))


if __name__=='__main__':
    main()
