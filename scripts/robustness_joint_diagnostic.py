"""Reproduce cancellation-sensitive R0048 MAP monotonicity failures."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from robustness_screen import data
from sca3_compass.molecular_data import PROJECT_ROOT,digest
from sca3_compass.robustness_io import write_json
from sca3_compass.robustness_methods import evaluate_candidates,contrasts
from sca3_compass.robustness_prior import energy_prior_candidates
from sca3_compass import robustness_joint_scale as joint
from sca3_compass.robustness_pattern_test import _json_value


def main():
    target=PROJECT_ROOT/'artifacts/robustness/joint-r48-case9-cancellation-diagnostic.json'
    if target.exists():
        raise SystemExit('Existing diagnostic must be preserved')
    seeds=[12,31,35,49,58,87]
    cases=[]
    original=joint._em_step
    trace=[]
    def audit_step(problem,weights,log_tau,state,bounds):
        new_weights,new_log_tau=original(problem,weights,log_tau,state,bounds)
        check=problem.state(new_weights,new_log_tau)
        old_tolerance=32*np.finfo(float).eps*max(1.,abs(state['objective']))
        if check['objective']<state['objective']-old_tolerance:
            trace.append({'old_objective':state['objective'],'new_objective':check['objective'],
                'decrease':state['objective']-check['objective'],'old_tolerance':old_tolerance,
                'old_kkt':state['weight_kkt_residual'],'new_kkt':check['weight_kkt_residual'],
                'constant_absolute_sum':float(np.abs(problem.constant).sum())})
        return new_weights,new_log_tau
    joint._em_step=audit_step
    rng=np.random.default_rng(np.random.SeedSequence([4803473,9]))
    with threadpool_limits(limits=1):
        for rep in range(max(seeds)+1):
            z,x,_=data(rng,{'distribution':'lognormal','rho':.1,'n':64,'effect':3.5})
            if rep not in seeds:
                continue
            _,diag=evaluate_candidates(z,x,'frontier')
            energy_prior_candidates(z,x,diag)
            for fold in ([1] if rep in [12,31] else [0]):
                info=diag['energy_prior']['folds'][fold]
                shape=np.asarray(info['study_shape']);p=info['target_only']
                y=(z-z.mean(-1)[...,None])@contrasts(6)
                q=np.einsum('gsk,st,gtk->g',y,np.linalg.inv(shape),y)/(1-diag['fit']['rho'])
                df=np.inf if p['gaussian_bic_selected'] else p['df']+20
                v=np.full(len(z),info['projection_variance']*p['scatter']) if np.isinf(df) else info['projection_variance']*(p['df']*p['scatter']+q)/df
                train=np.arange(len(z))%2!=fold
                trace.clear()
                fit=joint.fit_joint_pattern_scale(z.mean(-1)[train],v[train],shape,df)
                cases.append({'repetition':rep,'fold':fold,'training_means':z.mean(-1)[train].tolist(),
                    'base_variance':v[train].tolist(),'study_shape':shape.tolist(),'df':df,
                    'diagnostics':_json_value(fit['diagnostics']),'decrease_trace':list(trace)})
    joint._em_step=original
    write_json(target,{'phase':'NUMERICAL_DEVELOPMENT_DIAGNOSTIC','source_run':'R0048','cases':cases,
        'source_sha256':{str(p.relative_to(PROJECT_ROOT)):digest(p) for p in
            [Path(__file__),PROJECT_ROOT/'src/sca3_compass/robustness_joint_scale.py',PROJECT_ROOT/'src/sca3_compass/robustness_prior.py']}})
    print(json.dumps([{'rep':c['repetition'],'trace':c['decrease_trace']} for c in cases]))


if __name__=='__main__':
    main()
