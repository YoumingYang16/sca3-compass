"""Replay all six saved cancellation diagnostics, preserving the old failures."""
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from sca3_compass.molecular_data import PROJECT_ROOT,digest
from sca3_compass.robustness_io import write_json
from sca3_compass.robustness_joint_scale import fit_joint_pattern_scale
from sca3_compass.robustness_pattern_test import _json_value


def main():
    source=PROJECT_ROOT/'artifacts/robustness/joint-r48-case9-cancellation-diagnostic.json'
    output=PROJECT_ROOT/'artifacts/robustness/joint-r48-cancellation-recovery-check.json'
    if output.exists():
        raise SystemExit('Existing recovery receipt must be preserved')
    original=json.loads(source.read_text(encoding='utf-8'))
    rows=[]
    with threadpool_limits(limits=1):
        for case in original['cases']:
            fit=fit_joint_pattern_scale(np.array(case['training_means']),np.array(case['base_variance']),
                np.array(case['study_shape']),case['df'])
            info=fit['diagnostics'];old=case['diagnostics']
            assert info['converged'] and info['numerical_failure_count']==0
            assert all(s['converged'] for s in info['starts'])
            assert abs(info['objective']-old['objective'])<1e-8
            rows.append({'repetition':case['repetition'],'fold':case['fold'],'old_numerical_events':old['numerical_failure_count'],
                'new_numerical_events':info['numerical_failure_count'],'objective_difference':info['objective']-old['objective'],
                'tau_difference':fit['tau']-old['tau'],'diagnostics':_json_value(info)})
    write_json(output,{'phase':'NUMERICAL_REPLAY_NOT_CONFIRMATION','input_sha256':digest(source),'results':rows,
        'source_sha256':{str(p.relative_to(PROJECT_ROOT)):digest(p) for p in [Path(__file__),PROJECT_ROOT/'src/sca3_compass/robustness_joint_scale.py']}})
    print('All six saved problems / seven failing starts recovered; objective differences <1e-8')


if __name__=='__main__':
    main()
