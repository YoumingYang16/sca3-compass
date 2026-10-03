"""Three-action integrated selective e evidence, development and not accepted."""
import time
import numpy as np
from r5_common import fc
from r5_kernel import evaluate as component
VERSION='R5-A0.14-ACTION-SELECTIVE-DEVELOPMENT'


def evaluate(z,cs,ct,*,bound,seed,reference_draws=4095,inner_draws=4095,audit=False):
    started=time.perf_counter()
    branches=[component(z,cs,ct,bound=bound,seed=seed,reference_draws=reference_draws,
        inner_draws=inner_draws,audit=audit,method='action_component',calibration_law='triangular',action_index=j) for j in range(3)]
    failures=[b for b in branches if b['status']!='completed']
    if failures:return {**failures[0],'version':VERSION,'seconds':time.perf_counter()-started,
        'branch_statuses':[{'status':b['status'],'error':b.get('error'),'failure_receipt':b.get('failure_receipt')} for b in branches],
        'failure_policy':'ANY_COMPONENT_FAILURE_ZEROES_WHOLE_FAMILY'}
    rt=[b['reference_tuples'] for b in branches];weights=rt[0]['action_probabilities']
    if any(not np.array_equal(weights,r['action_probabilities']) for r in rt):raise ArithmeticError('inconsistent action weights')
    ce=np.stack([b['e'] for b in branches]);oe=np.stack([b['ordinary_e'] for b in branches])
    e=np.einsum('j,jgd->gd',weights,ce);ordinary=np.einsum('j,jgd->gd',weights,oe)
    return {'status':'completed','version':VERSION,'p':np.stack([b['p'] for b in branches]),
        'p_kind':'THREE_COMPONENT_ARRAYS_NOT_COMBINED_P','e':e,'ordinary_e':ordinary,'component_e':ce,
        'decision':fc.ebh(e,.05),'ordinary_decision':fc.ebh(ordinary,.05),
        'calibration':{'action_probabilities':weights,'action_weights':rt[0]['action_weights'],
            'observed_slack':branches[0]['calibration']['observed_slack'],'meta':rt[0]['meta']},
        'folds':branches[0]['folds'],'branch_folds':[b['folds'] for b in branches],
        'reference_tuples':{'branches':rt},'reference_seconds':sum(b['reference_seconds'] for b in branches),
        'seconds':time.perf_counter()-started}
