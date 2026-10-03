"""Selection-aware conditional-expectation e mixture; DEVELOPMENT ONLY."""
import time
import numpy as np
from r5_kernel import evaluate as component
from r5_common import fc

VERSION='R5-A0.7.1-AUDIT-METADATA-REPAIR-DEVELOPMENT'

def evaluate(z,cs,ct,*,bound,seed,reference_draws=4095,inner_draws=4095,audit=False,method='smooth',calibration_law='radial'):
    if method.startswith('soft_'):raise ValueError('conditional components are internal, not standalone valid methods')
    if method!='smooth':return component(z,cs,ct,bound=bound,seed=seed,reference_draws=reference_draws,inner_draws=inner_draws,audit=audit,method=method,calibration_law=calibration_law)
    start=time.perf_counter()
    kw=dict(bound=bound,seed=seed,reference_draws=reference_draws,inner_draws=inner_draws,audit=audit,calibration_law=calibration_law)
    version=VERSION if calibration_law=='radial' else 'R5-A0.13-TRIANGULAR-SELECTIVE-DEVELOPMENT'
    b=component(z,cs,ct,method='soft_bridge',**kw);t=component(z,cs,ct,method='soft_target',**kw)
    if b['status']!='completed' or t['status']!='completed':
        failed=b if b['status']!='completed' else t
        return {**failed,'version':version,'seconds':time.perf_counter()-start,
            'branch_failures':{name:{'error':val.get('error'),'status':val['status'],'seconds':val['seconds'],
                'failure_receipt':val.get('failure_receipt')} for name,val in [('bridge',b),('target',t)]},
            'reference_tuples':{'branch_'+name:{k:v for k,v in val['reference_tuples'].items()
                if k not in ['es','et','base','inner_target','inner_bridge']} for name,val in [('bridge',b),('target',t)]},
            'failure_policy':'ANY_COMPONENT_FAILURE_ZEROES_WHOLE_FAMILY_NO_WEIGHT_REDISTRIBUTION'}
    rb=b['reference_tuples'];rt=t['reference_tuples'];p=rb['borrow_probability']
    if p!=rt['borrow_probability'] or not 0<=p<=1:raise ArithmeticError('inconsistent observable gate probability')
    e=p*b['e']+(1-p)*t['e'];ordinary=p*b['ordinary_e']+(1-p)*t['ordinary_e']
    arrays=['es','et','base','inner_target','inner_bridge']
    receipt={key:np.stack([rb[key],rt[key]]) for key in arrays}
    receipt.update(meta=rb['meta'],centering=rb['centering'],
        branch_bridge={k:v for k,v in rb.items() if k not in arrays},
        branch_target={k:v for k,v in rt.items() if k not in arrays},
        interpretation='two separately selective component references, data-dependent e mixing justified by integrated selection coin')
    return {'status':'completed','version':version,'calibration_law':calibration_law,'p':np.stack([b['p'],t['p']]),
        'p_kind':'TWO_COMPONENT_P_ARRAYS_NOT_A_COMBINED_P_VALUE','e':e,'ordinary_e':ordinary,
        'component_e':np.stack([b['e'],t['e']]),'decision':fc.ebh(e,.05),'ordinary_decision':fc.ebh(ordinary,.05),
        'calibration':{'method':'smooth_selective_e','borrow_probability':p,'soft_tau':rb['soft_tau'],
            'centered_observed_slack':b['calibration']['observed_slack'],'source_weight':rb['meta']['source_weight'],
            'switch_threshold':b['calibration']['switch_threshold'],'external_bound':bound},
        'folds':t['folds'],'branch_folds':{'bridge':b['folds'],'target':t['folds']},
        'reference_tuples':receipt,'profile_counts':{},'seconds':time.perf_counter()-start,
        'reference_seconds':b['reference_seconds']+t['reference_seconds']}
