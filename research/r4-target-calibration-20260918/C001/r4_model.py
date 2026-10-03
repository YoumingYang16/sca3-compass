"""R4 observed-data adapter. No simulator truth/parameter inputs."""
import numpy as np
from r4_common import model, verify_runtime
VERSION='R4-TCB-0.2.0'

def bank(value, name):
    a=np.asarray(value,float)
    if a.ndim!=3 or a.shape[1:]!=(4,6) or len(a)<4 or not np.isfinite(a).all():
        raise ValueError(name+': finite independent centered N>=4 blocks required')
    return a

def evaluate(z, calibration_source, calibration_target=None, *, seed,
             reference_draws=4095, mode, external_drift_bound=None):
    """Current target-only and source endpoints call the EXACT frozen R3."""
    verify_runtime()
    cs=bank(calibration_source,'source')
    if mode=='bridge_count':
        from r4_kernel import evaluate_kernel
        if calibration_target is None or external_drift_bound is None:
            raise ValueError('bridge requires independent target bank and external bound')
        ct=bank(calibration_target,'target')
        result=evaluate_kernel(z,cs,ct,seed=seed,reference_draws=reference_draws,
            weight=len(ct)/(len(cs)+len(ct)),bound=float(external_drift_bound))
        result['r4']={'version':VERSION,'mode':mode,'source_blocks':len(cs),'target_blocks':len(ct),
            'reference_calibration_blocks':len(cs)+len(ct),'external_bound':external_drift_bound,
            'grade':'MODEL_EXACT_ARITHMETIC_MARGINAL_OVER_BANKS_AND_REFERENCES',
            'availability':'CONDITIONAL_TARGET_CALIBRATION_RESEARCH','real_data_certified':False,
            'assumptions':['independent zero-location banks','target kappa matches Ct',
                           'kappa_target/kappa_source<=external_bound','original R3 M0 otherwise']}
        return result
    if mode=='target_only':
        if calibration_target is None:
            raise ValueError('target_only unavailable without target calibration')
        if external_drift_bound is not None:
            raise ValueError('target_only does not use an external bound')
        used=bank(calibration_target,'target'); bound=1.
    elif mode in ['source_matched','source_bound']:
        used=cs
        if mode=='source_bound':
            if external_drift_bound is None:
                raise ValueError('external bound required, never inferred from data')
            bound=float(external_drift_bound)
        else:
            if external_drift_bound is not None: raise ValueError('wrong mode for bound')
            bound=1.
    else: raise ValueError('unknown explicit mode')
    result=model.evaluate(z,used,seed=seed,reference_draws=reference_draws,mismatch_bound=bound)
    result['r4']={'version':VERSION,'mode':mode,'source_blocks':len(cs),
        'target_blocks':0 if calibration_target is None else len(calibration_target),
        'reference_calibration_blocks':len(used),'external_bound':external_drift_bound,
        'grade':'MODEL_EXACT_ARITHMETIC_MARGINAL_OVER_BANKS_AND_REFERENCES',
        'availability':'CONDITIONAL_TARGET_CALIBRATION_RESEARCH' if mode=='target_only' else 'CURRENT_INPUT_ONLY',
        'real_data_certified':False}
    return result

def load_observed(path):
    """Strict array allowlist: never forward raw simulation archives."""
    with np.load(path,allow_pickle=False) as a:
        if not {'z','calibration_source'}<=set(a.files) or set(a.files)-{'z','calibration_source','calibration_target'}:
            raise ValueError('only observed z/source/optional target arrays allowed')
        return {k:a[k] for k in a.files}
