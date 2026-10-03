"""Fixed bridge observed scale. No learned drift radius or oracle inputs."""
import numpy as np
from r4_common import model,pb

def scale(cs,ct,weight,bound):
    if not np.isfinite(bound) or bound<1: raise ValueError('external D>=1 required')
    if weight==0:
        kg=model.geometric_kappa(cs);km=pb.kappa_estimator(cs)
        k=kg*bound;receipt={'source_geometric':kg,'target_geometric':None}
    elif weight==1:
        kg=model.geometric_kappa(ct);km=pb.kappa_estimator(ct)
        k=kg;receipt={'source_geometric':None,'target_geometric':kg}
    elif weight==len(ct)/(len(cs)+len(ct)):
        ks=model.geometric_kappa(cs);kt=model.geometric_kappa(ct)
        k=float(np.exp((1-weight)*(np.log(bound)+np.log(ks))+weight*np.log(kt)))
        km=pb.kappa_estimator(ct)  # SAME target DIR learner as target-only
        receipt={'source_geometric':ks,'target_geometric':kt}
    else: raise ValueError('unregistered bridge weight')
    if not np.isfinite(k) or k<=0: raise ArithmeticError('invalid observed bridge scale')
    return k,km,{**receipt,'inference_scale':k,'learning_median':km,
                'target_weight':weight,'external_bound':bound}
