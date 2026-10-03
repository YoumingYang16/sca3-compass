"""Training-only discovery-budget calibration; classical p-to-e mixtures.

Training PC probabilities select a threshold, never held probabilities.
They need not be valid for this selection argument. Conditional validity
of the HELD p-value given all training information is essential, however.
Estimated nuisances/cross-gene dependence are not covered by that premise.
The pilot is BH at .05 in the training signed family. Half its rejection
proportion times .05 gives the concentrated threshold. Zero pilot count
falls back to the existing capped calibrator. No post-hoc best threshold.
"""
import numpy as np
from .molecular_methods import checked_probabilities
from .robustness_calibrators import focused_calibrator


def pilot_calibrator(held_p, training_p, family, fraction=.5, alpha=.05):
    """Integral-one decreasing calibrator fixed by OTHER genes' pilot.

    Each nonzero pilot uses (1-f)*capped_BY + f*I(p<=tau)/tau,
    tau=.5*alpha*Rpilot/m_train. Count is pooled over BOTH directions;
    it is a density estimate, not an assertion that those genes are true.
    """
    held=checked_probabilities(held_p)
    training=checked_probabilities(training_p).reshape(-1)
    if not training.size or not isinstance(family,int) or family<2:
        raise ValueError('Nonempty training probabilities and family>=2 required')
    if not 0<alpha<1 or not 0<=fraction<=1:
        raise ValueError('Invalid alpha or mixture fraction')
    ordered=np.sort(training)
    passing=np.flatnonzero(ordered<=alpha*np.arange(1,len(training)+1)/len(training))
    count=int(passing[-1]+1) if len(passing) else 0
    threshold=.5*alpha*max(1,count)/len(training)
    selected_fraction=fraction if count else 0.
    values=focused_calibrator(held,family,threshold,selected_fraction,alpha,cap=max(1,family//8))
    return values,{'training_signed_count':len(training),'pilot_discoveries':count,
        'pilot_level':alpha,'threshold':threshold,'concentration_fraction':selected_fraction,
        'zero_count_fallback':count==0,'threshold_rule':'.5*alpha*Rpilot/training_family',
        'uses_held_probabilities_for_selection':False,'uses_truth_labels':False}
