import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_analyze_checkpoints import diagnostic_view,_DROP
from sca3_compass.robustness_analysis import diagnostic_counts


def test_projection_preserves_all_diagnostic_events_and_paths():
    original=[{'fit':{'converged':False,'objective':12.,'history':[1,2,3]},
        'folds':[{'converged':True,'values':[.1]*100,'failure':None,
                  'starts':[{'converged':False,'numerical_failure_count':2},
                            {'converged':True,'numerical_failure_count':0}],
                  'not_run':True,'nonfinite_values':0}],
        'unclassified':[float('nan'),None,1,'text'],
        'numerical_failure':True}, {'fit':{'converged':True}}]
    view=[diagnostic_view(d) for d in original]
    assert diagnostic_counts(view)==diagnostic_counts(original)
    assert 'history' not in view[0]['fit']
    assert 'values' not in view[0]['folds'][0]


def test_empty_and_nested_null_projection():
    assert diagnostic_view({'weights':np.arange(10).tolist()}) is _DROP
    original=[{'a':[{'b':None},{'b':1},[None,None]],'c':{'not_run':False}}]
    assert diagnostic_counts(original)==diagnostic_counts([diagnostic_view(d) for d in original])
