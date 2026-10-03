import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_screen import data
from sca3_compass.molecular_envelope_benchmark import fixed_truth


def test_shifted_nonpositive_means_are_positive_null_but_negative_alternative():
    for mode in ['global_null','single_study_only']:
        case={'distribution':'normal','rho':.8,'n':64,'effect':8,
              'truth':mode,'null_pipeline_shift':1}
        z,x,truth=data(np.random.default_rng(16211),case)
        assert z.shape==(256,4,6) and x.shape==(4,64,6)
        assert not truth[:,0].any()
        assert truth[:,1].all()


def test_original_common_location_truth_is_unchanged():
    for mode in [None,'global_null','single_study_only']:
        case={'distribution':'t5','rho':.8,'n':64,'effect':3.5,'truth':mode}
        mu=fixed_truth(256,4,{'effect':3.5,'truth':mode,'replicated_fraction':.2},.2)
        expected=np.stack(((mu>0).sum(1)>=2,(mu<0).sum(1)>=2),axis=1)
        for heterogeneous in [False,True,'gene_random','study_random']:
            _,_,actual=data(np.random.default_rng(17241),{**case,'pipeline_heterogeneity':heterogeneous})
            np.testing.assert_array_equal(actual,expected)
