from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('r2_analyzer_fixture',SCRIPTS/'robustness_r2_analyze.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_rescore_preserves_both_old_and_new_methods():
    record=mod.read(SCRIPTS.parent/'artifacts/robustness/R0074-repetitions/case-0000/rep-000000.json.gz')
    mod.rescore(record,list(record['metrics']))
    wrong=deepcopy(record)
    name=next(iter(wrong['metrics']))
    wrong['metrics'][name]['fdp']+=.01
    with pytest.raises(ValueError,match='Independent scoring disagreement'):
        mod.rescore(wrong,list(wrong['metrics']))


def test_whole_case_paired_mcse_not_gene_count():
    x=np.array([0.,1.,0.,1.])
    y=np.array([1.,1.,1.,1.])
    result=mod.paired_summary([x,y])
    assert result['mean_difference']==.75
    assert result['paired_mcse']==pytest.approx(np.sqrt(x.var(ddof=1)/4)/2)
