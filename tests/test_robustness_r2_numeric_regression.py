from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_r2_numeric_regression import repaired_aliases


@pytest.mark.parametrize('raises',[False,True])
def test_all_three_local_aliases_patched_and_restored_even_on_error(raises):
    old=[lambda x:x,lambda x:x+1,lambda x:x+2]
    modules=[SimpleNamespace(fit_calibration=f) for f in old]
    replacement=lambda x:0
    def run():
        with repaired_aliases(modules,replacement):
            assert all(m.fit_calibration is replacement for m in modules)
            if raises:raise RuntimeError('fixture')
    if raises:
        with pytest.raises(RuntimeError):run()
    else:run()
    assert all(m.fit_calibration is f for m,f in zip(modules,old))
