from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('r1_distribution_fixture', SCRIPTS/'robustness_r1_distributions.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_empirical_family_quantiles_not_gene_binomial_intervals():
    result = module.distribution([0., 0., 1.])
    assert result['mean'] == pytest.approx(1/3)
    assert result['sd'] == pytest.approx(np.sqrt(1/3))
    assert result['quantiles']['0.5'] == 0
    assert result['quantiles']['0.99'] == 1
    assert 'lower' not in result and 'upper' not in result


def test_deterministic_cost_or_metric_vector():
    result = module.distribution([2., 2., 2.])
    assert result['mean'] == result['min'] == result['max'] == 2.
    assert result['sd'] == 0.
    assert set(result['quantiles'].values()) == {2.}


@pytest.mark.parametrize('bad', [[], [np.nan], [np.inf], [[1., 2.]]])
def test_reject_incomplete_numerical_vectors(bad):
    with pytest.raises(ValueError): module.distribution(bad)
