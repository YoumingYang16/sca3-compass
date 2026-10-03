import numpy as np
from scipy.integrate import quad
from scipy.stats import beta, f
from sca3_compass.robustness_weighting import radial_weight, power_weight_table, power_radial_weight


def test_weight_mean_one_under_uniform_pivot():
    for a,b in [(1,2),(2,2),(2,4),(3,5),(5,3)]:
        assert abs(quad(lambda u:.1+.9*beta.pdf(u,a,b),0,1)[0]-1)<1e-12
        u=(np.arange(10000)+.5)/10000
        q=f.ppf(u,20,5)*20*.6
        np.testing.assert_allclose(radial_weight(q,20,5,.6,a,b),.1+.9*beta.pdf(u,a,b),atol=1e-12)


def test_monotone_weight_decreases_with_residual_energy():
    weight=radial_weight(np.geomspace(.001,10000,1000),20,5,.6,1,3)
    # SciPy's beta pdf can fluctuate by ~3e-14 at a flat endpoint. This
    # tolerance tests mathematical monotonicity without demanding exact ulps.
    assert np.all(np.diff(weight)<=1e-12) and np.all(weight>=.1)


def test_power_weights_have_exact_working_null_budget():
    for nu in [2.5,5.,30.]:
        for v in [.25,.8,.96]:
            table=power_weight_table(20,nu,.6,v)
            assert np.all(table>0) and abs(table.mean()-1)<1e-14
            u=(np.arange(64)+.5)/64
            q=20*.6*f.ppf(u,20,nu)
            weight,diagnostic=power_radial_weight(q,20,nu,.6,v)
            np.testing.assert_allclose(weight,table,atol=1e-14)
            assert not diagnostic["uses_observed_means"]
