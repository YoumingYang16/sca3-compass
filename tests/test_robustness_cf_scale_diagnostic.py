import sys
from pathlib import Path
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_cf_scale_diagnostic import student_cf,cf_fit


@pytest.mark.parametrize('df',[3.,5.,21.5,25.,200.])
def test_cf_matches_independent_fourier_integral(df):
    for frequency in [0.,1e-6,.1,.5,2.]:
        # Fourier-cycle integration at w=1e-6 samples intervals millions wide
        # and can miss the entire density near zero. Use ordinary adaptive
        # integration for low frequency; keep oscillatory quadrature elsewhere.
        expected=(2*quad(lambda x:t.pdf(x,df)*np.cos(frequency*x),0,np.inf,
            epsabs=1e-10,limit=200)[0] if frequency<.01 else
            2*quad(lambda x:t.pdf(x,df),0,np.inf,weight='cos',wvar=frequency,
            epsabs=1e-10,limit=200)[0])
        np.testing.assert_allclose(student_cf(frequency,df),expected,atol=2e-9,rtol=1e-8)


def test_constant_noise_scale_is_explicitly_unidentified():
    result=cf_fit(np.ones((30,4)),np.ones(30),np.eye(4),25.)
    assert result['status']=='NO_VARIANCE_CONTRAST_NOT_IDENTIFIED' and 'tau' not in result


def test_gaussian_cf_and_evenness():
    z=np.array([0.,.1,1.,10.])
    np.testing.assert_array_equal(student_cf(z,np.inf),np.exp(-z*z/2))
    np.testing.assert_array_equal(student_cf(z,25),student_cf(-z,25))
