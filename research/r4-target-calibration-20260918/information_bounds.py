"""Deterministic information calculations. Never imported by inference."""
from math import cosh,log,ceil,sqrt,pi
from scipy.integrate import quad
from r4_common import PHASE,write

def affinity(h): return cosh(h/4)**-2
def error_sum_lower(h,nt): return 1-sqrt(max(0,1-affinity(h)**(2*nt)))
def necessary_n(h,alpha=.05): return ceil(log(4*alpha*(1-alpha))/(4*log(1/cosh(h/4))))

def report():
    rows=[]
    for ratio in [1.25,1.5,2.,3.,5.]:
        h=log(ratio);r=1/ratio
        val,err=quad(lambda u:2*r*u/((1+u)*(1+r*u))**1.5,0,float('inf'),epsabs=1e-12)
        assert abs(val-affinity(h))<2e-11
        rows.append({'two_point_scale_ratio':ratio,'log_separation':h,'affinity':affinity(h),
            'quadrature':val,'quadrature_error_estimate':err,'necessary_Nt_95pct_fixed_diameter_lt_h':necessary_n(h),
            'two_error_sum_lower':{str(n):error_sum_lower(h,n) for n in [4,12,32,128]}})
    write(PHASE/'checks/information-bounds.json',{'scope':'PIVOT_ONLY_NOT_PC_POWER_BOUND',
        'rows':rows,'geometric_log_variance_constant':pi*pi/3-1,
        'derivation':'R4_THEORY.md T4/T5; exact analytical formula, float quadrature check only'})
    print(rows)

if __name__=='__main__': report()
