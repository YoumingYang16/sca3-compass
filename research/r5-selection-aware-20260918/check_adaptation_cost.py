"""M002 deterministic conditional exponential-tilt/order check. Not FDR data."""
import gzip,json,shutil
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_legendre,expit
from ancillary_calibration import ConditionalLogF
from finite_tail_diagnostic import limits
from selection_profile import LOG_CENTER
from r5_common import PHASE,sha,write

class Tilted:
    def __init__(self,base,r):
        if not 0<=r<2*base.n:raise ValueError('tilt moment outside domain')
        self.base=base;self.r=r;lo=base.mode-1;hi=base.mode+1
        while self.derivative(lo)<=0:lo=base.mode-2*(base.mode-lo)
        while self.derivative(hi)>=0:hi=base.mode+2*(hi-base.mode)
        self.mode=float(brentq(self.derivative,lo,hi));self.shift=float(self.log_density(self.mode))
    def log_density(self,y):return self.base.log_density(y)-self.r*np.asarray(y)
    def derivative(self,y):return self.base.derivative(y)-self.r

def nodes(law,n,center):
    lo,hi,norm,receipt=limits(law);x,w=roots_legendre(n);x=(lo+hi)/2+x*(hi-lo)/2
    w=w*(hi-lo)/2*np.exp(law.log_density(x)-law.shift)/norm;mass=float(w.sum());w=w/mass
    return x-center,w,{'limits':receipt,'mass_before_normalization':mass}

def prob(s,sx,sw,tx,tw,c,tau):
    h=expit((sx[:,None]-tx[None,:]+s-c)/tau)
    return float(sw@h@tw)

def calculate(ls,lt,ms,mt,c,tau,n):
    sx,sw,srec=nodes(ls,n,LOG_CENTER+ms);tx,tw,trec=nodes(lt,n,LOG_CENTER+mt)
    rows=[]
    for r in [.5,1.,2.]:
        tilt=Tilted(lt,r);rx,rw,rrec=nodes(tilt,n,LOG_CENTER+mt)
        delta=4*np.arctanh(r/(3*lt.n))
        for s in [0.,.5,1.,float(np.log(5))]:
            pi=prob(s,sx,sw,tx,tw,c,tau)
            pi_shift=prob(s+delta,sx,sw,tx,tw,c,tau)
            tilted_pi=prob(s,sx,sw,rx,rw,c,tau)
            rows.append({'r':r,'alpha':2*r,'slack':s,'delta':float(delta),'pi_target':pi,
                'moment_inflation_exact_tilt_quadrature':tilted_pi/pi,'lower_bound_quadrature':pi_shift/pi,
                'gap':(tilted_pi-pi_shift)/pi,'tilted_quadrature':rrec})
    return {'rows':rows,'source_quadrature':srec,'target_quadrature':trec}

def main():
    out=PHASE/'M002';out.mkdir(exist_ok=False)
    files=['check_adaptation_cost.py','finite_tail_diagnostic.py','ancillary_calibration.py',
           'selection_profile.py','r5_common.py','ADAPTATION_MOMENT_COST.md']
    write(out/'protocol.json',{'kind':'DETERMINISTIC_MATH_DIAGNOSIS','cases':list(range(10)),'rep':8,
        'r':[.5,1,2],'s':[0,.5,1,float(np.log(5))],'nodes':[512,1024],
        'stop':'fixed120conditional evaluations eachresolution; no samples/FDR/confirmation',
        'precision':'report differences; target1e-6 absolute ratio refinement, no automatic extension',
        'source':'D011 savedW/meta only','theory_sha':sha(PHASE/'ADAPTATION_MOMENT_COST.md')})
    for f in files:shutil.copy2(PHASE/f,out/f)
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in files+['protocol.json']}})
    rows=[]
    for case in range(10):
        path=PHASE/f'D011/raw/case-{case:02}/rep-00008.json.gz'
        record=json.load(gzip.open(path,'rt',encoding='utf8'));b=record['reference_receipt']['branch_target']
        ls=ConditionalLogF(b['ancillary_source']);lt=ConditionalLogF(b['ancillary_target'])
        ms,mt=b['centering'];c=b['selection']['switch_threshold'];tau=b['soft_tau']
        coarse=calculate(ls,lt,ms,mt,c,tau,512);fine=calculate(ls,lt,ms,mt,c,tau,1024)
        change=max(abs(a['moment_inflation_exact_tilt_quadrature']-b['moment_inflation_exact_tilt_quadrature']) for a,b in zip(coarse['rows'],fine['rows']))
        row={'case':case,'rep':8,'source_sha':sha(path),'Ns':ls.n,'Nt':lt.n,'coarse':coarse,'fine':fine,'max_ratio_refinement_change':change}
        write(out/f'case-{case:02}.json',row);rows.append(row)
        print(case,[(r['alpha'],r['moment_inflation_exact_tilt_quadrature'],r['lower_bound_quadrature']) for r in fine['rows'] if r['slack']==0],flush=True)
    write(out/'result.json',{'status':'DETERMINISTIC_CHECK_COMPLETE_NOT_PROOF_OR_G1_ACCEPTANCE','rows':rows,
        'minimum_numerical_inequality_gap':min(r['gap'] for v in rows for r in v['fine']['rows']),
        'no_new_observed_families':True,'formal_confirmation_count':0,'EXTERNAL_REVIEW':'NOT_CONDUCTED'})

if __name__=='__main__':main()
