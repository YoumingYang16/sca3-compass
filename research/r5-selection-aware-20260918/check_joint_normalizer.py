"""M003 known-H inverse-normalizer correctness experiment, not observed-family Power."""
import time,shutil
import numpy as np
from scipy.special import gammaln,expit
from scipy.stats import t
from r5_common import PHASE,sha,write
from ancillary_calibration import ConditionalLogF
from selection_profile import LOG_CENTER
from check_adaptation_cost import nodes
from joint_power_reference import inverse_pair

def main():
    out=PHASE/'M003';out.mkdir(exist_ok=False)
    files=['check_joint_normalizer.py','joint_power_reference.py','ancillary_calibration.py','check_adaptation_cost.py',
           'finite_tail_diagnostic.py','r5_common.py','selection_profile.py','JOINT_MOMENT_BUDGET.md']
    write(out/'protocol.json',{'kind':'REFERENCE_NORMALIZER_VERIFICATION_DIAGNOSTIC_ONLY','repeats':64,
        'seed':202609181920,'L':16,'numerator_draws':1024,'cap_per_endpoint':1000000,
        'fixture':'Ns=Nt=4,W=0,uncenteredY,a=.5,c=1,tau=.5,k=1,alpha4,IDENTITY_SHAPE_ONLY',
        'reason':'Analytic t20moment and deterministic logF integrals provide an independently checkable reciprocal expectation target',
        'stop':'64 fixed MCnormalizer repetitions, no new patient/simulationfamilies or confirmation',
        'precision':'MonteCarlo mean ratio near1, reportSE not significance-based extension; no test can replace analytic unbiasedness'})
    for f in files:shutil.copy2(PHASE/f,out/f)
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in files+['protocol.json']}})
    law=ConditionalLogF(np.zeros(4));x,w,_=nodes(law,1024,0.)
    u=x[:,None]-x[None,:];gate=expit((1-u)/.5)
    raw=np.exp(-2*x[None,:])*((1-gate)+np.exp(-u)*gate)
    mv=20**2*3/(2*(20-2)*(20-4))
    m0=float(mv*(w@raw@w));mt=float(np.exp(2*np.log(2)+gammaln(8-2)+gammaln(4+2)-gammaln(8)-gammaln(4)))
    mi=mv*mt;rows=[];start=time.perf_counter()
    for rep,ss in enumerate(np.random.SeedSequence(202609181920).spawn(64)):
        seed=int(ss.generate_state(1,dtype=np.uint64)[0])
        try:
            logR,rec=inverse_pair(np.zeros(4),np.zeros(4),-LOG_CENTER,-LOG_CENTER,.5,1.,.5,0.,16,seed,
                successes=16,numerator_draws=1024,cap=1000000,identity_shape=True)
            rr={'rep':rep,'seed':seed,'status':'completed','R0_times_true_M0':float(np.exp(rec['log_inverse_boundary'])*m0),
                'Rinf_times_true_Minf':float(np.exp(rec['log_inverse_infinity'])*mi),
                'minR_times_true_maxM':float(np.exp(logR)*max(m0,mi)),'receipt':rec}
        except (ArithmeticError,RuntimeError,np.linalg.LinAlgError) as err:
            rr={'rep':rep,'seed':seed,'status':'failed_closed','error':repr(err),'failure_receipt':getattr(err,'receipt',None),
                'R0_times_true_M0':0.,'Rinf_times_true_Minf':0.,'minR_times_true_maxM':0.}
        write(out/f'rep-{rep:03}.json',rr);rows.append(rr)
        if (rep+1)%8==0:print(rep+1,flush=True)
    sums={key:{'mean':float(np.mean([r[key] for r in rows])),
                'SE_descriptive':float(np.std([r[key] for r in rows],ddof=1)/8)}
          for key in ['R0_times_true_M0','Rinf_times_true_Minf','minR_times_true_maxM']}
    write(out/'result.json',{'status':'REFERENCE_DIAGNOSIS_COMPLETE_NOT_FDR_CONFIRMATION','summary':sums,
        'fixture_M0':m0,'fixture_Minf':mi,'failures':sum(r['status']!='completed' for r in rows),
        'seconds':time.perf_counter()-start,'raw_files':[f'rep-{r["rep"]:03}.json' for r in rows],
        'actual_shape_law':'NOT_USED_IDENTITY_FIXTURE_ONLY','ordinary_family_count_added':0})
    print(sums,flush=True)

if __name__=='__main__':main()
