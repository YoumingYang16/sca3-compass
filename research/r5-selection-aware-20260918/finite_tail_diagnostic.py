"""D012: deterministic quadrature on ten prespecified saved reference libraries."""
import gzip,json
import numpy as np
from scipy.integrate import quad,cumulative_trapezoid
from scipy.special import expit,roots_legendre
from ancillary_calibration import ConditionalLogF
from selection_profile import LOG_CENTER
from r5_common import PHASE,sha,read,write

def limits(law):
    lo=law.mode-1.;hi=law.mode+1.
    for _ in range(100):
        dl=float(law.derivative(lo));dr=float(law.derivative(hi))
        left=np.inf if dl<=0 else np.exp(law.log_density(lo)-law.shift)/dl
        right=np.inf if dr>=0 else np.exp(law.log_density(hi)-law.shift)/(-dr)
        if max(left,right)<1e-14:break
        if left>=1e-14:lo=law.mode-2*(law.mode-lo)
        if right>=1e-14:hi=law.mode+2*(hi-law.mode)
    else:raise ArithmeticError('cannot delimit quadrature')
    norm,err=quad(lambda x:float(np.exp(law.log_density(x)-law.shift)),lo,hi,epsabs=1e-12,epsrel=1e-12,limit=200)
    return lo,hi,norm,{'omitted_mass_tangent_bound_float':float((left+right)/norm),
        'normalizer_quad_error':err,'interval':[lo,hi],'not_interval_certificate':True}

def calculate(ls,lt,ms,mt,c,tau,base,nsource,ntarget):
    sl,sh,sn,sr=limits(ls);tl,th,tn,tr=limits(lt)
    zz,ww=roots_legendre(nsource);sy=(sl+sh)/2+zz*(sh-sl)/2
    sw=ww*(sh-sl)/2*np.exp(ls.log_density(sy)-ls.shift)/sn
    source_mass=float(sw.sum());sw=sw/source_mass
    sy=sy-LOG_CENTER-ms
    ty=np.linspace(tl,th,ntarget);td=np.exp(lt.log_density(ty)-lt.shift)/tn
    y=ty-LOG_CENTER-mt;h=np.empty(ntarget)
    for start in range(0,ntarget,256):
        yy=y[start:start+256]
        h[start:start+256]=expit((sy[None,:]-yy[:,None]-c)/tau)@sw
    cdf=cumulative_trapezoid(td,y,initial=0);weighted=cumulative_trapezoid(td*h,y,initial=0)
    target_mass=float(cdf[-1]);pi=float(weighted[-1]/target_mass)
    cdf=cdf/target_mass;selcdf=weighted/weighted[-1]
    if not 0<pi<1:raise ArithmeticError('invalid branch mass')
    out=[]
    for x in [2.,4.,8.,16.]:
        cutoff=np.full(len(base),-np.inf);positive=base>0;cutoff[positive]=2*np.log(base[positive]/x)
        v0=np.interp(cutoff,y,cdf,left=0,right=1);vt=np.interp(cutoff,y,selcdf,left=0,right=1)
        p0=float(v0.mean());pt=float(vt.mean());resolved=p0>100*tr['omitted_mass_tangent_bound_float']
        ratio=pt/p0 if resolved else None
        out.append({'threshold':x,'target_tail':p0,'selected_target_tail':pt,
                    'ratio':ratio,'universal_ratio_upper':1/pi,
                    'target_tail_MCSE_descriptive':float(v0.std(ddof=1)/np.sqrt(len(v0))),
                    'ratio_MCSE_delta_descriptive':float((vt-ratio*v0).std(ddof=1)/np.sqrt(len(v0))/p0) if resolved else None,
                    'tail_effective_V_count':float(v0.sum()**2/(v0@v0)) if v0@v0>0 else 0.,
                    'quadrature_resolved_not_population_certification':bool(resolved)})
    return {'pi_target':pi,'source_mass_numerical_before_normalization':source_mass,'target_mass_numerical_before_normalization':target_mass,
        'quadrature_source':sr,'quadrature_target':tr,'rows':out,
        'notes':'Finite V-library average, not a new estimate using fresh independent V; no Power inference'}

def main():
    out=PHASE/'D012_2';out.mkdir(exist_ok=False)
    write(out/'protocol.json',{'kind':'DETERMINISTIC_SAVED_REFERENCE_DIAGNOSIS','cases':list(range(10)),
        'rep':8,'source':'D011','thresholds':[2,4,8,16],'quadrature_pairs':[[512,4097],[1024,8193]],
        'stop':'exactly10earliestinputs; no new random draws/fitting/confirmation',
        'theory_sha':sha(PHASE/'FINITE_SELECTION_COST.md'),'script_sha':sha(__file__)})
    rows=[]
    for case in range(10):
        path=PHASE/f'D011/raw/case-{case:02}/rep-00008.json.gz'
        row=json.load(gzip.open(path,'rt',encoding='utf8'));ref=row['reference_receipt']
        b=ref['branch_target'];ls=ConditionalLogF(b['ancillary_source']);lt=ConditionalLogF(b['ancillary_target'])
        ms,mt=b['centering'];c=b['selection']['switch_threshold'];tau=b['soft_tau']
        evidence=PHASE/'D011'/row['evidence_path'];assert sha(evidence)==row['evidence_sha']
        with np.load(evidence,allow_pickle=False) as ev:
            assert np.array_equal(ev['reference_base'][0],ev['reference_base'][1])
            base=ev['reference_base'][1]
        coarse=calculate(ls,lt,ms,mt,c,tau,base,512,4097)
        fine=calculate(ls,lt,ms,mt,c,tau,base,1024,8193)
        rows.append({'case':case,'rep':8,'source_sha':sha(path),'evidence_sha':row['evidence_sha'],
            'coarse':coarse,'fine':fine,'absolute_ratio_refinement_changes':[
                abs(a['ratio']-b['ratio']) if a['ratio'] is not None and b['ratio'] is not None else None
                for a,b in zip(coarse['rows'],fine['rows'])]})
        write(out/f'case-{case:02}.json',rows[-1])
        print(case,[(r['threshold'],round(r['ratio'],5) if r['ratio'] is not None else 'below_resolution') for r in fine['rows']],flush=True)
    write(out/'result.json',{'status':'DETERMINISTIC_DIAGNOSIS_COMPLETE_NOT_NEW_EXPERIMENT','rows':rows,
        'external_review':'NOT_CONDUCTED','repair':'D012_2 reviewer correction: normalize source quadrature weights and target grid consistently; earlier outputs preserved. No deployed method change. Near-saturation bound NOT evaluated.',
        'not_claimed':['novel general theorem','exact floating certificate','FDR confirmation','Power bound']})

if __name__=='__main__':main()
