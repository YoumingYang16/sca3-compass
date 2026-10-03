"""Small scalar selection-mechanism negative control, NOT full FDR evidence.

Known-shape Gaussian projected numerator; finite logF bank errors. Fixed seeds,
2048new scalar test draws per three shifts, shared reference banks. Conditional
rates only; NOT6144 independent full-family confirmations or formal intervals.
"""
from pathlib import Path
import sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from r5_common import PHASE,write,sha
from selection_profile import draw_errors,Profile,threshold,inner_rank

def main():
    out=PHASE/'D000';out.mkdir(exist_ok=False);start=time.perf_counter()
    seed=202609181601;ns=32;nt=4;n=2048;mi=4095;mo=1023
    write(out/'protocol.json',{'id':'R5-D000','scope':'SCALAR_MECHANISM_DIAGNOSTIC_NOT_FAMILY_FDR',
        'seed':seed,'ns':ns,'nt':nt,'draws_per_shift':n,'shifts':[0.,float(np.log(5.)),5.],
        'inner':mi,'outer':mo,'no_CI':'references shared; no unconditional replication claim',
        'source_sha256':sha(Path(__file__)),'profile_sha256':sha(PHASE/'selection_profile.py')})
    streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    ri,ro,rd,_=streams;a=ns/(ns+nt);c=threshold(ns,nt)
    it=np.sort(ri.normal(size=mi)*np.exp(-draw_errors(ri,nt,mi)/2))
    ib=np.sort(ri.normal(size=mi)*np.exp(-draw_errors(ri,ns+nt,mi)/2))
    es=draw_errors(ro,ns,mo);et=draw_errors(ro,nt,mo);base=ro.normal(size=mo)
    profile=Profile(es,et,base,ns,nt,it,ib);rows=[];arrays={'inner_target':it,'inner_bridge':ib,'outer_es':es,'outer_et':et,'outer_base':base}
    for j,s in enumerate([0.,np.log(5.),5.]):
        es=draw_errors(rd,ns,n);et=draw_errors(rd,nt,n);z=rd.normal(size=n)
        u=s+es-et;borrow=u<=c;logk=np.where(borrow,et+a*u,et);stat=z*np.exp(-logk/2)
        k=np.where(borrow,inner_rank(stat,ib),inner_rank(stat,it));unsafe=k/(mi+1)
        safe=profile.calibrate_rank(k)
        pt=inner_rank(z*np.exp(-et/2),it)/(mi+1)
        pb=inner_rank(z*np.exp(-(et+a*u)/2),ib)/(mi+1)
        rows.append({'shift':float(s),'borrow_fraction':float(borrow.mean()),
            'unsafe_selected_rejection':float((unsafe<=.05).mean()),
            'joint_profile_rejection':float((safe<=.05).mean()),
            'target_rejection':float((pt<=.05).mean()),'fixed_bridge_rejection':float((pb<=.05).mean())})
        arrays.update({f'{j}_{k}':v for k,v in {'es':es,'et':et,'z':z,'borrow':borrow,'unsafe':unsafe,'profile':safe,'target':pt,'bridge':pb}.items()})
    np.savez_compressed(out/'arrays.npz',**arrays)
    write(out/'result.json',{'status':'DEVELOPMENT_DIAGNOSTIC_ONLY','rows':rows,'seconds':time.perf_counter()-start,
        'arrays_sha256':sha(out/'arrays.npz'),'evaluated_rank_queries':len(profile.cache),
        'gate_effect':'No acceptance gate passes from this diagnostic'})
    print(rows,flush=True)

if __name__=='__main__':main()
