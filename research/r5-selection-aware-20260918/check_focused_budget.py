"""M004 deterministic focused-payoff check, known H=I DIAGNOSTIC_ONLY."""
import gzip,json,shutil
import numpy as np
from scipy.special import betainc,expit,gammaln
from ancillary_calibration import ConditionalLogF
from check_adaptation_cost import nodes
from selection_profile import LOG_CENTER
from r5_common import PHASE,sha,write

SLACKS=[0.,.25,.5,.75,1.,1.5,2.,3.,4.,8.,16.]
THRESHOLDS=[0.,2.,4.,8.]

def call_moment(scale,q,nu=20.,alpha=4.):
    """E[(scale*V_+**alpha-q**alpha)+], V Student t_nu."""
    scale=np.asarray(scale,float)
    moment=np.exp(alpha/2*np.log(nu)+gammaln((alpha+1)/2)+gammaln((nu-alpha)/2)
                  -np.log(2)-.5*np.log(np.pi)-gammaln(nu/2))
    if q==0:return scale*moment
    cut=q/scale**(1/alpha);x=nu/(nu+cut*cut)
    result=scale*moment*betainc((nu-alpha)/2,(alpha+1)/2,x)-q**alpha*.5*betainc(nu/2,.5,x)
    if np.min(result)<-1e-12:raise ArithmeticError('negative analytic call')
    return np.maximum(result,0.)

def calculate(ws,wt,meta,n):
    ls,lt=ConditionalLogF(ws),ConditionalLogF(wt)
    if meta is None:
        x,xw,_=nodes(ls,512,0.);y,yw,_=nodes(lt,512,0.)
        ms=xw@x-LOG_CENTER;mt=yw@y-LOG_CENTER
        vs=xw@((x-xw@x)**2);vt=yw@((y-yw@y)**2)
        a=vt/(vs+vt);c=np.sqrt(vs+vt);tau=c*np.sqrt(3)/np.pi
    else:ms,mt,a,c,tau=meta
    x,xw,xrec=nodes(ls,n,LOG_CENTER+ms);y,yw,yrec=nodes(lt,n,LOG_CENTER+mt)
    k=(yw@np.exp(-2*y))/((xw@np.exp(-2*a*x))*(yw@np.exp(-2*(1-a)*y)))
    rows=[]
    for q in THRESHOLDS:
        target=call_moment(np.exp(-2*y),q);infty=float(yw@target);values=[]
        for s in SLACKS:
            u=x[:,None]-y[None,:]+s;gate=expit((c-u)/tau)
            bridge=call_moment(k*np.exp(-2*y[None,:]-2*a*u),q)
            val=float(xw@(gate*bridge+(1-gate)*target[None,:])@yw)
            values.append(val)
        endpoint=max(values[0],infty);imax=int(np.argmax(values))
        rows.append({'threshold':q,'means':values,'infinity':infty,'endpoint_max':endpoint,
                     'maximum_grid':values[imax],'argmax_grid':SLACKS[imax],
                     'relative_interior_excess':values[imax]/endpoint-1})
    return {'meta':{'ms':ms,'mt':mt,'a':a,'c':c,'tau':tau,'k':k},'rows':rows,
            'quadrature':[xrec,yrec]}

def main():
    out=PHASE/'M004_1';out.mkdir(exist_ok=False)
    files=['M004_RULE.md','check_focused_budget.py','check_adaptation_cost.py',
           'finite_tail_diagnostic.py','ancillary_calibration.py','selection_profile.py','r5_common.py']
    for f in files:shutil.copy2(PHASE/f,out/f)
    write(out/'protocol.json',{'kind':'FIXED_KNOWN_SHAPE_MATH_DIAGNOSIS','resolutions':[256,512],
         'slacks':SLACKS,'thresholds':THRESHOLDS,'no_observed_families':True,'confirmation':False})
    write(out/'freeze.json',{'files':{f:sha(out/f) for f in files+['protocol.json']}})
    path=PHASE/'D011/raw/case-03/rep-00008.json.gz'
    row=json.load(gzip.open(path,'rt',encoding='utf8'))['reference_receipt']['branch_target']
    meta=[*row['centering'],row['meta']['source_weight'],row['selection']['switch_threshold'],row['soft_tau']]
    fixtures=[('zero32_4',np.zeros(32),np.zeros(4),None),('zero4_4',np.zeros(4),np.zeros(4),None),
              ('savedC3rep8',row['ancillary_source'],row['ancillary_target'],meta)]
    outputs=[]
    for name,ws,wt,m in fixtures:
        coarse=calculate(ws,wt,m,256);fine=calculate(ws,wt,m,512)
        changes=[max(abs(np.array(c['means'])-np.array(f['means'])))/f['endpoint_max'] for c,f in zip(coarse['rows'],fine['rows'])]
        result={'fixture':name,'coarse':coarse,'fine':fine,'relative_refinement_changes':changes,
                'saved_input_sha':sha(path) if m else None,'known_identity_shape_DIAGNOSTIC_ONLY':True}
        write(out/(name+'.json'),result);outputs.append(result)
        print(name,[(v['threshold'],v['relative_interior_excess'],v['argmax_grid']) for v in fine['rows']],flush=True)
    write(out/'result.json',{'status':'FIXED_DIAGNOSTIC_COMPLETE_NOT_A_GENERAL_THEOREM','outputs':outputs,
         'no_formal_confirmation':True,'EXTERNAL_REVIEW':'NOT_CONDUCTED'})

if __name__=='__main__':main()
