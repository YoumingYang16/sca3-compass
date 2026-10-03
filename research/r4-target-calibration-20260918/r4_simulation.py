"""Simulator only. Independent streams; truth NEVER passed to inference."""
import numpy as np
from r4_common import exp
from sca3_compass.molecular_envelope_benchmark import fixed_truth

def generate(protocol,case,rep):
    c=protocol['cases'][case];g=protocol['family_G']
    # The stream key uses a design pairing ID, NOT Ns/Nt. Independent Z, Cs,
    # Ct and auxiliary randomization. Changing Nt never changes Z or Cs.
    pair=c.get('pair_id',case)
    ss=np.random.SeedSequence([protocol['seed'],pair,rep]).spawn(4)
    rz,rs,rt=[np.random.default_rng(s) for s in ss[:3]]
    shape=exp.simulation_equicorrelation(4,c.get('study_rho',.65))
    left=exp.covariance_root(shape)
    rho_t=c['rho_target'];rho_s=c['rho_source']
    pipe=exp.covariance_root(exp.simulation_equicorrelation(6,rho_t))
    noise=left@rz.normal(size=(g,4,6))@pipe.T
    radii=exp.radial(rz,(g,1,1),c['distribution']);noise*=radii
    mu=fixed_truth(g,4,{'effect':c['effect'],'truth':c.get('truth'),
                       'replicated_fraction':c.get('fraction',.2)},.2)
    if c.get('mixed_sign_null'):
        mu=np.zeros((g,4));mu[:,0]=c['effect'];mu[:,1]=-c['effect']
    if c.get('continuous'):
        mu=np.take_along_axis(mu,np.argsort(rz.random((g,4)),axis=1),axis=1)
        jitter=c.get('effect_jitter',.7)
        mu*=np.exp(jitter*rz.normal(size=mu.shape)-jitter*jitter/2)
    z=noise+mu[...,None]
    truth=np.stack([(mu>0).sum(1)>=2,(mu<0).sum(1)>=2],axis=1)
    def make_bank(rng,n,rho,kind):
        # Blockwise factor/radius independent of Gaussian matrix, arbitrary
        # invertible left transforms are legal. Use simple identity here.
        a=rng.normal(size=(n,4,6))@exp.covariance_root(exp.simulation_equicorrelation(6,rho)).T
        return a*exp.radial(rng,(n,1,1),kind)
    cs=make_bank(rs,c['ns'],rho_s,c.get('source_distribution',c['distribution']))
    ct=make_bank(rt,c['nt'],rho_t,c.get('target_distribution',c['distribution']))
    seed=int(ss[3].generate_state(1,dtype=np.uint64)[0])
    kt=(1+5*rho_t)/(6*(1-rho_t));ks=(1+5*rho_s)/(6*(1-rho_s))
    return {'z':z,'calibration_source':cs,'calibration_target':ct}, {
        'truth':truth,'mu':mu,'shape_DIAGNOSTIC_ONLY':shape,
        'kappa_target_DIAGNOSTIC_ONLY':kt,'kappa_source_DIAGNOSTIC_ONLY':ks,
        'radii_DIAGNOSTIC_ONLY':radii[:,0,0]},seed
