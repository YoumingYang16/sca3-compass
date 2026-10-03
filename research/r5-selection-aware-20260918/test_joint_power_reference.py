import numpy as np
import pytest
from scipy.special import betaln
from joint_power_reference import TiltEnvelope,log_dominating_moment,negative_binomial_trials,JointReferenceFailure,positive_int,shape_sizebias_log_accept
from joint_power_kernel import pc_e,evaluate

def test_analytic_envelope_integral_and_normalizer_importance():
    from scipy.integrate import quad
    e=TiltEnvelope(np.zeros(4),0,0)
    # Integrate separately at the two envelope corners.
    exact=sum(quad(lambda x:float(np.exp(e.log_bound(x))),a,b,epsabs=1e-11)[0] for a,b in
              [(-np.inf,e.cl),(e.cl,e.cr),(e.cr,np.inf)])
    assert np.isclose(exact,np.exp(e.logC),rtol=1e-9)
    mean,rec=e.normalizer_ratio(np.random.default_rng(447),100000)
    z=np.exp(4*np.log(2)+betaln(8,4));estimated=np.exp(e.logC)*mean
    assert abs(estimated-z)<6*np.exp(e.logC)*rec['SE_descriptive']

def test_tilted_envelope_covers_and_moment_constant():
    assert np.isclose(np.exp(log_dominating_moment(4)),200.)
    for r in [0.,.1,1.,2.]:
        e=TiltEnvelope(np.array([-2.,-.2,.5,1.7]),r,.4)
        x=np.linspace(-30,30,1001)
        assert np.all(e.log_density(x)<=e.log_bound(x))

def test_negative_binomial_uses_exact_success_index_not_batch_overshoot():
    class Fixed:
        def random(self,n):return np.full(n,.5)
    n,receipt=negative_binomial_trials(lambda n:np.zeros(n),Fixed(),3,100,batch=64)
    assert n==3 and receipt['generated_proposals']==64
    with pytest.raises(JointReferenceFailure) as err:
        negative_binomial_trials(lambda n:np.full(n,-1000.),Fixed(),3,10,batch=64)
    assert err.value.receipt['proposed']==10

def test_direct_pc_e_uses_minimum_null_subset():
    marginal=np.array([[[0.,1.,2.,100.],[-2.,0.,0.,0.]]]);projected=np.ones_like(marginal)
    ordinary,projection=pc_e(marginal,projected,0.)
    assert np.isclose(ordinary[0,0],17/3) and ordinary[0,1]==0
    assert np.array_equal(projection,np.ones((1,2)))

def test_bad_bank_fails_closed_no_retry():
    r=np.random.default_rng(913);z=r.normal(size=(16,4,6));cs=r.normal(size=(4,4,6));ct=np.zeros((4,4,6))
    v=evaluate(z,cs,ct,bound=1.,seed=4,successes=1,meta_draws=31,numerator_draws=31,proposal_cap=1000)
    assert v['status']=='conservative_numerical_failure' and not v['decision'].any()

def test_finite_budgets_and_reference_cap_failure_receipt():
    for x in [np.inf,1.5,True,-1]:
        with pytest.raises(ValueError):positive_int(x,'budget')
    from joint_power_reference import inverse_pair
    with pytest.raises(JointReferenceFailure) as err:
        inverse_pair(np.zeros(4),np.zeros(4),0.,0.,.5,1.,.5,0.,16,91,successes=2,numerator_draws=16,cap=2)
    assert err.value.receipt['stage']=='boundary'
    assert 'numerator_source' in err.value.receipt

def test_identity_shape_size_bias_matches_one_over_96():
    la=shape_sizebias_log_accept(np.random.default_rng(313),100000,16,4.,True)
    weights=np.exp(la);se=weights.std(ddof=1)/np.sqrt(len(weights))
    assert abs(weights.mean()-1/96)<6*se
