import copy
import pytest
from sca3_compass.robustness_confirmation_protocol import validate_plan

def fixture():
    cases=[{'stratum':'core','distribution':'normal' if i<27 else 't5','assumption_class':'matched_working_model'} for i in range(54)]
    plan={'reported_methods':['a','b','c'],'candidate':'a','strata':{'core54':list(range(54)),'normal27':list(range(27)),'t5_27':list(range(27,54))},
        'families':{f:['b','c'] for f in ['H','F_safe','F_empirical']},
        'development_selected_baselines':{f:{str(i):'b' for i in range(54)} for f in ['H','F_safe','F_empirical']},
        'validity_scope_indices':list(range(54)),'repetition_counts':[1000]*54,'reporting_error_budget':.025,
        'error_allocation':{'envelopes':.012,'FDR_candidate':.006,'FDR_other':.004,'contributions':.003},
        'contribution_pairs':{'1':['a','b'],'2':['a','c'],'3':['b','c'],'4':['c','b']}}
    return plan,cases

def test_exact_design():validate_plan(*fixture())

@pytest.mark.parametrize('bad',['duplicate_case','empty_scope','duplicate_method','noninteger_count','over_budget','wrong_dev_baseline','wrong_stratum'])
def test_reject_changed_design(bad):
    p,c=fixture()
    if bad=='duplicate_case':p['strata']['core54'][-1]=0
    if bad=='empty_scope':p['validity_scope_indices']=[]
    if bad=='duplicate_method':p['reported_methods'].append('b')
    if bad=='noninteger_count':p['repetition_counts'][0]=1000.
    if bad=='over_budget':p['error_allocation']['envelopes']=.013
    if bad=='wrong_dev_baseline':p['development_selected_baselines']['H']['0']='a'
    if bad=='wrong_stratum':p['strata']['normal27'][0]=27
    with pytest.raises(ValueError):validate_plan(p,c)
