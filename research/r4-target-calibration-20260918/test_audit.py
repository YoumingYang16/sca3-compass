"""Synthetic record-mutation tests; no confirmation observations are read."""
from copy import deepcopy
import numpy as np
import pytest
from audit_r4 import summary_structure,independent_ebh,independent_score
from r4_common import read,PHASE

def fixture_summary():
    p=read(PHASE/'C001/protocol.json')
    keys={a+'_minus_'+b: {} for a,b in p['comparisons']}
    rows=[]
    for c,scene in enumerate(p['cases']):
        null=scene.get('mixed_sign_null') or scene.get('truth') in ['global_null','single_study_only']
        rows.append({'case':c,'scene':deepcopy(scene),'n':p['repetitions'],
                     'methods':{k:{} for k in p['methods']},'paired_power':{} if null else deepcopy(keys)})
    s={'status':'COMPLETE','stage':p['stage'],'alpha':p['alpha'],'interval_cap':p['interval_cap'],
       'intervals_count':p['planned_intervals'],'index_sha256':'SYNTHETIC_INDEX','freeze_sha256':'SYNTHETIC_FREEZE',
       'rows':rows,'core_paired_power':deepcopy(keys),'allocation_comparisons':{
           f'case{a}_minus_case{b}_{k}':{} for a,b in p['allocation_comparisons'] for k in ['target','bridge','strong_target']}}
    return p,s

def test_complete_synthetic_structure():
    p,s=fixture_summary()
    assert len(summary_structure(p,s,'SYNTHETIC_INDEX','SYNTHETIC_FREEZE'))==10

@pytest.mark.parametrize('mutation',['missing_all','duplicate','extra_method','alpha','scene','n','paired','core','allocation','count','index'])
def test_corrupt_reports_rejected(mutation):
    p,s=fixture_summary()
    if mutation=='missing_all': s['rows']=[]
    elif mutation=='duplicate': s['rows'][1]=deepcopy(s['rows'][0])
    elif mutation=='extra_method': s['rows'][0]['methods']['unregistered']={}
    elif mutation=='alpha': s['alpha']=.05
    elif mutation=='scene': s['rows'][0]['scene']['D']=2.
    elif mutation=='n': s['rows'][0]['n']=511
    elif mutation=='paired': s['rows'][0]['paired_power']={}
    elif mutation=='core': s['core_paired_power']['posthoc']={}
    elif mutation=='allocation': s['allocation_comparisons']={}
    elif mutation=='count': s['intervals_count']=132
    elif mutation=='index': s['index_sha256']='different'
    with pytest.raises(ValueError): summary_structure(p,s,'SYNTHETIC_INDEX','SYNTHETIC_FREEZE')

@pytest.mark.parametrize('e,expected',[
    ([80,0,0,0],[True,False,False,False]),
    ([40,40,0,0],[True,True,False,False]),
    ([20,20,20,20],[True]*4),([79,0,0,0],[False]*4),([0]*4,[False]*4)])
def test_independent_ebh_known_ties(e,expected):
    assert np.array_equal(independent_ebh(np.array(e,float)),expected)

def test_independent_score_known_truth():
    d=np.array([[1,0],[1,1]],bool);t=np.array([[1,0],[0,1]],bool)
    assert independent_score(d,t)=={'power':1.,'fdp':1/3,'tp':2,'fp':1,'discoveries':3}
