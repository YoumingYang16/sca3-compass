from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from molecular_v1_package import tables,percent,pp


def test_report_explicit_units_and_no_null_power_fabrication():
    assert percent(None)=='undefined'
    assert percent(.05)=='5.0000%'
    assert pp(.031)=='+3.1000'


def test_report_does_not_erase_outside_boundary():
    summary={'methods':['K_NR'],'paired':{name:{'power':{'K_NR':.5},'comparisons':{}} for name in ['core','normal','t5']},
        'scene_rows':[{'index':0,'scope':'D1','case':{'name':'inside'},'methods':{'K_NR':{'power':.5,'fdp':.01,'fdr_interval':[0.,.03]}}},
                      {'index':67,'scope':'OUTSIDE','case':{'name':'severe_drift'},'methods':{'K_NR':{'power':.99,'fdp':.74,'fdr_interval':[.7,.8]}}},
                      {'index':77,'scope':'D1','case':{'name':'singleton'},'methods':{'K_NR':{'power':None,'fdp':.01,'fdr_interval':[0.,.03]}}}],
        'cpu_seconds':10.,'wall_seconds':3.,'failed':[],'soft_fallback_families':[[77,0]]}
    result=tables(summary)
    assert '|67|OUTSIDE|severe_drift|K_NR|99.0000%|74.0000%' in result
    assert '|77|D1|singleton|K_NR|undefined|' in result
    assert 'not optimized standalone' in result
    assert 'declared conservative fold-fallback families 1' in result
    assert 'NOT 95% coverage over all historical selection' in result
