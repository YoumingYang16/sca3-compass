"""TEST_FIXTURE checks, never scientific evidence."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from robustness_r2_closeout_report import validate, fdr_cell, pct


def fixture():
    return {'audit_complete':True,'independent_confirmation_complete':False,
        'new_formal_intervals_computed':False,'successful_families':82,'failed_families':2,'planned_families':84,
        'rows':[{'n_planned':1,'n_successful':int(i<82),'n_failed':int(i>=82)} for i in range(84)]}


def test_failed_c2_cannot_be_presented_as_confirmation():
    a=fixture();validate(a)
    for key in ['independent_confirmation_complete','new_formal_intervals_computed']:
        b=fixture();b[key]=True
        with pytest.raises(ValueError):validate(b)
    a['rows'].pop()
    with pytest.raises(ValueError):validate(a)


def test_no_missing_output_disguised_as_fdr_point_or_population_interval():
    row={'n_failed':1,'methods':{'K':{'fdp':None,'fdp_sample_completion_bounds_not_CI':[.01,.0105]}}}
    text=fdr_cell(row,'K')
    assert '补全范围[1.0000, 1.0500]' in text and '缺1次' in text
    assert '置信' not in text
    row={'n_failed':0,'methods':{'K':{'fdp':.02}}}
    assert fdr_cell(row,'K')=='2.0000'
    assert pct(None)!='0' and pct(0)=='0.0000'


def test_failed_records_must_remain_in_planned_denominator():
    a=fixture();a['planned_families']=82
    with pytest.raises(ValueError):validate(a)
