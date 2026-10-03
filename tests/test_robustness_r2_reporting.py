"""Deterministic renderer fixtures, never real C2 scientific outputs."""
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from robustness_r2_evidence_report import fmt,interval,render,figures
from robustness_r2_development import read
from robustness_r2_confirm_analyze import distribution


def fixture():
    plan=read(ROOT/'configs/robustness_C2_analysis.json')
    methods={m:{'power':.3,'fdp':.01,'fdp_interval':[.002,.03],
                'mean_tp':15.,'mean_fp':1.} for m in plan['reported_methods']}
    ci={'mean_difference':.01,'lower':-.02,'upper':.03,'paired_mcse':.001}
    rows=[{'case_index':i,'case':case,'n':n,'methods':deepcopy(methods),
           'I':deepcopy(ci) if i in plan['power_defined_indices'] else None}
          for i,(case,n) in enumerate(zip(plan['cases'],plan['repetition_counts']))]
    for row in rows:
        if row['I'] is None:
            for m in row['methods'].values():m['power']=None
    result={'analysis_plan':plan,'whole_family_repetitions':82800,'whole_family_failures':0,
        'input_evidence_bytes_verified':123,'incremental_statistical_criteria_pass':False,
        'broad_success_criteria_pass':False,'I':{s:deepcopy(ci) for s in plan['strata']},
        'comparisons':{s+'/'+f:{'sample_envelope_difference':.01,'lower':-.02,'upper':.03}
                       for s in plan['strata'] for f in plan['families']},
        'stratum_methods':{s:deepcopy(methods) for s in plan['strata']},'rows':rows,
        'R1_failed_folds':[],'K_failed_folds':[],'pattern_fallback_folds':0,
        'recorded_cpu_seconds':1.,'summed_family_wall_seconds':2.,'generator_wall_seconds':.5,
        'analysis_elapsed_seconds':.3}
    return result,{'receipts':[]},{'contributions':{'core54/test':result['comparisons']['core54/H']}}


def test_units_undefined_values_and_asymmetric_bounds():
    assert fmt(None)=='未定义'
    assert fmt(.01234)=='1.2340'
    assert interval({'mean_difference':.01,'lower':-.005,'upper':.04})=='1.0000 [-0.5000, 4.0000]'


def test_report_contains_all_stresses_negative_evidence_and_no_auto_promotion():
    result,replay,c1=fixture();out=render(result,replay,c1)
    assert '待主研究者最终审查' in out and '广泛成功条件：False' in out
    assert '全部同信息72个简单对照' in out and '未定义' in out
    assert '不追加第三次' in out
    for i in range(54,84):assert f'|{i} ' in out


def test_all_three_scope_panels_render_to_fresh_unit_fixture(tmp_path):
    result,_,_=fixture()
    figures(result,tmp_path)
    assert {p.name for p in tmp_path.iterdir()}=={'paired-and-envelope.png','paired-and-envelope.svg',
        'scoped-and-failure-fdr.png','scoped-and-failure-fdr.svg'}
    assert all(p.stat().st_size>1000 for p in tmp_path.iterdir())


def test_distribution_describes_tails_not_only_average():
    r=distribution(np.r_[np.zeros(99),1.])
    assert r['q95']==0 and r['max']==1 and r['sd']==pytest.approx(.1)
