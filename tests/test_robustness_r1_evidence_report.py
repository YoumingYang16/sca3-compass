"""Deterministic reporting fixtures, not new scientific experiments."""
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'scripts/robustness_r1_evidence_report.py'
spec = importlib.util.spec_from_file_location('r1_evidence_report_fixture', path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

MAIN = 'R1B_guard_pilotc0.5_projection_gate_eBH'
GUARD = 'R1B_guard_pilotc0.5_ordinary_bonf_eBH'
PLUGIN = 'R1B_plugin_pilotc0.5_ordinary_bonf_eBH'


def fixture():
    cases = [{'name': 'ordinary'}, {'name': 'heavy'}]
    plan = {'candidate': MAIN, 'reported_methods': [MAIN, GUARD, PLUGIN],
            'validity_scope_indices': [0, 1], 'strata': {'core54': [0, 1]},
            'families': {'F_safe': [GUARD], 'F_empirical': [GUARD]}}
    protocol = {'run_id': 'FIXTURE', 'cases': cases, 'repetition_counts': [100, 200],
                'analysis_plan': plan}
    rows = []
    for i, case in enumerate(cases):
        methods = {}
        for method, power in [(MAIN, [.8, .4][i]), (GUARD, [.7, .2][i]), (PLUGIN, [.9, .5][i])]:
            methods[method] = {'power': power, 'fdp': .01, 'mean_tp': power*10,
                               'fdp_interval': [.001, .02 if i == 0 else .04]}
        rows.append({'case_index': i, 'case': case, 'n': protocol['repetition_counts'][i],
                     'methods': methods, 'fallback_folds': 0})
    analysis = {'run_id': 'FIXTURE', 'complete': True, 'phase': 'C1',
                'analysis_plan': deepcopy(plan), 'whole_family_repetitions': 300,
                'rows': rows, 'comparisons': {'core54/F_safe': {'lower': .01}},
                'contributions': {'fixture': {'lower': -.1, 'upper': .1}}}
    return analysis, protocol


def test_fixed_primary_results_unchanged_and_case_not_repeat_weighting():
    a, p = fixture()
    before = deepcopy(a)
    result = report.summarize(a, p)
    assert result['primary_comparisons_unmodified'] == a['comparisons']
    assert result['primary_contributions_unmodified'] == a['contributions']
    assert result['all_reported_methods_descriptive'][MAIN]['strata']['core54']['mean_power'] == pytest.approx(.6)
    plugin = result['additional_plugin_comparisons']['core54/F_safe_plugin_descriptive']
    assert plugin['candidate_minus_plugin_envelope'] == pytest.approx(-.1)
    assert plugin['interval'] is None
    assert 'DESCRIPTIVE' in plugin['status']
    assert a == before


@pytest.mark.parametrize('change', ['incomplete', 'wrong_phase', 'missing_count', 'duplicate_case', 'wrong_case', 'no_scope', 'changed_plan'])
def test_refuse_incomplete_or_mismatched_science(change):
    a, p = fixture()
    if change == 'incomplete': a['complete'] = False
    elif change == 'wrong_phase': a['phase'] = 'development'
    elif change == 'missing_count': a['whole_family_repetitions'] -= 1
    elif change == 'duplicate_case': a['rows'][1]['case_index'] = 0
    elif change == 'wrong_case': a['rows'][1]['case'] = {'name': 'different'}
    elif change == 'no_scope':
        p['analysis_plan']['validity_scope_indices'] = []
        a['analysis_plan'] = deepcopy(p['analysis_plan'])
    elif change == 'changed_plan': a['analysis_plan']['candidate'] = PLUGIN
    with pytest.raises(ValueError): report.summarize(a, p)


def test_inconclusive_fdr_upper_is_not_evidence_of_inflation():
    a, p = fixture()
    a['rows'][1]['methods'][MAIN]['fdp_interval'] = [.02, .08]
    result = report.summarize(a, p)
    item = result['all_reported_methods_descriptive'][MAIN]
    assert item['scope_upper_above_nominal_cases'] == [1]
    assert item['scope_lower_above_nominal_cases'] == []
    assert result['candidate_scope_fdr_condition'] is False


def test_missing_plugin_is_not_silently_removed_from_comparison():
    a, p = fixture()
    p['analysis_plan']['reported_methods'].remove(PLUGIN)
    a['analysis_plan'] = deepcopy(p['analysis_plan'])
    with pytest.raises(ValueError, match='Missing equally'): report.summarize(a, p)


def test_undefined_power_cannot_enter_core():
    a, p = fixture()
    a['rows'][0]['methods'][MAIN]['power'] = None
    with pytest.raises(ValueError, match='Undefined Power'): report.summarize(a, p)
