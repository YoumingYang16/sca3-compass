"""Assemble completed, frozen R1 evidence; no fitting or new inference.

Primary confidence statements are copied from the frozen analyzer, never
recomputed under a different allocation. Plugin comparisons are explicitly
descriptive and cannot replace the prespecified guard-front-end F envelopes.
"""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(analysis, protocol):
    if not analysis['complete'] or analysis['phase'] != 'C1':
        raise ValueError('Complete C1 analysis required; never inspect a partial confirmation')
    plan = protocol['analysis_plan']
    if analysis['analysis_plan'] != plan:
        raise ValueError('Analysis/frozen plan mismatch')
    if analysis['run_id'] != protocol['run_id']:
        raise ValueError('Run identity mismatch')
    if analysis['whole_family_repetitions'] != sum(protocol['repetition_counts']):
        raise ValueError('Fixed sample not complete')
    rows = {r['case_index']: r for r in analysis['rows']}
    if len(rows) != len(analysis['rows']) or set(rows) != set(range(len(protocol['cases']))):
        raise ValueError('Exact distinct scenario membership required')
    for i, row in rows.items():
        if row['n'] != protocol['repetition_counts'][i] or row['case'] != protocol['cases'][i]:
            raise ValueError('Scenario identity/count mismatch')
    names = plan['reported_methods']
    main = plan['candidate']
    scoped = [rows[i] for i in plan['validity_scope_indices']]
    if not scoped:
        raise ValueError('Empty validity scope')
    descriptive = {}
    for method in names:
        worst = max(scoped, key=lambda r: r['methods'][method]['fdp'])
        worst_upper = max(scoped, key=lambda r: r['methods'][method]['fdp_interval'][1])
        scoped_excess = [r['case_index'] for r in scoped if r['methods'][method]['fdp_interval'][0] > .05]
        unresolved = [r['case_index'] for r in scoped if r['methods'][method]['fdp_interval'][1] > .05]
        strata = {}
        for label, indices in plan['strata'].items():
            if any(rows[i]['methods'][method]['power'] is None for i in indices):
                raise ValueError('Undefined Power entered a core stratum')
            strata[label] = {
                'mean_power': sum(rows[i]['methods'][method]['power'] for i in indices) / len(indices),
                'mean_true_discoveries': sum(rows[i]['methods'][method]['mean_tp'] for i in indices) / len(indices),
                'max_scenario_mean_fdp': max(rows[i]['methods'][method]['fdp'] for i in indices),
            }
        descriptive[method] = {
            'strata': strata,
            'scope_max_mean_fdp': worst['methods'][method]['fdp'],
            'scope_max_mean_fdp_case': worst['case_index'],
            'scope_max_fdr_upper': worst_upper['methods'][method]['fdp_interval'][1],
            'scope_max_fdr_upper_case': worst_upper['case_index'],
            'scope_upper_above_nominal_cases': unresolved,
            'scope_lower_above_nominal_cases': scoped_excess,
            'scope_all_fdr_upper_le_nominal': not unresolved,
        }
    plugin = {}
    for stratum, indices in plan['strata'].items():
        for family in ['F_safe', 'F_empirical']:
            methods = [m.replace('R1B_guard_', 'R1B_plugin_', 1) for m in plan['families'][family]]
            if not set(methods) <= set(names):
                raise ValueError('Missing equally informed plugin comparators')
            winners = {str(i): max(methods, key=lambda m: rows[i]['methods'][m]['power']) for i in indices}
            plugin[stratum + '/' + family + '_plugin_descriptive'] = {
                'candidate_minus_plugin_envelope': sum(
                    rows[i]['methods'][main]['power'] - rows[i]['methods'][winners[str(i)]]['power']
                    for i in indices) / len(indices),
                'sample_winner_by_case': winners,
                'interval': None,
                'status': 'DESCRIPTIVE_ONLY_NOT_A_FROZEN_PRIMARY_ENVELOPE',
            }
    return {
        'candidate': main,
        'primary_comparisons_unmodified': analysis['comparisons'],
        'primary_contributions_unmodified': analysis['contributions'],
        'all_reported_methods_descriptive': descriptive,
        'additional_plugin_comparisons': plugin,
        'preclassified_scope_count': len(scoped),
        'candidate_scope_fdr_condition': descriptive[main]['scope_all_fdr_upper_le_nominal'],
        'fallback_folds': sum(r['fallback_folds'] for r in rows.values()),
        'outside_scope': [
            {'case_index': i, 'case': r['case'], 'candidate': r['methods'][main]}
            for i, r in rows.items() if i not in plan['validity_scope_indices']
        ],
        'not_claimed': ['general bootstrap coverage', 'clinical efficacy', 'originality',
                        'all strong baselines beaten', 'original broad objectives solved'],
    }


def markdown(result, run):
    lines = [f'# {run}: 冻结R1证据汇总', '',
             '本报告仅整理已经完成的独立确认和工程回放，不增加检验、改变误差预算或选择新版本。', '',
             'Power越高越好；差值单位为百分点。区间来自冻结的联合报告方案，而非普通未校正95%区间。', '',
             '|集合/比较|样本包络差|联合覆盖下界|联合覆盖上界|', '|---|---:|---:|---:|']
    for label, row in result['primary_comparisons_unmodified'].items():
        lines.append(f"|{label}|{100*row['sample_envelope_difference']:+.4f}|{100*row['lower']:+.4f}|{100*row['upper']:+.4f}|")
    lines += ['', '## 有效性与复现', '',
              f"预先分类范围：{result['preclassified_scope_count']}个场景。主候选全部FDR上界≤5%：{result['candidate_scope_fdr_condition']}。",
              f"参数库fallback折数：{result['fallback_folds']}。这是限定模拟范围的证据，不是未知参数一般覆盖定理。",
              '', '## 固定版本与强对照的效率—可靠性', '',
              '以下Power均值为描述性结果；每个方法的逐场景FDR区间已包含在C1的预分配误差预算内。不能由FDR均值小于5%推断全范围已控制。', '',
              '|固定版本|核心Power%|模型内最坏FDR均值%|模型内最高FDR上界%|全部上界≤5%|',
              '|---|---:|---:|---:|---|']
    shown = [result['candidate'], 'R1B_plugin_pilotc0.5_projection_gate_eBH',
             'R1B_guard_pilotc0.5_support_simes_empirical_eBH',
             'R1B_plugin_pilotc0.5_support_simes_empirical_eBH',
             'R1B_guard_pilotc0.5_ordinary_bonf_eBH',
             'R1B_plugin_pilotc0.5_ordinary_bonf_eBH',
             'pilotc0.5_pattern_projection_support_gate_eBH']
    for name in shown:
        if name not in result['all_reported_methods_descriptive']:
            continue
        d = result['all_reported_methods_descriptive'][name]
        lines.append(f"|{name}|{100*d['strata']['core54']['mean_power']:.4f}|{100*d['scope_max_mean_fdp']:.4f}|{100*d['scope_max_fdr_upper']:.4f}|{d['scope_all_fdr_upper_le_nominal']}|")
    lines += ['', '## 未加保护的同信息基线：补充描述，不能替换主要F', '',
              '|集合/比较|主候选减plugin包络（百分点）|证据标签|', '|---|---:|---|']
    for label, row in result['additional_plugin_comparisons'].items():
        lines.append(f"|{label}|{100*row['candidate_minus_plugin_envelope']:+.4f}|描述性；无新增确认区间|")
    lines += ['', '## 全部假设外压力，原样保留', '',
              '|场景|Power%|FDR均值%|FDR区间%|', '|---|---:|---:|---|']
    for row in result['outside_scope']:
        d = row['candidate']
        p = '未定义' if d['power'] is None else f"{100*d['power']:.4f}"
        lines.append(f"|{row['case_index']} {row['case']['name']}|{p}|{100*d['fdp']:.4f}|[{100*d['fdp_interval'][0]:.4f}, {100*d['fdp_interval'][1]:.4f}]|")
    lines += ['', '## 解释限制', '',
              '- 主要H/F区间的下界为样本包络减有界差分半径；上界基于独立开发时固定的基线，不必对称。',
              '- 正包络下界仅支持对应范围的平均优势，不支持每个场景都更好。',
              '- F是同样采用guard前端的简单方法；plugin补充表防止把这一条件比较扩大成普遍优势。',
              '- 观察FDR上界高于5%表示该区间未证实≤5%；只有下界高于5%才支持该场景超标。',
              '- 共同数据、512个相关有符号声明不是512次独立实验；确认重复单位是完整模拟家族。',
              '- 本汇总不是对发表、真实患者应用或首创性的评定。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    folder = args.project_root.resolve() / 'artifacts/robustness'
    paths = {key: folder / f'{args.run_id}-{suffix}' for key, suffix in {
        'analysis': 'confirmation-analysis.json', 'protocol': 'screen.protocol.json',
        'index': 'results-index.json', 'replay': 'replay.json'}.items()}
    analysis, protocol, replay = (read(paths[k]) for k in ['analysis', 'protocol', 'replay'])
    if analysis['index_sha256'] != sha(paths['index']):
        raise ValueError('Index changed since frozen analysis')
    if replay['protocol_sha256'] != sha(paths['protocol']) or replay['run_id'] != args.run_id:
        raise ValueError('Replay/frozen protocol mismatch')
    if not replay['receipts'] or any(r['status'] != 'EXACT_ARRAYS_AND_METRICS' for r in replay['receipts']):
        raise ValueError('Completed exact replay required')
    result = summarize(analysis, protocol)
    result.update(run_id=args.run_id, phase='C1', created_utc=datetime.now(timezone.utc).isoformat(),
                  provenance='SIMULATION_NOT_PATIENT_DATA', script_sha256=sha(__file__),
                  sources={k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
                  complete_family_count=analysis['whole_family_repetitions'],
                  replayed_families=len(replay['receipts']))
    out = folder / f'{args.run_id}-evidence-report'
    out.mkdir(exist_ok=False)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'README_ZH.md').write_text(markdown(result, args.run_id), encoding='utf-8')
    shutil.copy2(__file__, out / 'generate_source.py')
    print(out)


if __name__ == '__main__':
    main()
