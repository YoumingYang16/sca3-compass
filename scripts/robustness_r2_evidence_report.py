"""Generate C2 tables/figures only from complete frozen analysis and replay.

No new samples, significance tests, selection rules or promotion decision.
"""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sca3_compass.robustness_io import write_json
from robustness_r2_development import read,sha


def fmt(value):
    return '未定义' if value is None else f'{100*value:.4f}'


def interval(row):
    estimate=row.get('mean_difference',row.get('sample_envelope_difference'))
    return f"{fmt(estimate)} [{fmt(row['lower'])}, {fmt(row['upper'])}]"


def render(result,replay,c1):
    p=result['analysis_plan'];candidate=p['candidate'];reference=p['reference']
    lines=['# R2/C2完整证据：待主研究者最终审查','',
        f"生成：{datetime.now(timezone.utc).isoformat()}。全部新实验为模拟，不是患者数据。",'',
        f"C2/R0076固定{result['whole_family_repetitions']:,}次、84场景，整家族失败{result['whole_family_failures']}。"
        f"原始输入/证据核查{result['input_evidence_bytes_verified']:,}字节；"
        f"{len(replay['receipts'])}个原输入fresh回放，124个e值方法、24个p值组件和评分完全一致。",'',
        f"冻结的增量统计条件：{result['incremental_statistical_criteria_pass']}；"
        f"广泛成功条件：{result['broad_success_criteria_pass']}。"
        '这些布尔值不代替主研究者对局部损失、成本和范围的审查；不存在一般未知参数有效性证明。','',
        '## H/F/I主要比较','',
        '单位均为百分点，Power差越大越好。以下端点共享预定C2报告误差.025；不是普通单项95%。', '',
        '|比较|整体54|普通27|t5的27|','|---|---:|---:|---:|',
        '|I：K−同批冻结R1|'+'|'.join(interval(result['I'][s]) for s in ['core54','normal27','t5_27'])+'|']
    for family,label in [('H','历史H15包络'),('F_safe_same_K','同K安全交集简单包络'),
                         ('F_empirical_same_K','同K经验Simes包络'),('F_all_same_information','全部同信息72个简单对照')]:
        lines.append('|'+label+'|'+'|'.join(interval(result['comparisons'][s+'/'+family]) for s in ['core54','normal27','t5_27'])+'|')
    lines += ['', 'I是在同一C2新数据上的配对增量，不跨C1/C2相减。H/F先逐场景取期望Power包络再等权；样本包络向下偏，其下端与DEV固定比较器上端使用不同构造。I使用独立完整家族的经验Bernstein界；MCSE另报。', '',
        '|固定方法|核心Power%|平均正确发现数|原68最大观察FDR% / 最大同时上界%|新8场景最大观察FDR% / 最大同时上界%|',
        '|---|---:|---:|---:|---:|']
    for method,label in [(reference,'R1完整保护'),(candidate,'K仅ρ保护'),
        ('R1B_plugin_pilotc0.5_projection_gate_eBH','P原plugin'),
        ('R1B_plugin_pilotc0.5_support_simes_empirical_eBH','P支持Simes c=.5'),
        ('R2K_pilotc0.5_support_simes_empirical_eBH','K支持Simes c=.5')]:
        core=result['stratum_methods']['core54'][method]
        def worst(indices):
            a=max(result['rows'][i]['methods'][method]['fdp'] for i in indices)
            b=max(result['rows'][i]['methods'][method]['fdp_interval'][1] for i in indices)
            return fmt(a)+' / '+fmt(b)
        lines.append(f"|{label}|{fmt(core['power'])}|{core['mean_tp']:.4f}|{worst(p['original_validity_scope_indices'])}|{worst(p['expanded_matched_indices'])}|")
    lines += ['', '## 所有压力场景（不删失败）','',
        'FDR是每家族FDP的均值；Power无定义时不写0。括号内为K的预定同时FDR区间。','',
        '|场景编号与名称|范围|R1 Power%|K Power%|K−R1点估计和同时区间，百分点|K FDR% [区间]|P FDR%|',
        '|---|---|---:|---:|---:|---:|---:|']
    for i in range(54,84):
        row=result['rows'][i];c=row['methods'][candidate];r=row['methods'][reference]
        label=row['case'].get('name',str(i))
        scope='旧模型内' if i in p['original_validity_scope_indices'] else '新增匹配数值范围' if i>=76 else '保留的假设外'
        ci=c['fdp_interval'];ival='无定义' if row['I'] is None else interval(row['I'])
        lines.append(f"|{i} {label}|{scope}|{fmt(r['power'])}|{fmt(c['power'])}|{ival}|{fmt(c['fdp'])} [{fmt(ci[0])}, {fmt(ci[1])}]|{fmt(row['methods']['R1B_plugin_pilotc0.5_projection_gate_eBH']['fdp'])}|")
    locals_=sorted([row for row in result['rows'] if row['I'] is not None],key=lambda x:x['I']['mean_difference'])
    lines += ['', '## 最弱局部增量与统计精度','',
        '|场景|K−R1点估计和区间，百分点|配对MCSE，百分点|', '|---|---:|---:|']
    for row in locals_[:10]:
        lines.append(f"|{row['case_index']} {row['case'].get('name','core')}|{interval(row['I'])}|{fmt(row['I']['paired_mcse'])}|")
    lines += ['', '局部区间宽不能宣布等效或证明没有退化；没有事后加入非劣阈值。场景差异、随机模拟误差、开发选择偏差是三个不同概念。C2前已冻结所有选择；旧开发数据不是C2独立重复。', '',
        '## 贡献、失败与资源','',
        '原C1四格贡献保留原区间（不重新分配C1错误预算）：', '',
        '|C1整体对照|百分点及预定区间|', '|---|---:|']
    for label,row in c1['contributions'].items():
        if label.startswith('core54/'):
            lines.append('|'+label.split('/',1)[1]+'|'+interval(row)+'|')
    lines += ['', '四格是前端连同TRAIN pilot的贡献，不是纯尺度因果效应；最简单Bonferroni不是最强F。旧own-Q删除的独立贡献未由该四格识别。K只取消目标形状/径向bootstrap最坏情形，未改变投影家族，不能将这一效率增量写成新投影创新。', '',
        f"C2内部R1失败fold {len(result['R1_failed_folds'])}；K失败fold {len(result['K_failed_folds'])}；"
        f"模式fallback {result['pattern_fallback_folds']}。失败fold按冻结规则p=1，未丢弃家族。",
        f"记录CPU合计{result['recorded_cpu_seconds']:.3f}秒；家族墙钟之和{result['summed_family_wall_seconds']:.3f}秒（不是并行批次耗时）。"
        f"生成批次墙钟{result['generator_wall_seconds']:.3f}秒；全量审计分析{result['analysis_elapsed_seconds']:.3f}秒。",
        '本批共同计算R0/R1/K与全部对照，且K共享当前家族的新R1中间结果，不是可部署API成本对照。部署API成本只能引用已完成的独立计时材料，并披露返回标签数不同。', '',
        '## 结论约束与复现入口','',
        '- E：完整执行、逐数组重算和fresh回放；不是理论或临床有效性。',
        '- V：已知真实参数链条有推导；有限bootstrap/估计前端仍是有限场景实证。漂移等旧假设外失败全部保留。',
        '- P：只对对应预定区间下端为正的比较声称优势。不能用同K前端优势替代全部同信息强基线优势。',
        '- N：与已有moderated-t、部分合取、非负投影、bootstrap敏感度保护及eBH的组合关系已说明；尚无首创性/发表保证。','',
        '原始证据：`../R0076-confirmation-analysis.json`、`../R0076-replay.json`、`../R0076-results-index.json`、`../R0076-repetitions/`、`../R0076-analysis-arrays/`。',
        '冻结源与协议：`../R0076-source/`、`../R0076-screen.protocol.json`。生成本表/图：`generate_source.py`；所有失败记录保留在原始压缩记录及分析JSON。','',
        '本轮C1和C2已用完两次确认机会，不追加第三次。由主研究者审查后选择保留版本；广泛目标未达成时必须明确保留未完成状态。']
    return '\n'.join(lines)+'\n'


def figures(result,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white','figure.facecolor':'white'})
    strata=['core54','normal27','t5_27'];labels=['All 54','Normal 27','Student t5 27']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for j,s in enumerate(strata):
        row=result['I'][s];point=100*row['mean_difference']
        axes[0].errorbar(point,j,xerr=[[point-100*row['lower']],[100*row['upper']-point]],fmt='o',color='#155E75',capsize=4)
    axes[0].set(yticks=range(3),yticklabels=labels,xlabel='Paired Power difference (percentage points)',title='K minus frozen R1 | C2 only')
    axes[0].axvline(0,color='#777777',lw=1)
    families=['H','F_safe_same_K','F_empirical_same_K','F_all_same_information']
    for j,f in enumerate(families):
        row=result['comparisons']['core54/'+f];point=100*row['sample_envelope_difference']
        axes[1].errorbar(point,j,xerr=[[point-100*row['lower']],[100*row['upper']-point]],fmt='o',color='#7C3AED',capsize=4)
    axes[1].set(yticks=range(4),yticklabels=['Historical H','Same-K safe','Same-K empirical','All-information simple'],
        xlabel='Envelope Power difference (percentage points)',title='K vs strong comparators | 54-case macro')
    axes[1].axvline(0,color='#777777',lw=1)
    fig.suptitle('Fixed C2: 82,800 independent simulated families — not patient data',fontsize=13)
    for ax in axes:ax.grid(axis='x',alpha=.2)
    for suffix in ['png','svg']:fig.savefig(out/f'paired-and-envelope.{suffix}',dpi=180)
    plt.close(fig)
    p=result['analysis_plan'];main=p['candidate'];ref=p['reference'];plugin='R1B_plugin_pilotc0.5_projection_gate_eBH'
    groups=[('Original 68 working cases',p['original_validity_scope_indices']),
            ('Eight new small-calibration cases',p['expanded_matched_indices']),
            ('Eight retained assumption violations',[i for i in range(76) if i not in p['original_validity_scope_indices']])]
    fig,axes=plt.subplots(3,1,figsize=(13,11),layout='constrained')
    for ax,(label,indices) in zip(axes,groups):
        uppermax=6
        for m,tag,color,shift in [(ref,'R1','#64748B',-.18),(main,'K','#155E75',0),(plugin,'P/plugin','#C2410C',.18)]:
            rows=[result['rows'][i]['methods'][m] for i in indices];point=100*np.array([r['fdp'] for r in rows]);lo=100*np.array([r['fdp_interval'][0] for r in rows]);hi=100*np.array([r['fdp_interval'][1] for r in rows])
            ax.errorbar(np.arange(len(indices))+shift,point,yerr=[point-lo,hi-point],fmt='o',markersize=3,
                        color=color,alpha=.8,capsize=2,label=tag)
            uppermax=max(uppermax,float(hi.max())*1.08)
        ax.axhline(5,color='#991B1B',ls='--',lw=1,label='Nominal 5%')
        ax.set(xticks=range(len(indices)),xticklabels=[str(i) for i in indices],
               xlabel='Frozen case index (all cases retained)',ylabel='FDR (%)',title=label,ylim=(-.3,uppermax))
        ax.tick_params(axis='x',labelsize=7 if len(indices)>20 else 10)
        ax.grid(axis='y',alpha=.2);ax.legend(ncol=4,fontsize=9,loc='upper left')
    fig.suptitle('Per-case expected FDP and simultaneous bounds | no pooled-FDR masking',fontsize=13)
    for suffix in ['png','svg']:fig.savefig(out/f'scoped-and-failure-fdr.{suffix}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project-root',type=Path,default=ROOT);args=p.parse_args()
    root=args.project_root.resolve();base=root/'artifacts/robustness'
    result=read(base/'R0076-confirmation-analysis.json');replay=read(base/'R0076-replay.json');c1=read(base/'R0073-confirmation-analysis.json')
    if not result['complete'] or not replay['complete'] or replay['analysis_sha256']!=sha(base/'R0076-confirmation-analysis.json'):
        raise ValueError('Complete matching analysis and exact replay required')
    out=base/'R0076-evidence-report';out.mkdir(exist_ok=False)
    shutil.copy2(__file__,out/'generate_source.py')
    with (out/'README_ZH.md').open('x',encoding='utf-8') as stream:stream.write(render(result,replay,c1))
    figures(result,out)
    write_json(out/'manifest.json',{'run_id':'R0076','analysis_sha256':replay['analysis_sha256'],
        'replay_sha256':sha(base/'R0076-replay.json'),'C1_analysis_sha256':sha(base/'R0073-confirmation-analysis.json'),
        'script_sha256':sha(__file__),'generated_files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'status':'COMPLETE_GENERATED_MATERIALS_REQUIRE_MAIN_VISUAL_AND_SCIENTIFIC_REVIEW',
        'generated_utc':datetime.now(timezone.utc).isoformat()})
    print(out,flush=True)


if __name__=='__main__':main()
