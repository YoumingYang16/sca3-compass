"""Failure-aware C2 DESCRIPTIVE closeout; no simulation or formal inference.

Repeat with --out a NEW project-local directory to regenerate identical tables
and scientific figures from the hash-bound audit. Original evidence is read-only.
"""
from research_window import wait_start_gate
wait_start_gate()
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pct(x):return '未定义/缺失' if x is None else f'{100*x:.4f}'


def validate(audit):
    if not audit['audit_complete'] or audit['independent_confirmation_complete'] or audit['new_formal_intervals_computed']:
        raise ValueError('Completed descriptive failed-batch audit ONLY')
    if audit['successful_families']+audit['failed_families']!=audit['planned_families'] or not audit['failed_families']:
        raise ValueError('Preserve every fixed attempted family, including failure')
    if len(audit['rows'])!=84 or any(r['n_successful']+r['n_failed']!=r['n_planned'] for r in audit['rows']):
        raise ValueError('Missing cases or denominator')
    if any(sum(r[key] for r in audit['rows'])!=audit[total] for key,total in
           [('n_planned','planned_families'),('n_successful','successful_families'),('n_failed','failed_families')]):
        raise ValueError('Per-case and global denominators disagree')


def fdr_cell(row, method):
    v=row['methods'][method]
    if row['n_failed']:
        lo,hi=v['fdp_sample_completion_bounds_not_CI']
        return f'缺{row["n_failed"]}次；补全范围[{pct(lo)}, {pct(hi)}]'
    return pct(v['fdp'])


def render(audit, repair, c1):
    validate(audit)
    plan=audit['plan'];k=plan['candidate'];r1=plan['reference']
    p='R1B_plugin_pilotc0.5_projection_gate_eBH'
    lines=['# R1/R2 收尾：保留 R1，K 尚未获独立确认','',
        '**本轮收尾不等于原研究目标全部成功。C2失败批次只作描述性分析，不报告新的正式确认区间。**','',
        f"来源：固定C2/R0076全部{audit['planned_families']:,}次尝试，{audit['successful_families']:,}次成功、{audit['failed_families']}次整家族失败。"
        '全部为SIMULATION_NOT_PATIENT_DATA，不是患者数据。失败未被删除、补零或替换。','',
        '## 1. 当前决定与证据等级','',
        '- R1：保留为原68场景的限定实证参考。C1独立结果不被R2失败抹去，但新增小校准场景暴露的数值缺陷必须披露。',
        '- K：保留冻结研究候选及原始输出，不晋升为经确认的优越方法。开发或本批描述性正增量不能替代完整确认。',
        '- 数值修复：独立POST-C2开发模块；只修复已知计算缺陷，不覆盖冻结版本，不回填失败记录，也不继承C1/C2验证标签。',
        '- 原目标A（整体竞争力）与B（核心重尾有效且有竞争力）未共同验收成功；没有一般估计后FDR保证或真实患者验证。','',
        '## 2. 同批配对增量和强基线比较','',
        'Power越高越好；差值单位为百分点。以下仅描述性。MCSE是完整模拟家族、分层等权配对平均的蒙特卡洛标准误，不是正式置信区间，也不涵盖开发选择偏差。', '',
        '|量|整体54|普通27|t5的27|','|---|---:|---:|---:|']
    strata=['core54','normal27','t5_27']
    lines.append('|K−同批R1，点估计（MCSE）|'+'|'.join(
        f"{pct(audit['strata'][s]['I']['mean_difference'])}（{pct(audit['strata'][s]['I']['paired_mcse'])}）" for s in strata)+'|')
    for f,label in [('H','原15基线H包络'),('F_safe_same_K','同K保护的安全简单基线包络'),
            ('F_empirical_same_K','同K保护的经验简单基线包络'),('F_all_same_information','全部同信息72个简单对照包络')]:
        lines.append('|K−'+label+'|'+'|'.join(pct(audit['comparisons'][s+'/'+f]['sample_envelope_difference']) for s in strata)+'|')
    lines += ['', '包络先逐场景取样本平均Power最强基线，再等权汇总；该样本包络存在选择偏差，不能凭此点估计宣称总体优越性。'
        '完整54核心场景不含两次缺失，但冻结停止规则仍把整批限定为描述性；未事后放宽规则。','',
        '|固定方法|核心Power%|平均正确发现数|','|---|---:|---:|']
    selected=[(r1,'R1'),(k,'K'),(p,'P/plugin'),('R1B_plugin_pilotc0.5_support_simes_empirical_eBH','P支持Simes'),
        ('R2K_pilotc0.5_support_simes_empirical_eBH','K支持Simes')]
    for m,label in selected:
        v=audit['strata']['core54']['methods'][m]
        lines.append(f"|{label}|{pct(v['power'])}|{v['mean_tp']:.4f}|")
    lines += ['', '## 3. 全部场景与失败边界','',
        'FDR估计是整家族FDP均值，不是合并假发现数/发现数。缺失场景不报告完整FDR点估计：'
        '给出的补全范围仅利用0≤FDP≤1形成样本代数范围，不是总体FDR置信区间，不代表失败时自动返回零发现。', '',
        '|编号/场景|范围|R1 Power%|K Power%|K−R1百分点|K FDR%或缺失范围|R1 FDR%或缺失范围|P FDR%或缺失范围|',
        '|---|---|---:|---:|---:|---|---|---|']
    for row in audit['rows']:
        i=row['case_index'];scope='C1原工作范围' if i in plan['original_validity_scope_indices'] else '新增匹配数值范围' if i in plan['expanded_matched_indices'] else '保留假设外'
        delta=row['I']['mean_difference'] if row['I'] else None
        lines.append(f"|{i} {row['case'].get('name','')}|{scope}|{pct(row['methods'][r1]['power'])}|{pct(row['methods'][k]['power'])}|{pct(delta)}|{fdr_cell(row,k)}|{fdr_cell(row,r1)}|{fdr_cell(row,p)}|")
    lines += ['', '### 最弱局部增量（全部有定义场景排序，不删除退化）','',
        '|场景|K−R1，百分点|配对MCSE，百分点|','|---|---:|---:|']
    for row in sorted([r for r in audit['rows'] if r['I']],key=lambda r:r['I']['mean_difference'])[:10]:
        lines.append(f"|{row['case_index']} {row['case'].get('name','')}|{pct(row['I']['mean_difference'])}|{pct(row['I']['paired_mcse'])}|")
    lines += ['', '## 4. 原C1贡献分解（保持原确认结论）','',
        'A=plugin普通Bonferroni；B=guard普通Bonferroni；C=plugin原始投影；D=guard原始投影。'
        '固定c=.5，guard同时改变TRAIN pilot；不是纯尺度因果效应，也不是最强F基线比较。', '',
        '|C1核心对照|百分点|原预定联合区间|','|---|---:|---|']
    for label,v in c1['contributions'].items():
        if label.startswith('core54/'):
            lines.append(f"|{label.split('/',1)[1]}|{pct(v.get('mean_difference',v.get('sample_envelope_difference')))}|[{pct(v['lower'])}, {pct(v['upper'])}]|")
    cg=c1['contributions']['core54/projection_extra_guard'];cp=c1['contributions']['core54/projection_extra_plugin']
    interaction=cg['sample_envelope_difference']-cp['sample_envelope_difference']
    lines += ['',f"协同交互(D−B)−(C−A)={pct(interaction)}个百分点；由原C1已联合覆盖区间相减得"
        f"[{pct(cg['lower']-cp['upper'])}, {pct(cg['upper']-cp['lower'])}]。区间跨零，不能确认协同；这不是新增C2推断。"]
    lines += ['', '旧own-Q删除收益不能从这四格中单独识别。R1的投影条件增益与保护效率损失均真实存在；二者不能相互抵消。','',
        '### C2数值回退与增量的描述性分解','',
        '|层|折回退分组（以家族为单位）|家族等权比例%|对总I的贡献，百分点|','|---|---|---:|---:|']
    for s,groups in audit['component_partition_descriptive'].items():
        for label,v in groups.items():
            lines.append(f"|{s}|{label}|{pct(v['weighted_fraction'])}|{pct(v['contribution_to_I'])}|")
    lines += ['', '分组互斥且贡献加总为同层I；该条件分解不是随机干预或因果证明。核心以外失败不能由核心表掩盖。','',
        '## 5. 两次原始失败与隔离修复','',
        '- case80/rep465：高斯数值解已接近解析最优，但优化器返回不收敛。',
        '- case81/rep500：两个Student起点中已有成功解；原选择器因约4.35e−13的目标函数差选择了失败解。',
        '- 修复保持似然、参数边界、两Student起点、BIC惩罚不变。高斯仅在解析解严格位于原边界内部时使用解析解，否则仍走有界优化；Student优先选择成功有限解。',
        '- 若全部Student起点失败，或高斯候选不可用，仍明确失败；不能丢弃一个拟合失败的模型后偷偷改变BIC选择。','',
        f"独立修复回归完成{len(repair['rows'])}个固定旧输入，每个两次fresh运行，全部132个R1/K e/p数组精确复现。"
        '其中含两次原失败输入和8个预先指定参照输入；没有新增样本或Power/FDR评分。','',
        '|旧输入|原初始化收敛|修复初始化收敛|两次fresh一致|相对原成功输出的最大归一变化|','|---|---|---|---|---:|']
    for row in repair['rows']:
        change=row['original_comparison']
        lines.append(f"|{row['case']}/{row['rep']}|{row['original_fit']['converged']}|{row['repaired_fit']['converged']}|{row['fresh_repeat_exact']}|"+
            ('原输出缺失' if change is None else f"{change['max_change_div_max_1_abs_old']:.3g}")+'|')
    lines += ['', '归一变化为max|new−old|/max(1,|old|)，不等于统计结论不变的证明。10个回归输入不能证明一般数值可靠性或统计有效性。','',
        '## 6. 成本、限制与下一轮唯一优先方向','',
        f"原C2批次墙钟{audit['original_generator_wall_seconds']/3600:.3f}小时；记录CPU合计{audit['recorded_cpu_seconds']/3600:.3f}核小时。"
        f"家族墙钟之和{audit['summed_family_wall_seconds']/3600:.3f}小时，不能当作实际并行批次耗时。"
        f"全量审计核查{audit['input_evidence_bytes_verified']:,}字节，耗时{audit['audit_elapsed_seconds']:.1f}秒。",
        f"原冻结记录内R1软失败{len(audit['R1_soft_failed_folds'])}折、K软失败{len(audit['K_soft_failed_folds'])}折；另有2次整家族失败。"
        '这些是不同层次的计数，不能当作独立样本数。','',
        '下一轮唯一优先问题：在可迁移工作模型及有限校准信息下，建立更有依据的参数不确定性处理，减少不必要保守性，'
        '并同步提供给强简单基线。先检验有效性，再比较Power。是否能获得实质贡献仍未知。',
        '暂时停止：新投影家族、广泛参数扫描、特征函数尺度分支自动恢复、继续给P扩大适用范围，以及无约束协方差漂移下的统一成功承诺。',
        '下一轮停止判据应在新结果前确定：明确复合零假设反例不能修复则停止该候选；新增复杂度只有开发微弱收益且缺少机制证据则不晋升；'
        '任何后续独立确认必须另订有效协议，不使用本批当未见数据、不追加本轮第三次确认。','',
        'E：原R1工程和C1复现已核验；K原C2有两个整次失败；修复仅有10输入回归。',
        'V：已知真实参数推导与估计前端实证严格区分；有限bootstrap不是参数覆盖证书。',
        'P：C1同保护条件优势成立，但H/all-information强基线优势不成立；C2仅描述性。',
        'N：Gaussian解析MLE和成功起点选择是已知数学/工程修复，不是新算法贡献。与moderated-t、PC、eBH等关系见主研究材料；未认证首创或发表。','',
        '## 7. 可复现性与交付边界','',
        '本目录由`scripts/robustness_r2_closeout_report.py`生成；使用`--out <新的项目内目录>`重新生成，不覆盖原输出。'
        '机器可读证据见manifest中的SHA绑定路径；原逐次输入、e值、失败traceback、源码与协议完整保留。',
        '原成功批次专用C2分析/回放/图表未运行，因为前提不满足；它们未被强行绕过。'
        '本描述性报告不等于原C2确认报告，也没有形成修复版的新独立确认。',
        '本轮完成封包后停在决策检查点；不是继续后台搜索、临床验证、论文录用或招生效果承诺。']
    return '\n'.join(lines)+'\n'


def figures(audit,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white','svg.hashsalt':'sca3-failed-C2-fixed'})
    strata=['core54','normal27','t5_27'];labels=['All 54','Normal 27','Student t5 27']
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for j,s in enumerate(strata):
        v=audit['strata'][s]['I'];point=100*v['mean_difference']
        axes[0].plot(point,j,'o',color='#155E75')
        axes[0].annotate(f"{point:+.3f} (MCSE {100*v['paired_mcse']:.3f})",(point,j),xytext=(7,7),textcoords='offset points',fontsize=9)
    axes[0].set(yticks=range(3),yticklabels=labels,xlabel='Paired Power difference (percentage points)',title='K minus frozen R1 | descriptive, not confirmed')
    axes[0].axvline(0,color='#777777',lw=1);axes[0].margins(x=.6,y=.35)
    families=['H','F_safe_same_K','F_empirical_same_K','F_all_same_information']
    for j,f in enumerate(families):
        point=100*audit['comparisons']['core54/'+f]['sample_envelope_difference']
        axes[1].plot(point,j,'o',color='#7C3AED')
        axes[1].annotate(f'{point:+.3f}',(point,j),xytext=(7,5),textcoords='offset points')
    axes[1].set(yticks=range(4),yticklabels=['Historical H','Same-K safe','Same-K empirical','All-information simple'],
        xlabel='Sample envelope Power difference (percentage points)',title='Strong-baseline gaps | no confirmation intervals')
    axes[1].axvline(0,color='#777777',lw=1);axes[1].margins(x=.25,y=.2)
    for ax in axes:ax.grid(axis='x',alpha=.2)
    fig.suptitle('FAILED C2: all 82,800 attempts retained | simulation, not patient data',fontsize=13)
    for ext in ['png','svg']:fig.savefig(out/f'descriptive-competition.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)
    p=audit['plan'];k=p['candidate'];r1=p['reference'];plugin='R1B_plugin_pilotc0.5_projection_gate_eBH'
    groups=[('Original 68 working cases',p['original_validity_scope_indices']),
        ('Expanded 8 matched cases: incomplete case80/81 shown as ranges',p['expanded_matched_indices']),
        ('Retained assumption violations',[i for i in range(76) if i not in p['original_validity_scope_indices']])]
    fig,axes=plt.subplots(3,1,figsize=(13,10.5),layout='constrained')
    for ax,(label,indices) in zip(axes,groups):
        ymax=6
        for m,tag,color,offset in [(r1,'R1','#64748B',-.18),(k,'K','#155E75',0),(plugin,'P/plugin','#C2410C',.18)]:
            for x,i in enumerate(indices):
                row=audit['rows'][i];v=row['methods'][m]
                if row['n_failed']:
                    lo,hi=100*np.array(v['fdp_sample_completion_bounds_not_CI'])
                    ax.vlines(x+offset,lo,hi,color=color,lw=2)
                    ax.plot(x+offset,(lo+hi)/2,'x',color=color,ms=5)
                    ymax=max(ymax,hi*1.12)
                else:
                    point=100*v['fdp'];ax.plot(x+offset,point,'o',color=color,ms=3)
                    ymax=max(ymax,point*1.12)
            ax.plot([],[],'o',color=color,label=tag,ms=4)
        ax.axhline(5,color='#991B1B',ls='--',lw=1,label='Nominal 5%')
        ax.set(xticks=range(len(indices)),xticklabels=[str(i) for i in indices],xlabel='Frozen case index',
            ylabel='Mean FDP (%)',title=label,ylim=(-.3,ymax))
        ax.tick_params(axis='x',labelsize=6 if len(indices)>20 else 10)
        ax.grid(axis='y',alpha=.2);ax.legend(ncol=4,fontsize=9,loc='upper right')
    fig.suptitle('Descriptive FDR estimates, not confidence bounds | missing ranges are algebraic only',fontsize=12)
    for ext in ['png','svg']:fig.savefig(out/f'descriptive-fdr.{ext}',dpi=180,metadata={'Date':None} if ext=='svg' else {})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to((ROOT/'artifacts/robustness').resolve()) or out.exists():
        raise ValueError('A NEW directory inside artifacts/robustness is required')
    base=ROOT/'artifacts/robustness'
    inputs={label:base/path for label,path in {
        'audit':'R0076-failed-audit/summary.json','repair':'R0076-post-C2-numeric-regression-v2/summary.json',
        'C1':'R0073-confirmation-analysis.json','protocol':'R0076-screen.protocol.json',
        'index':'R0076-results-index.json','frozen_replay':'R0076-failed-batch-frozen-replay.json'}.items()}
    hashes={key:sha(path) for key,path in inputs.items()}
    audit,repair,c1=read(inputs['audit']),read(inputs['repair']),read(inputs['C1'])
    validate(audit)
    replay=read(inputs['frozen_replay'])
    if not c1['complete'] or not replay['complete'] or replay['independent_confirmation_complete'] or replay['post_C2_repair_used']:
        raise ValueError('Original C1 and separate frozen failed-batch engineering replay required')
    if replay['audit_sha256']!=hashes['audit']:
        raise ValueError('Engineering replay does not bind this failed-batch audit')
    if audit['protocol_sha256']!=hashes['protocol'] or audit['original_index_sha256']!=hashes['index']:
        raise ValueError('Audit is not bound to original failed C2')
    for row in audit['rows']:
        if sha(row['array_path'])!=row['array_sha256']:raise ValueError('Audit array changed')
    if not repair['complete'] or repair['independent_confirmation']:raise ValueError('Separate completed DEVELOPMENT regression required')
    if len(repair.get('patched_aliases',[]))!=3:raise ValueError('Full three-alias repair regression required')
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'generate_source.py')
    text=render(audit,repair,c1)
    text+=f"\n补充工程复现：按原冻结选择规则，{len(replay['receipts'])}个原成功输入的124个e数组、24个p数组、评分及fresh K回退全部精确一致。"
    text+='原两个整次失败原样保留；这不是补齐确认，也未使用修复版。\n'
    (out/'README_ZH.md').write_text(text,encoding='utf-8')
    figures(audit,out)
    if any(sha(path)!=hashes[key] for key,path in inputs.items()):raise ValueError('Source evidence altered during rendering')
    manifest={'kind':'FAILURE_PRESERVING_DESCRIPTIVE_CLOSEOUT_NOT_C2_CONFIRMATION',
        'generated_utc':datetime.now(timezone.utc).isoformat(),'inputs':{k:{'path':str(v),'sha256':hashes[k]} for k,v in inputs.items()},
        'script_sha256':sha(__file__),'generated_files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'no_new_samples':True,'no_new_formal_intervals':True,'main_visual_review':'PENDING_SEPARATE_RECEIPT'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':main()
