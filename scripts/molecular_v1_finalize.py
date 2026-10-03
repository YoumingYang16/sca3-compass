"""Finalize an actually reviewed package. No new scientific observations."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import shutil
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from sca3_compass.robustness_io import write_json
from sca3_compass.robustness_registry import update_registry
from molecular_v1_package import read,sha,percent,pp


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True);args=parser.parse_args()
    package=args.package.resolve();review=read(args.review);summary=read(package/'evidence/confirmation.json')
    if review.get('scientific_review_passed') is not True:
        raise ValueError('Actual main evidence review required')
    if review['confirmation_sha256']!=sha(package/'evidence/confirmation.json') or review['replay_sha256']!=sha(package/'evidence/replay.json'):
        raise ValueError('Review does not cover actual evidence')
    if not read(package/'actual-command-verification.json')['passed'] or not read(package/'verification.json')['scientific_pass']:
        raise ValueError('Actual package call and scientific audit must pass')
    receipt=read(package/'release.json')
    if receipt['packaging_status']!='PENDING_ACTUAL_CLI_AND_FINAL_REVIEW':
        raise ValueError('Already finalized or unexpected package status')
    shutil.copy2(args.review,package/'FINAL_REVIEW.json')
    shutil.copy2(package/'release.json',package/'release-provisional.json')
    receipt.update(packaging_status='COMPLETE',final_review_sha256=sha(package/'FINAL_REVIEW.json'),
                   completed_utc=datetime.now(timezone.utc).isoformat())
    # The canonical receipt stays PENDING while the actual final CLI is checked.
    # The draft's identical bytes can only be promoted after that check passes.
    candidate_receipt=package/'release-final-candidate.json'
    write_json(candidate_receipt,receipt)
    command=read(package/'actual-command.json')['command']
    command[command.index('--output')+1]=str(package/'example/output-final.json')
    command[command.index('--validation-receipt')+1]=str(candidate_receipt)
    result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
    write_json(package/'actual-final-command.json',{'command':command,'returncode':result.returncode,
        'stdout':result.stdout,'stderr':result.stderr})
    if result.returncode:raise RuntimeError('Final receipt-bound CLI failed')
    final=read(package/'example/output-final.json');old=read(package/'example/output.json')
    if final['validation_receipt_sha256']!=sha(candidate_receipt):
        raise ValueError('Example receipt hash is stale')
    if any(final[k]!=old[k] for k in ['discoveries','p_values','e_values','adjusted_p_values']):
        raise ValueError('Metadata finalization altered inference')
    write_json(package/'release.json',receipt)
    scope=[r for r in summary['scene_rows'] if r['scope']=='D1']
    pairs=summary['paired'];rows=[]
    for label,key in [('整体','core'),('普通','normal'),('t5重尾','t5')]:
        v=pairs[key]['comparisons']['B_fair_conditional_eBH']
        rows.append(f"|{label}|{percent(pairs[key]['power']['K_NR'])}|{percent(pairs[key]['power']['B_fair_conditional_eBH'])}|{pp(v['mean_difference'])}|[{pp(v['lower'])}, {pp(v['upper'])}]|")
    worst=max(scope,key=lambda r:r['methods']['K_NR']['fdr_interval'][1])
    drift=next(r for r in summary['scene_rows'] if r['index']==67)
    raw=pairs['core']['comparisons']['B_original'];strong=pairs['core']['comparisons']['B_strong']
    strong_worst=max(scope,key=lambda r:r['methods']['B_strong']['fdr_interval'][1])
    decision='\n'.join([
        '# 第一版交付决策：K-NR-1.0.0','',
        '**限定模型研究版V1通过本轮冻结的实证验收；证据级别EMPIRICAL_ONLY。**',
        '不是已证明的有限校准FDR保证版，不是通用真实患者部署版，也不是原创性或录用证明。原C2失败未改判。',
        '',f"R0078固定独立确认：{summary['attempts']:,}个新模拟家族、84既有场景，其中76个D1场景；没有新参数域或患者队列。一次确认，无追加样本/备用确认。",
        '', '|层|K Power|同校准普通PC/eBH Power|差值（百分点）|本轮联合区间（百分点）|',
        '|---|---:|---:|---:|---|',*rows,'',
        f"对历史原普通高斯管线Bonferroni-PC-BY整体差{pp(raw['mean_difference'])}点；旧版重尾校准有问题，不能单独用它证明公平优势。验收使用同信息/同条件尺度的增强普通PC/eBH及普通PC/BY，二者均需逐场景错误控制和收益通过。",
        '早期直接Student边际修复和边际bootstrap保护没有自动获得有效性；DEV失败/不确定性及全部结果保留。公平对照在正式确认前按公开修订采用已有条件普通检验。此次通过不等于补过旧边际修复的失败。',
        '',f"候选最大逐D1场景FDR联合上界{percent(worst['methods']['K_NR']['fdr_interval'][1])}（case{worst['index']}）；每一D1场景均需上界<=5%，不是跨场景平均掩盖失控。",
        f"对预选强简单参照整体差{pp(strong['mean_difference'])}点；公开此结果，不宣称研究最优。参照自身FDR见完整表。",
        f"强参照case{strong_worst['index']}的FDR点估计{percent(strong_worst['methods']['B_strong']['fdp'])}、联合上界{percent(strong_worst['methods']['B_strong']['fdr_interval'][1])}，未通过整个D1的错误率验收；这不等于该点估计已证明总体FDR>5%。",
        '强参照在case56、82、83具有联合区间支持的局部Power优势，且在这三个场景自身FDR上界通过。不能用它在别的场景未通过验收来否认这些局部劣势。具体数值见FINAL_REVIEW.json及evidence/confirmation.json。',
        f"严重漂移case67本次FDR={percent(drift['methods']['K_NR']['fdp'])}，仍不支持。历史R0076的74.7393%保留，绝不称任意漂移下重尾问题已解决。",
        '', 'COMPARISON.md包含局部差异、联合区间、全部84场景、失败/回退和成本。没有显著局部损失不等于逐场景非劣。新授权报告预算不覆盖全部历史选择。',
        '算法采用已有K并整合数值修复，不包装为原创突破。本轮在冻结交付检查点停止，不自动进入更高性能、一般证明或论文创新研究。',''])
    (package/'RELEASE_DECISION.md').write_text(decision,encoding='utf-8')
    example_seed=command[command.index('--seed')+1]
    cli=f'.venv/Scripts/python.exe "{package}/source/scripts/molecular_v1.py" --input "{package}/example/input.npz" --output "{package}/example/my-output.json" --acknowledge-research-scope --validation-receipt "{package}/release.json" --seed {example_seed}'
    fence=chr(96)*3
    readme='\n'.join(['# K-NR-1.0.0：限定模型研究版','',
        '先读RELEASE_DECISION.md和SUPPORTED_SCOPE.md。证据EMPIRICAL_ONLY；共同位置、独立性、可迁移角度结构与径向模型未被运行时认证。',
        '', '## 实际调用','',
        '项目根目录运行；输出路径必须未存在。已执行命令/退出码见actual-final-command.json。示例为模拟数据，不是患者数据。',
        '',fence+'powershell',cli,fence,'',
        '输入NPZ键z[256,4,6]、calibration[4,n,6]。示例移除truth。输出版本、证据级别、范围/假设、诊断、方向性发现及实际p/e数组。CLI不带验收凭据仍显示V0状态，不能靠名称晋升。',
        '', '## 复现','',
        'source/为R0078冻结源码；release.json逐文件SHA核验。evidence/protocol.json记录实际环境、种子、案例和固定次数。全部逐次输入/结果保留在项目artifacts/robustness/R0078-v1-repetitions；evidence有84逐场景指标数组。',
        '项目根目录使用原归档脚本，--out为新的不存在路径：','',fence+'powershell',
        '.venv/Scripts/python.exe artifacts/robustness/R0078-v1-source/scripts/molecular_v1_validation.py analyze --project-root . --run R0078 --out artifacts/robustness/R0078-independent-reanalysis',
        '.venv/Scripts/python.exe artifacts/robustness/R0078-v1-source/scripts/molecular_v1_validation.py replay --project-root . --run R0078 --out artifacts/robustness/R0078-independent-replay.json',
        fence,'',
        '只重算同一批次，不是新增确认。精确回放在本机固定依赖验证，其他平台不保证逐位一致。COMPARISON.md由package_source.py的tables函数从evidence/confirmation.json确定性生成；verification.json记录均值/区间/配对界的重算。',
        '', '## 状态与边界','',
        'COMPUTED、COMPUTED_WITH_DECLARED_FOLD_FALLBACK、UNSUPPORTED_INPUT、NUMERICAL_FAILURE、INSUFFICIENT_EVIDENCE有区别。正常计算不证明真实生成过程符合D1。',
        'bootstrap不是置信域，有限实验不是一般保证。严重漂移仍失控。未临床认证、未解封新队列、未写/提交论文。原失败、边际修复失败和工程调用错误全部保留。',''])
    (package/'README.md').write_text(readme,encoding='utf-8')
    env=read(package/'evidence/protocol.json')['environment']
    (package/'requirements-research.txt').write_text(
        f"numpy=={env['numpy']}\nscipy=={env['scipy']}\nthreadpoolctl==3.6.0\n",encoding='utf-8')
    shutil.copy2(__file__,package/'finalize_source.py')
    manifest={p.relative_to(package).as_posix():sha(p) for p in sorted(package.rglob('*')) if p.is_file()}
    write_json(package/'MANIFEST.json',{'files':manifest,'method_version':'K-NR-1.0.0',
        'kind':'FINAL_RELEASE_PACKAGE','completed_utc':datetime.now(timezone.utc).isoformat()})
    archive=package.parent/(package.name+'.zip')
    if archive.exists():raise FileExistsError('Archive already exists')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for file in sorted(package.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(package.parent).as_posix())
    registry=ROOT/'artifacts/robustness/EXPERIMENT_REGISTRY.json'
    registry_backup=ROOT/'artifacts/robustness/V1-reference/registry-before-release.json'
    if registry_backup.exists():raise FileExistsError('Release registry backup already exists')
    shutil.copy2(registry,registry_backup)
    entry=next(e for e in read(registry)['experiments'] if e['id']=='R0078')
    entry.update(independent_confirmation_complete=True,scientific_acceptance='SCOPED_EMPIRICAL_V1',
        release_package=str(package),release_manifest_sha256=sha(package/'MANIFEST.json'),
        release_archive_sha256=sha(archive),evidence_level='EMPIRICAL_ONLY',
        general_finite_calibration_guarantee=False,original_R0076_reclassified=False)
    update_registry(registry,entry)
    print(package,sha(archive))


if __name__=='__main__':main()
