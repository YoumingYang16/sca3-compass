# K-NR-1.0.0：限定模型研究版

先读RELEASE_DECISION.md和SUPPORTED_SCOPE.md。证据EMPIRICAL_ONLY；共同位置、独立性、可迁移角度结构与径向模型未被运行时认证。

## 实际调用

项目根目录运行；输出路径必须未存在。已执行命令/退出码见actual-final-command.json。示例为模拟数据，不是患者数据。

```powershell
.venv/Scripts/python.exe "C:\Users\ROG\Documents\Codex\2026-09-11\plugin-browser-openai-bundled-x20\sca3-compass\releases\K-NR-1.0.0/source/scripts/molecular_v1.py" --input "C:\Users\ROG\Documents\Codex\2026-09-11\plugin-browser-openai-bundled-x20\sca3-compass\releases\K-NR-1.0.0/example/input.npz" --output "C:\Users\ROG\Documents\Codex\2026-09-11\plugin-browser-openai-bundled-x20\sca3-compass\releases\K-NR-1.0.0/example/my-output.json" --acknowledge-research-scope --validation-receipt "C:\Users\ROG\Documents\Codex\2026-09-11\plugin-browser-openai-bundled-x20\sca3-compass\releases\K-NR-1.0.0/release.json" --seed 254564622
```

输入NPZ键z[256,4,6]、calibration[4,n,6]。示例移除truth。输出版本、证据级别、范围/假设、诊断、方向性发现及实际p/e数组。CLI不带验收凭据仍显示V0状态，不能靠名称晋升。

## 复现

source/为R0078冻结源码；release.json逐文件SHA核验。evidence/protocol.json记录实际环境、种子、案例和固定次数。全部逐次输入/结果保留在项目artifacts/robustness/R0078-v1-repetitions；evidence有84逐场景指标数组。
项目根目录使用原归档脚本，--out为新的不存在路径：

```powershell
.venv/Scripts/python.exe artifacts/robustness/R0078-v1-source/scripts/molecular_v1_validation.py analyze --project-root . --run R0078 --out artifacts/robustness/R0078-independent-reanalysis
.venv/Scripts/python.exe artifacts/robustness/R0078-v1-source/scripts/molecular_v1_validation.py replay --project-root . --run R0078 --out artifacts/robustness/R0078-independent-replay.json
```

只重算同一批次，不是新增确认。精确回放在本机固定依赖验证，其他平台不保证逐位一致。COMPARISON.md由package_source.py的tables函数从evidence/confirmation.json确定性生成；verification.json记录均值/区间/配对界的重算。

## 状态与边界

COMPUTED、COMPUTED_WITH_DECLARED_FOLD_FALLBACK、UNSUPPORTED_INPUT、NUMERICAL_FAILURE、INSUFFICIENT_EVIDENCE有区别。正常计算不证明真实生成过程符合D1。
bootstrap不是置信域，有限实验不是一般保证。严重漂移仍失控。未临床认证、未解封新队列、未写/提交论文。原失败、边际修复失败和工程调用错误全部保留。
