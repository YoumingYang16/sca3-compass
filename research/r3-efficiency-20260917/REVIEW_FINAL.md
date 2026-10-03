# R3 最终有界证据审查

2026-09-17。唯一独立审查者。只读核对实际结果、冻结协议/源码、已有重放/测试/诊断收据及已完成草稿；只编辑本文件。没有运行方法、拟合、模拟、测试、重放、重新计算区间或新增统计比较，没有增派代理。

## 最终科学结论

**限定模型内：保留完整 grade A / 理想算术有限样本保证，并确认相对 R2 的效率增益。完整用户目标：未达到。**

建议最终状态采用 `R3_RESULTS.md:25-28` 已写的：

    MODEL_VALID_WITH_CONFIRMED_EFFICIENCY_GAIN
    FULL_R3_CRITERIA_NOT_MET

不能标无条件 `PROVED_AND_USEFUL`：两个预指定漂移场景均有同时区间支持的 FDR 恶化，直接违反用户第 15 条；贡献新颖性仍未建立。也不应标 `POWERFUL_BUT_UNPROVED`，因为这会错误抹去已成立的限定模型证明。无需另开方法、漂移救援或确认；应保留正负结果并按原范围收尾。

未发现使本次 C001 限定数值结论失效的新计算/统计阻塞。下文列出封版时必须保持的解释及一处具体文档更正。尚在编写的最终状态/中文报告不被当作缺失科学证据，也不预先视为已经审查通过。

## 1. 本次实际核对了什么

路径均相对于 `research/r3-efficiency-20260917/`；JSON 位置以下用精确字段路径表示。

- `C001/freeze.json` 所列 12 个源码/测试/协议文件均重新核对哈希，无差异；模型、区间、执行器与分析器匹配最终预审快照。读取了冻结协议和 `C001/analyze_confirm.py`，沿用已完成的模型/依赖链理论审查，不声称用模拟重新证明定理。
- `C001/index.json` 为 complete，15360 条、15360 个唯一 `(case,rep)`，精确等于 15×1024 的预定笛卡尔集合；全部 status=completed。`summary.json` 的 freeze/index/protocol 哈希匹配实际文件。
- `C001/summary.json` 确实包含 289 个双侧区间、cap384、联合覆盖下限 .975；三个 M0 门为 true，两个漂移场景均为 `WORSENING_DETECTED`，`full_user_success_excluded_by_drift=true`。
- **`RECOMPUTED_SUMMARY.json` 已实际出现并核对**：与原 summary 均为 158506 bytes，SHA256 完全相同。已读 `recompute_summary.py:18-29`：它载入冻结分析器，只重定向最后输出，分析前仍执行 `audit_rows`，不重跑方法、不改区间。这里确认的是主任务既有重算产物字节一致；本审查没有再次运行该重算。
- `REPRODUCTION_AUDIT.json` 记录 cases0/1/5/10/14 的 rep0，共 109 个数组精确重放。已读 `reproduce.py:17-44` 的比较范围，并独立核对这五个原始记录与 index/receipt 的哈希、规范路径、任务标识、公式生成的算法种子、freeze、输入/证据哈希和 NPZ 成员数量（21/21/21/21/25）。本次没有重新比较或重新生成 109 个数值数组；精确重放结论来自该身份绑定收据及其源码。
- `frozen-tests.xml` 实际记录 15 tests、0 failures/errors/skipped，10.281s，类名位于 `C001` 冻结测试。测试是正确性证据，不是新增独立确认。
- `IMPLEMENTATION_DIAGNOSTICS.json` 与相同 freeze/index 绑定；`implementation_diagnostics.py:23-35` 对已有记录检查哈希、四折计数和布尔标志。15360 家庭、61440 折记录的 direction/PILOT/scale-ablation 三类未收敛标志均为零。没有本次重复全量扫描，也不能从这三个标志推断所有可能数值问题均不存在或优化取得全局最优。
- `EXAMPLE_RECEIPT.json` 的输入哈希/freeze 已核对：实际固定 case2/rep0 保存输入、辅助 seed22、completed、**0 discoveries**。`run_example.py:19-23` 仅读取 z/cal 供推断；保存文件中的 truth 并未传给模型。该示例不是原算法种子的 C001 决策重放，不增加独立 n；保留零发现，不选种子美化。
- R2 delivery manifest、原 C001 index/freeze/protocol，以及 V1 release ZIP 的身份哈希仍匹配预审/归档锚点。没有重做 V1/R2 科学审计，也未改动其文件。

完整原始 schema/评分/eBH 审计由已读冻结 `r3_experiment.py:173-244` 及实际分析/重算执行；本次独立检查的是冻结身份、完整任务集合、所有关键汇总和五例原始身份链，不声称独立重算了每一个家庭的指标。

## 2. M0 三门与全部局部配对结果

核心位置：`C001/summary.json/core_paired_improvement`、`empirical_gates`；逐场景位置 `rows[case].paired_power.PB_grid` 与 `rows[case].methods.R3_main.fdp`。所有下表差值/区间单位为 **百分点（pp）**，不是相对百分比。

核心六场景等权平均增益为 **+6.894339767 pp，97.5% 同时区间 [5.460835587, 8.200966993] pp**。其 n=1024 是预定六场景组成的独立组，不是把 6144 个不同分布家庭误当同分布样本。核心平均额外正确签名发现 **3.51611328125/家庭**，可由六个 `mean_tp` 差直接复核（每核心家庭 51 个真备择）。

| M0 非零场景 | R3-minus-R2 Power | 已注册同时区间 |
| --- | ---: | ---: |
| 0 normal/N32 | +9.662 | [6.494,13.049] |
| 1 t5/N4 | +3.594 | [1.380,5.965] |
| 2 t5/N12 | +5.946 | [3.503,8.447] |
| 3 t5/N32 | +8.006 | [5.784,10.241] |
| 4 t5/N128 | +7.320 | [5.598,9.064] |
| 5 t3/N12 | +6.838 | [4.644,9.134] |
| 6 continuous | +3.949 | [2.531,5.296] |
| 7 dense90 | +3.333 | [1.921,4.687] |
| 11 effect2.625 | +6.834 | [4.566,9.159] |
| 12 t1.5/N12 | +2.734 | [1.079,4.393] |

十个下端全为正，强于预设逐场景 >=-2pp 的非劣要求；没有局部 M0 损失被核心平均掩盖。这只支持这些固定场景的平均 Power 差，不是每个随机家庭的逐路径占优或对任意 M0 参数一致提高。

case12 的精确增益为 .02734375、区间 [.010792051238747824,.04393181533942503]；R3 Power 约9.773%，R2约7.039%。因此无限方差、有限均值的指定 t1.5 场景有确认增益，但绝对检出能力依旧低。这里 mean 的条件是共同径向乘子的可积性（若用 sqrt(V) 表示乘子，即 E sqrt(V)<infinity），不是要求 E V 有限。

13 个 M0 的主方法 FDR 上端全部低于5%。最大均值为 **0.48828125%**，最大同时上端为 **2.469514209888235%**，均在 case9 singleton。FDR 是完整家庭 FDP 的期望/模拟均值，不是每个家庭 FDP 都 <=5%；有限场景验收不把模型证明扩大成分布自由结论。

`figures/RESULT_TABLES.md` 的这十行以及 partial-null/FDR 表与实际 summary 对应。case8/9/10 无真备择，Power 保持 undefined；没有用零填补后参与 Power 平均。

## 3. 组件归因：哪些已确认，哪些没有

位置：`summary.rows[case].paired_power.{R3_scale,R3_median,R3_ordinary}`；`figures/RESULT_TABLES.md` 的 2×2 表；已读 `R3_RESULTS.md` 的组件章节。

- **full-minus-scale-only：10/10 M0 非零场景下端 >0。** 支持“独立 DIR/SHAPE 架构包在仅改变尺度之外有增量价值”。该包同时改变训练样本量、学得方向与参考 law；不能称纯粹去掉 lambda_min 一步的独立因果效应。
- **full-minus-orientation+median：10 个点差均正，但只有 case5 t3/N12 下端 >0。** 该场景 +2.24609375pp，区间 [0.16605299,4.34665586]pp。normal 的区间约[-1.591,4.790]pp、t5/N4约[-1.070,3.529]pp、t1.5约[-0.969,2.292]pp均跨零。不能称几何替代在所有场景，特别是小 N，获得了独立确认的额外 Power 优势。
- **full-minus-R3 ordinary：8 个非零场景支持增加。** continuous 为 -0.001915pp，区间[-1.153209,1.149364]pp；dense90为+0.077700pp，区间[-1.074873,1.228481]pp，二者增益未解决。相应 gamma=0 的签名-折选择比例为89.6728515625%和84.1796875%，是普通组件选择，不是数值失败。
- 2×2 交互和 core 组件均值只是描述性算术；没有新交互显著性检验、也不能把不同组件收益加成“全部 Power 损失分解”。几何 log 矩改善是解析事实，不自动证明它是这些点差的原因。
- t5 同设定 N4/12/32/128 曲线支持小校准时有改善，但增益点值不是 N4 最大（3.594pp vs N32 的8.006pp）。不得把重点场景改写成“最受益场景”，也不选择 N 或 primary。

本节不改变冻结候选；不能因几何增量的部分区间跨零就事后改用 median 版本。

## 4. Strong 参照及部分合取零假设

位置：`summary.rows[case].paired_power.B_strong`，以及 cases8/9/10 的 `methods.B_strong.fdp`。

R3 在9个 M0 非零场景仍显著低于 Strong；唯一反向场景是 continuous，R3-minus-Strong =+2.510340pp，区间[0.771508,4.225636]pp。M0 非零场景的 Strong 缺口仍大：例如 normal约-37.837pp，t5/N4约-35.145pp，t1.5约-11.627pp，相应区间均在零下。这里 t1.5 是 case12，不属于六核心 cases0–5。

Strong 的实际零假设表现必须原样保留：

| 场景 | Strong FDR 均值 % | 同时区间 % |
| --- | ---: | ---: |
| global null（8） | 3.515625 | [1.333959,7.209325] |
| singleton（9） | 5.761719 | [2.772852,10.200963] |
| mixed-sign null（10） | 2.246094 | [0.633135,5.425211] |

三者都不能通过同一“上端<=5%”经验资格门；但其区间均包含5%，**不能声称本次已证明 Strong 真 FDR>5%**。它缺乏相同完整有限校准证明，属于此目标下的经验参照；这也不意味着其 M0 非零场景的较高 Power 可以被全盘归因于 FDR 超标。不得把剩余 Strong 差距称为有限有效性的必然价格。

## 5. 第 15 条：实际失败，不是仅仅不确定

精确来源：`summary.rows[13/14].paired_boundary_fdr.unprotected`；`boundary_status`。单位仍为 pp。

| 原始漂移 | R3-minus-R2 FDR | 已注册同时区间 | 结论 |
| --- | ---: | ---: | --- |
| 13：kappa ratio2.2 | +1.420061554 | [0.198344224,2.612948411] | 检出恶化 |
| 14：kappa ratio4.6 | +3.666894981 | [2.270841159,5.001258790] | 检出恶化 |

中度漂移 R3 FDR=4.845600%，绝对区间[4.102085,6.077279]%；点值低于5%并不能通过5%上端验收，亦不抵销对 R2 的配对恶化。它本身也不是“已证明绝对 FDR>5%”。严重漂移 R3 FDR=20.008372%，区间[18.425918,21.578339]%，明确远高于5%。

默认方法因此直接未满足用户第 15 条。漂移下高 Power 不能算作可接受的效率成功；不能删除场景、用 Delta 替换默认结论或追加救援确认。

Delta5 是预设的、依赖外部 M_Delta 前提的敏感性输出：R3 在两场景 Power 约4.140%/31.687%，FDR上端约1.154%/1.403%。两项 Delta 配对 FDP 差区间均跨零，不能宣称相对 R2 Delta 的非增大已经证明。它只说明一个已给定上界下的不同保护强度，不能满足已失败的默认方法要求，也不能由此推荐从数据中选择 Delta。

## 6. 用户八项成功条件与额外第 15 条

| 用户要求 | 本次证据判断 |
| --- | --- |
| 1. 保持 R2 同级有限校准保证 | 是，限定 M0/理想算术；不是一般真实基因数据保证 |
| 2. 证明覆盖完整算法 | 是，既有 DIR/SHAPE/PILOT/PC/grid/eBH 链条及最终冻结实现对应 |
| 3. 独立确认 FDR 达预定门 | 是，13 个 M0 上端均通过；漂移另外判定 |
| 4. 相对 R2 可信且实际有意义的效率增益 | 限定模拟任务内支持：核心+6.894pp、约+3.516正确签名发现；不是临床效用证据 |
| 5. 主要小校准/重尾不新增关键退化 | 支持，10个非零 M0 配对下端均正，包括 t1.5；不保证所有未知参数下无退化 |
| 6. 缩小 Strong 差距或说明剩余必要代价 | 对这些配对场景差距缩小有支持；九场景仍落后，剩余必要代价未获 Power 下界证明 |
| 7. 合理复杂性与计算成本 | 没有增加观测/校准需求，额外训练/参考计算成本有限且有记录；现有 bundle 计时不能精确隔离 primary-only 开销 |
| 8. 可核查的新颖贡献 | **未建立**；一般独立分割/invariance/有限 pivot 原理既有，具体贡献优先权尚未解决 |
| 第15条：漂移不更差 | **失败**，两个配对恶化区间下端均正 |

`CONTRIBUTION_MAP.md:33-41` 和 `EFFICIENCY_LITERATURE_MATRIX.md:46-67` 的审慎定位应保留。本轮没有另做文献路线；没有读到 CFAR2010 全文，不能用出版方摘要或本地实现不同来宣布其定理不适用/本方法优先。已有摘要先例足以约束一般 invariance 新颖性宣传，却不足以证明完全等价或完成优先权排除。

## 7. 成本、失败及诊断范围

`summary.rows[*].runtime` 显示 M0 的 R3 bundle 均值约1.291–2.139s、R2 bundle约.959–1.639s；逐场景 bundle 比约1.305–1.358。可以报告研究实现的有限额外计算负担，不能将它冒充独立 primary-only 速度比较。index 的 `seconds_current_invocation` 为7253.9558219s；末个 progress 计时与 index 封尾可能稍有差异，完整运行约120.9分钟。

R3/R2 数值失败均0/15360，Delta调用无失败；**V1 顶层 fallback 共2次，case8与case12各1次**，均保留在指标中。三个 R3 学习收敛标志全为 true，并不等于 V1/R2 的所有内部角色都被审计或所有方法零 fallback。`IMPLEMENTATION_DIAGNOSTICS.json.limitations` 已正确指出这点。

R3 的无失败/精确重放是工程证据，不是通用浮点 FDR 认证；非零 gamma、gamma=0 和 optimizer 不收敛是不同事件。单例0发现保留，不能为了演示效果再挑另一个 seed。

## 8. 封版前的具体文字事项（不要求新实验）

1. **修正 `R3_CLAIM_EVIDENCE.md:12` C5 的 `CRB>2/N`。** 正确是“CRB = 2/N；在所述全 eta 无偏 regular 类中，固定有限 N 的 Var_eta(T)>2/N，等号不可达”。不可达并未把 CRB 数值提高，也未证明一个统一正间隙。这是文档错误，不影响模拟或方法证明。
2. 将仍标 PENDING 的 C7/C8/C10 等条目按实际 summary、重放、重算收据完成；C9 必须明确默认漂移非恶化要求失败，不可把它和 Delta 表现合成一个 pass。主任务尚在编写的文件无需在完成前被视为审查失败。
3. 本次实际读到的 `R3_RESULTS.md`（SHA256见末尾）已正确披露 M0收益、几何增量区间、Strong/partial-null 限定、两个漂移失败、t1.5低绝对Power、V1两次fallback、计时含ablation及零发现例子。其数值解释没有发现实质性错误。`RESEARCH_FINDINGS_ZH.md`/`FINAL_STATUS.json` 若尚未完成，本审查不声称已核验其最终文字；它们应保持同一限定状态。
4. 论文摘要不得把“scoped model A + confirmed efficiency gain”缩写成用户完整成功；不得把信息 CRB、已知形状的矩阈值或CFAR摘要推成全算法最优性、新颖性或临床结论。原始冻结源码/协议/阈值/样本量不改。

上述文档整合完成后，可封存这一有正负结论的限定证据包；不需要再确认、拟合或扩展研究。审查通过证据一致性，不等于判定用户研究目标全部完成。

## 核对锚点

```text
C001/summary.json == RECOMPUTED_SUMMARY.json
85bbb0fae69f6a45b6d96c82e6c40727914b74be1b6a29131169b661fee64697
C001/index.json
0f7423b20278321a5286d3326d548fa6adb65c9ae2c2d158a0b2d5b713339b1d
C001/freeze.json
4ef7e448c1ee6ee6064a0e31d53e6f36e21b9825e6a7920ef82aebcee9e0b620
C001/protocol.json
eedeeb64b5eec4f52a00043a233d7cf7d7d8a07cbe2668b13d26d95d78043e46
REPRODUCTION_AUDIT.json
0a916d5c94cd1064d9987a0edfbd6087616c0dac2922b91033dfb6d3ef3ffcb5
IMPLEMENTATION_DIAGNOSTICS.json
ab78e9eaacd596f6577299e0204c3db6821ef7d5f27150b56ec6c8cfb799bbf0
EXAMPLE_RECEIPT.json
3d85def0e56906bd8631bbc6efda26188fa1e69a85d0959501c4ee2334be286b
frozen-tests.xml
0c1f2a8122e372e87913ba7b7de99ba2b27711fb081a76b71a5f4473adaef349
figures/RESULT_TABLES.md
8551df4541ef4ddde26573a2029db2d7831ef45eca6b348d4fb583f85e343c1f
R3_RESULTS.md (本次读取版本)
f274c5a79f957d8332267a2fe8c2b0a1294c1562bcfdea7335b359a87a31ca89
R3_CLAIM_EVIDENCE.md (含待改C5的本次版本)
48d0288dfa3a0a0c22a470118bd398366af4901625a115e972476546d3c54e62
```

## 收尾时已出现材料的追加核对

在本审查写入后，`RESEARCH_FINDINGS_ZH.md` 已出现，本人已读其全部130行（SHA256 `1a641a24d6bc22ecd65985204761c8c9ac8a619f434755aaae3c35f507cc21a2`）。其核心/局部结果、Strong限定、组件归因、两个漂移失败、零发现示例和 `FULL_R3_CRITERIA_NOT_MET` 定位均与上述证据一致，没有新数值阻塞。

中文精度建议：`:23` 的“90%高信号”宜写“90%稠密非零信号”，避免把备择比例当效应幅度；`:76` 的“各研究独立径向噪声不在证明中”宜明确为**目标块内**，因为校准块的独立 study 左变换仍在既有矩阵 pivot 允许范围内。这些是文字范围限定，不要求任何新计算。

同次读取的 `R3_CLAIM_EVIDENCE.md` 已把原 PENDING 的确认/漂移/重放条目更新为实际结果，加入组件区间限定；第8节事项2因此已完成。C5 的 `CRB>2/N` 仍在该次快照的第13行，仍需按事项1作精确改写。最终 `FINAL_STATUS.json` 尚未出现，本审查不臆测其内容，不因此否定已经完成的科学证据核验。

## 最终报告与待封存状态的追加审查（2026-09-17）

按主任务最新请求，完整读取六份最终报告：`R3_RESULTS.md`、`RESEARCH_FINDINGS_ZH.md`（含全部11问及新增FDP尾部段落）、`R3_CLAIM_EVIDENCE.md`、`VALIDITY_EFFICIENCY_FRONTIER.md`、`CONTRIBUTION_MAP.md`、`R3_PAPER_CORE.md`。同时读取现已出现的临时 `FINAL_STATUS.json`、更新的 `README.md` 与结果封存辅助脚本 `seal.py`。本节更新前面的“尚未出现”状态，但不修改历史审查文字。

本次只读取实际 summary、已有区间和身份哈希，另核对描述性均值算术；没有执行项目 Python、测试、方法、拟合、重放、重算区间或封存脚本，没有另开代理。**最终报告的主要科学解释通过；没有新增需要科学重跑的阻塞。** 仍需完成下述已知数学措辞修正，不能把该结论解释为完整用户成功。

### 实际交叉核对与解释

- `R3_RESULTS.md` 的全部非零M0表、`VALIDITY_EFFICIENCY_FRONTIER.md:46-70` 的六方法前沿表和核心均值与 `C001/summary.json` 一致。重新核对的核心A/B/C/D均值分别为 **28.6359400531 / 29.9600439134 / 33.9633118873 / 35.5302798203%**，并非新的统计比较。289个双侧区间仍使用cap384及至少97.5%联合覆盖；没有增补新的置信比较。
- `R3_RESULTS.md` 组件段落、中文第3问、`CONTRIBUTION_MAP.md:33-38`、`R3_PAPER_CORE.md:183-189` 均正确区分：10个M0非零场景D−B下端都为正；D−C仅t3/N12下端为正。架构包含DIR/SHAPE样本量与学得方向的改变，不是纯矩阵元素替换的因果效应；几何精确矩性质不推出普遍Power收益。没有把描述性交互或组件均值升级为新增显著性结论。
- `R3_PAPER_CORE.md:146-151` 正确区分t1.5的scatter尺度与不存在的方差，也指出场景间其他参数不同；中文第5/8问没有把重尾增益当纯尾厚度因果效应或声称N4获益最大。t1.5配对增益及低绝对Power均如实保留。
- Strong singleton **5.76171875%，区间[2.7728516323,10.2009630936]%** 与所有最终报告一致：不能通过上端<=5%的验收，但并未证明真FDR>5%。九个非零M0场景仍显著落后Strong的结论保留，未将其较高Power一概归因于超标。
- 两个默认漂移的配对FDP增量区间仍为 **[0.1983442244,2.6129484113]pp** 和 **[2.2708411586,5.0012587899]pp**，均严格为正。英文结果、中文第10问、claim C9、前沿与论文核心一致明确第15条失败；Delta5只作有外部上界前提的敏感性结果，不替换默认算法，也不声称其配对非增大已证实。
- 新增 `R3_RESULTS.md:93-97` 与 `RESEARCH_FINDINGS_ZH.md:80` 的主方法FDP经验p99已逐一核对：t5/N4 **8.8738562092%**、t5/N12 **6.2064393939%**、t3/N12 **6.3523936170%**、t1.5 **8.9166666667%**；显示值8.874/6.206/6.352/8.917正确。主方法cases8/9/10的p99确为0；有发现时这些全PC零假设家庭FDP=1。它们是已有描述分位数，不是总体99%分位数的置信界或逐家庭/逐患者安全保证，报告限定恰当。
- 中文11问均有明确回答，包括未识别损失、不完整归因、剩余Strong差距、计算成本及不可声称的结论。`R3_CLAIM_EVIDENCE.md:20`、`CONTRIBUTION_MAP.md:40-48`、中文第9/11问及论文限制段均保留最近CFAR优先性缺口。没有根据出版方摘要宣称已读全文、排除先例或建立全球首创。
- 原summary与重算文件SHA256仍均为 `85bbb0fae69f6a45b6d96c82e6c40727914b74be1b6a29131169b661fee64697`。最终文字正确保留109数组/5例、15测试、0发现固定示例、V1两次顶层fallback，以及非全浮点/非全局优化认证。三张PNG的视觉检查是主任务报告，本次没有重复视觉检查，亦不将其视为额外科学证据。

### 必须修正与非阻塞精度项

1. **仍待关闭的文档错误：`R3_CLAIM_EVIDENCE.md:13` C5依旧写 `CRB>2/N`。** 可直接替换为：`CRB = 2/N; for every globally log-unbiased regular estimator in the pivot-only experiment, Var_eta(T) > 2/N at finite N (equality unattainable); NOT a detection-Power/minimax/full-raw-data bound.` 等号不可达不等于CRB数值大于2/N；其他最终报告中的“该估计量类方差严格大于2/N”没有这个错误。此项仅需文档修正，不涉及冻结算法或确认结果。
2. **非阻塞文字精度：** 中文 `:23` 的“90%高信号”及 `:84` 的“高信号”宜改“90%稠密非零信号/高非零比例”，避免混淆非零比例与效应幅度；`:76` 的“各研究独立径向噪声”宜明确为“目标块内各研究独立径向噪声”，不误排除校准块允许的可逆study左变换。前述建议仍适用。
3. **非阻塞命令记述一致性：** `README.md:62-65` 作为“ACTUALLY executed”的命令未显示 `-B`，但`:70`说这些调用使用了 `-B`。主任务应按实际命令统一展示或将文字限定到确实使用该标志的调用；不能把等效复现命令冒充逐字命令日志。这不改变已有输出身份或统计结论。

### 临时状态与封存边界

已实际读取的 `FINAL_STATUS.json` 是 `FINAL_REVIEW_BEFORE_SEAL`，`bounded_phase_complete=false`、`internal_final_review_integrated=false`；这正确反映当前待整合状态，不再是缺失文件。其 `full_user_success=false`、`novelty_established=false`、八项条件/第15条、有限M0证明范围、主要数值及限制均与实际证据一致。完成报告修正与封存检查后，只可将前两个工作完成标志按实际进度置true；完整用户成功及新颖性标志仍应为false。

只读检查 `seal.py:23-31`，新增比对确实使用R3冻结记录中的四个R2身份锚点，以及R2冻结依赖中的V1 ZIP哈希；`:35-53` 要求阶段/审查完成、summary字节一致、重放身份正确，并禁止在已检出漂移恶化的情况下宣称完整成功。脚本位于冻结算法外，没有新增推断计算。本审查未执行seal、未替主任务产生manifest，也不因脚本存在而声称封存已经完成。最终manifest及封存操作仍由主任务负责。

**收尾意见：修正C5并统一上述报告文字后，可以整合本审查并封存正负结果俱全的限定证据包；无需再做科学运行。证据审查通过不等于用户完整研究条件通过。**

本次读取的最终报告/辅助文件快照（后续整合文字后由最终manifest记录新哈希）：

```text
R3_RESULTS.md
3779578344a4b28b0242edacd4acb107350b15ea5ee4d5395338b194f2c04bdc
RESEARCH_FINDINGS_ZH.md
035536ad9bcb60457f40eecd86780b408451d0fffb791ab29350ddd459fde4e9
R3_CLAIM_EVIDENCE.md
c334c947284c6af0f8158ff0ab70a64aef75686bc4f2873302129c6ba383dd50
VALIDITY_EFFICIENCY_FRONTIER.md
ba6843f810c9687a95235e3dab5a694e331e1e28d781e07f3856250b316e8690
CONTRIBUTION_MAP.md
82b00c9f2a7a28981dac1de48a1c289f7d3f4d24363ee875252c814a28b31b23
R3_PAPER_CORE.md
bb3db1fc2e156a667e4b9ad8b80805ae3001dc8e198d862a103b2999accc28ba
FINAL_STATUS.json (provisional)
ed837f34093fb4d2d2c9863692b5807e5c613f2bb5786d8c05177df30c1a5bf0
README.md
adc6bd40d2861d6ac04bbc854958bf9844566da3cb4bb0dec2e892d52af5fe9c
seal.py
5d8fd4dfdaac5864787f0211d99e039e77fccaac769417c85cc543f2a64f6ef8
```

## 最终文档事项关闭（2026-09-17）

本次仅核对主任务报告的文字修正，不扩大审计，不运行代码、实验、拟合或新比较。

- **C5关闭。** `R3_CLAIM_EVIDENCE.md:13` 已正确写为 CRB=2/N、指定全参数log-unbiased regular pivot-only类中有限N的Var(T)>2/N、等号不可达，且未建立统一正间隙。前文各处“C5待改”是历史快照，不再是当前未决事项。
- **范围/翻译修正关闭。** 中文报告已用“90%稠密非零信号/高非零信号比例”和“检验族”；`:76` 与 `R3_ALGORITHM.md:11-20` 明确区分目标块共享径向乘子的要求与校准块允许的可逆研究方向左变换。没有改变模型或校准输入。
- **本审查自身纠错完成。** 第4节原“六核心的Strong缺口”包含了非核心t1.5例子，现已改为“M0非零场景”，并明确t1.5=case12、核心集合仍为cases0–5；各数值和原结论不变。
- **计时/临时状态核对完成。** `R3_RESULTS.md:5-6,181` 与 `FINAL_STATUS.json` 已将完整调用耗时7253.955821899988s（约120.899min）和末条progress的7253.798983699991s分别列示。临时状态仍如实保留待整合标志，`full_user_success=false`、`novelty_established=false` 与证据一致。
- `README.md:62-70` 的 `-B` 命令展示与概括文句仍有上一节所述非阻塞记述差异，可由主任务在封存前统一；不影响证据身份或科学结论，不要求再开审查/科学运行。

**本次唯一独立审查完成：无剩余科学或数学阻塞，可整合审查并由主任务完成限定交付封存。** M0限定证明和确认效率增益保留；两个默认漂移恶化及未确立新颖性仍排除用户完整成功。本声明不是对尚未执行的最终manifest操作作完成认证。除本审查文件外未修改其他文件。

### 最后差异关闭：README命令（2026-09-17）

仅核对 `README.md:62-70,90-92`：完成调用及后续复现命令现均显式包含 `-B`，与说明一致。此前该项非阻塞文档差异已关闭；C5、中文范围/术语、计时及本审查t1.5场景归属修正已在上节核实关闭，本次未重复审计或执行任何所列命令。

**全部已列审查事项关闭，REVIEW_FINAL.md定稿，可纳入最终manifest。** 无新增科学结论，`full_user_success=false`、`novelty_established=false` 保持不变；最终封存操作仍由主任务执行。
