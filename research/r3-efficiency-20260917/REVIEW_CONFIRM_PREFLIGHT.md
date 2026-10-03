# R3 C001 正式确认预审

审查日期：2026-09-17。唯一独立审查者；只读源码、协议、既有收据及相关原始文献，只编辑本审查文件。没有运行项目代码、测试、拟合、重放或模拟，也没有新增代理。最后检查时本目录尚无 `C001`。

**最终预审处置：PASS FOR FREEZE / B1、B2 均已关闭。** 允许主任务按本文末尾“最终关闭复核”所绑定快照冻结并启动唯一 R3 C001；这不是结果验收、效用/新颖性判定或研究成功声明。下文早期 HOLD 及差分记录保留作为审查历史，现由末尾最终处置取代。

## 首次快照结论（历史记录）

**数学及当前固定实验设计通过预审；冻结前须关闭下面 B1、B2 两个窄范围问题。** 这不是要求新方法或新实验，也不是对效能、贡献新颖性或最终用户成功的确认。

- 固定赌注均值区间的理想算术证明成立；289 个双侧区间没有发现漏计。
- `R3-GSR-1.0.0` 相对冻结 D003 的模型源码只有 VERSION 改动；既有角色独立性审查继续适用。
- `new` 分支没有发现真值、真实形状或真实 kappa 进入方法推断。确认样本使用新种子，历史 R2 数据仅用于开发及依赖身份核对。
- D003 的既有输出一致性问题可关闭：收据报告 312 families / 4680 个原有数组精确一致；本次独立核对了收据所绑定的两个 index 哈希、完整任务集合和冻结文件。没有自行重跑其数组扫描。
- 恢复、空 gamma 和五个预指定重放的最新改动已纳入本次审查。

## B1 — 把漂移限制落实为完整成功判定，而非仅报告信号

位置：`protocol-confirm.json:20-30`；`analyze_confirm.py:50-56`；用户附件第 15 条（附件第 552-561 行）及本轮补充指令。

当前协议要求披露漂移恶化，分析器也计算 `additional_drift_worsening_detected`，但尚未明确把它连到“禁止用户完整成功”的状态。该字段 TRUE 是坏消息，不能与其它 `empirical_gates` 做一个朴素的 `all(values)` 来认定成功。

冻结前的最小修正：

1. 两个原始、未加 Delta 的漂移比较中，只要任一个 `R3_main - PB_grid` FDP 差的同时区间下端 > 0，明确输出 `full_user_success_excluded_by_drift = true`；不得标无条件 `PROVED_AND_USEFUL`。完整披露代价，并停止扩展漂移研究或确认救援。
2. 不把“未检出恶化”写成“不更差”。对于这两个固定场景：下端 > 0 为检出恶化；上端 <= 0 才支持该场景的非增大；跨 0 为未解决。即使两个场景均支持非增大，也不是任意漂移下的定理。
3. M0 的三个数值验收、漂移状态、效用判断和贡献判断分别保存。Delta5 的有效性前提及表现不能抵销未保护版本的漂移损失。

用户最新说明已经确认上述意图；这里要求将意图写入冻结合同/判定输出。**沿用已计划的 4 个边界区间即可，不增加比较族或数据。** 对未解决的第 15 条，只能作明确限定的研究结论，不能称该要求已验证。

## B2 — 保存证据审计对缺失数组仍然 fail-open

位置：`r3_experiment.py:166-196`，特别是 `:184`、`:190-196`；写入位置 `:61-64`、`:77`、`:84-90`。

目前 `audit_rows` 检查精确的 decision 标签，但只在 `e_<label>` 存在时重算 eBH；没有要求必需 e 数组存在，也未验证 p/reference 的精确集合。故一个具有正确 decision/metrics、但漏存 e 或 p/reference 的档案，仍可通过这部分审计。读取 input 时也只要求能读到 truth，而非完整输入 schema。文件哈希保证字节未变，不保证最初保存的科学证据完整。

冻结前的最小修正：在当前审计器，或在冻结前明确指定为最终接受前置条件的独立 verifier 中，实施状态相关的精确 schema：

- 正常完成、非漂移任务：8 个 decision，6 个必需 e（四个 R3 标签及两个 PB 标签），3 个 R3 p，4 个 reference，共 21 个数组。漂移场景另有两个 Delta decision 和两个 Delta e，共 25 个。这里按当前实际保存合同计数，不要求保存原本未计划的 V1 e。
- 数值保守失败要有独立 schema：例如 R3 失败时允许无 p/reference，但其四个 e/decision 必须全零；不得对所有状态一概豁免必需 e。
- 检查精确 key 集合、固定 shape/dtype、有限性、e 非负、p 范围、reference 长度及排序；所有必需 e 都必须重算 eBH，缺失即拒绝。状态仅允许预定枚举。
- input 要求 `z/calibration/truth/shape/kappa` 五个 key、G=256/S=4/J=6、场景 N、truth 布尔形状及有限输入。它们仍然只是生成/评分证据，不能传入推断接口。

这是一项**正式证据接受合同的缺口**，不是已经发现错误结果、输入污染或方法失效。关闭它不需要新观察、新拟合或参数变动。当前哈希/任务/评分检查本身保留。

## 1. 固定赌注区间：证明与实现对应

位置：`bounded_intervals.py:9-36`；`INTERVAL_PROTOCOL.md:3-18`。

对独立的 [0,1] 变量、每个均值 <= u、0<u<=1，固定 0<c<1 的因子为

    1 + c (X_i/u - 1) >= 1-c > 0,
    E[1 + c (X_i/u - 1)] <= 1.

独立性使乘积期望 <=1；12 个固定乘积的等权平均也如此。Markov 给出阈值 1/delta 的拒绝控制，不需要为 12 个赌注另付 Bonferroni。每项对 u 单调不增，故反演可得下端；对 1-X 镜像处理上端，线性变换适用于 [-1,1] 配对差。

二分的上括号取样本均值是正确的：在 u=bar(X) 时，各固定 c 的因子平均为 1，AM-GM/Jensen 给出乘积 <=1，不会已经越过 >1 的拒绝阈值。全零、全一的边界处理也与退化分布相容。理想算术下取拒绝侧 lo 作为下端是向外的；额外 1e-12 guard 不是通用浮点认证，文档已经正确限定。

直接核对了 [Waudby-Smith–Ramdas 原文第 4 节 Proposition 2 与 Appendix B.6](https://arxiv.org/html/2010.09686)：非负资本过程及有限混合的引用相符。这里使用的是该经典原理的固定单侧网格/镜像构造，不声称发明新 CI，也不声称照搬论文所有自适应策略。

每个双侧区间分配 alpha/(2*384)，alpha=.025。最多 384 个区间同时覆盖率至少 97.5%；区间之间可以依赖。该概率描述模拟均值估计，不是方法 FDR q=.05，也不涵盖历史各阶段的联合选择。

非阻塞限定：当前 `interval()` 没有完整校验 cap、alpha、1D 输入等通用 API 参数；本协议给定的 384/.025/1024 和调用数组没有触发这一缺口。应避免把它宣称为已通过任意输入的通用区间库。`test_intervals.py` 已读，未运行。

## 2. 289 比较族、估计对象与判定规则

位置：`analyze_confirm.py:15-32,47-55`；`protocol-confirm.json:13-29`。

| 项目 | 双侧区间数 |
| --- | ---: |
| 15 场景 × 8 标签 FDP | 120 |
| 12 非零场景 × 8 标签 Power | 96 |
| 2 漂移 × 2 Delta 标签 × 2 指标 | 8 |
| 12 非零场景 × 5 个 R3 配对 Power 比较 | 60 |
| 2 漂移 × 2 个配对 FDP 差 | 4 |
| 六核心场景等权 R3-minus-R2 汇总 | 1 |
| 合计 | 289 |

纯零场景 8、9、10 的 Power 为 undefined，不能填零后增加/改变估计对象。13 个 M0 中有 10 个非零场景，恰是局部非劣门的范围。漂移场景不混入 M0 门或核心平均。

核心汇总对每个 rep j 先平均六个场景的配对差，再对 1024 个独立的六场景组求均值 CI；其范围仍为 [-1,1]。这正确估计固定六场景的等权平均，不是所有场景都改善的证明，也不是以基因数或折数充当独立重复。

5 个配对对手含 PB_grid、scale-only、median-orientation、R3 ordinary、strong，覆盖本阶段已承诺的主要机制和性能对比。PB_grid_ordinary 与 K_NR 有各自绝对指标区间，但没有直接配对差区间；不要给它们的配对差添上未经登记的显著性声明。已有联合绝对区间可用于明确标为派生的保守差界。95 个未用 cap 槽位不是看完结果后再选比较的许可。

局部下端 >=-.02 与核心下端 >0 是两个不同要求。2pp 是预先定义的退化容忍量，不是正向效用阈值；51 真备择核心中约等于 1.02 个真发现，但在 dense/其它真值数量下不能仍称“约一个”。强参照差、计算复杂性、额外真发现及新颖性仍须独立评价；通过三个统计门不自动满足完整用户任务。

## 3. 新模式、独立性与实现身份

位置：`r3_common.py:5-19`；`r3_experiment.py:22-35,55-97,104-109`；冻结依赖 `research/r2-finite-20260917/C001/experiment.py:49-73`。

- 数据生成使用 `SeedSequence([1709177301, case, rep])`；与 R2 C001 的根种子 1709173301 不同。算法参考种子为 `2*((seed*32+case)*4096+rep)+1`，在本次 15×1024 域上由进位表示直接可证单射。数据与算法的种子编码不同。这里是可复现的伪随机实现；理论仍以理想独立随机输入为前提，不把有限 PRNG 称为概率独立性的数学证明。
- new 分支只向 R3/R2 推断传 z、cal、固定算法种子；truth 只用于 `score`，真实 shape/kappa 只归档或计算诊断比值。与多个比较器共享同一家庭和参考随机元组是预定配对，不是跨家庭共享观察。
- 每个实际家庭仍新建参考 bank，不冻结一个全场景共同 bank。DIR 的 `positive_direction` 使用 DIR shape，SHAPE 只用于推断分母；PILOT 仍只选 calibrator/gamma。VERSION 与最终协议一致，模型源码相对 D003 只改版本字符串。
- 实际生成器沿用 R2 已冻结的 calibration 行生成方式。这里不构成新模型外问题：R2 冻结 `THEORY.md:35-43` 已明确允许校准块的随机可逆左变换，矩阵 pivot 逐块消去独立 study 半径。该说明不能扩展成 target 的 study-specific 半径也合法；目标块仍要求共同径向乘子。
- 父进程/worker initializer 核对 R3 三个实际模块路径及冻结哈希；R2/V1 依赖继续走既有只读身份核对。任务集合、原始记录路径/哈希、算法种子、版本、评分的检查均存在。
- 输入哈希与声明种子不是重新生成输入的独立证明。最终材料应精确区分“源码路径核对/保存输入身份”与“从生成种子完整重建”。当前五例协议写的是从保存输入重放；不能把它报告成已经验证所有生成输入。没有要求本轮新增生成实验。

非阻塞防护建议：`validate()` 目前只硬校验 G/ref_count 等少量设置，且 `one()` 将非 reuse 的值走 new 分支。冻结工具最好显式校验 input_mode、正整数预算、精确标签/场景/core 集合和 VERSION；当前被审查的协议值本身正确。这不是另设可调确认配置的授权。

## 4. 恢复、失败和最终证据

位置：`r3_experiment.py:111-164,173-196`；`analyze_confirm.py:33-45`；`test_execution.py`。

最新显式 resume 保留已有记录，不重跑已记录 hard failure；孤儿 input/evidence 拒绝继续；既有 index 禁止重启。hard failure 最终阻断统计分析，数值保守失败保留在全部家庭分母中，而非只分析成功样本。主任务新增的三个测试源码已读，未执行。

Windows 分支已改为 `Get-Process`，不存在先前快照中直接调用 `os.kill(pid,0)` 的问题。该修正必要：Python 官方文档规定 [Windows 上其它 signal 值会调用 TerminateProcess](https://docs.python.org/3/library/os.html#os.kill)。没有为验证此事调用任何进程信号。

恢复仍须操作层确认旧 worker 也已退出：仅查 owner PID 不等于查整个进程树。当前 PowerShell 非零返回码统一视为“不存活”，检查命令异常亦可能落入该分支；无法确认查询成功时应拒绝恢复，而非默认旧任务消失。这是恢复安全限定；正常新启动不受影响，不得以恢复为由删除孤儿或重抽种子。

空 gamma 列表现返回 None，避免全数值失败场景产生 NaN JSON；运行时间对 None 的排除意味着相应均值是有计时记录的子集，不是失败成本为零。报告要同时给失败数/整任务耗时。V1 顶层 fallback 计数不应冒充其所有内部 gating/convergence 分类。

五个预指定重放是 cases 0/1/5/10/14 的 rep0，已在冻结前协议列出；不计作五个新增独立家庭、不影响 289。最终接受仍需它们实际完成并有身份绑定的收据；目前不能因协议存在就宣称重放通过。Delta5 只覆盖已有 M_Delta 假设，不新增漂移稳健性结论。

## 5. D003 replay 与矩文档收尾

`audit_dev_replay.py:7-23` 对两个 index 中的对应记录核对原始哈希、完成状态、相同种子/source_record、证据哈希，逐个比较 D002 已保存数组及原有指标。收据 `D003-replay-audit.json` 记录 312 families、4680 identical_arrays。

本次只读独立核对：两个 index 均 complete，312 条、312 个唯一任务，等于各自协议的 13×24 全笛卡尔集合，无任务差异，全部 completed；两个 freeze 所列文件哈希全部吻合。收据 index 哈希分别为：

    D002 205ddb69653e3691986d6f899a818260af39402c9b3d56137b0833e1f9f1ae87
    D003 7bbd7de4dff410d77fa79d6f208007162cdd2d316e065f2b1b80dcfdf2646290

据此关闭此前“需要原有 312 输出精确比较”的问题。审查并不声称本次重做了 4680 次数组比较；该结论依赖已读 auditor 和上述绑定收据。D003 仍是开发复用，不是独立确认。

`CALIBRATION_MOMENTS.md:7-22` 的公式与严格区间正确：

    E X^t = 2^(-t) Gamma(2+t) Gamma(1-t),       -2<t<1;
    E B_g^p = exp(-p)[Gamma(2+p/N)Gamma(1-p/N)]^N, -2N<p<N.

端点也发散；log 中心化不等于 E B_g=1。已在上次审查指出的文字限定尚未落实：`:24-30` 应写 s>=0 时负矩阈值 s<2N；r>=0 时 known-shape 绝对矩阈值 r<min(20,4N)。否则把 r 允许到 <=-1 会碰到 Student 在零附近的负矩发散。可直接补这两个定义域，不改变方法。此文档小修不是确认算法的数学阻塞。有限矩改善仍不推出全算法尾部指数、随机占优或一致 Power 改善。

## 审查快照

以下为本次末次读取的 SHA256；主任务后续变更须另作差分确认，不能把本审查当作未来任意版本的放行。

```text
protocol-confirm.json  5785f92a06058ffc13042663bc7bef99a1a78de176877b3fd6268db60eeb94b5
bounded_intervals.py   d2143c54a5b9fdcc7175a1eb7712859ee66e1e20b718922baaca59f4ac946402
INTERVAL_PROTOCOL.md   4235a3214791a44ab8501282bf5d8b5c0877f126b4b36221be56103783d797ec
analyze_confirm.py     544347f4fb75237a98a6f286d3d9f5cd96a889df0d05be69c9fd4831b1a2ff23
r3_experiment.py       d15ac79e3160ea82fcb47aa7a04cee0e2c7c5d9622ba46f46735b65682fa3ad2
r3_model.py            45a1aecdf7de461987d5623a98864abdfa768f367ddad42cfe641e1ecf055447
test_execution.py      5305f88e02c8afdaa0cf73fa71f697f2b7ef683e04b8f5fc911d22bf19e94883
CALIBRATION_MOMENTS.md b0966f10c5dc46d17989983c7a4b3fc9df55775c3e9c75b4bf35d8a010730924
audit_dev_replay.py    fd4518dd94aa815efd60c7fa2e70472f5db975e1723eff3416aedd1c37756227
D003-replay-audit.json 7f3c9fd4090f65517ad4fa3d6a8f80cdc8fbbbd5e607dc6128b4bac672fccc3cd
```

没有新效能结论、临床外推、一般漂移保证或新定理/新颖性声明。

## 最后差分复核：矩阶、15 项测试收据与描述性阈值

本次仍仅只读审查，没有执行测试或方法。

- `CALIBRATION_MOMENTS.md:24-30` 已明确 s>=0、r>=0；此前矩阶文字问题关闭。
- `preflight-tests.xml` 记录 15 tests、0 failures、0 errors、0 skipped，耗时 9.033s，时间戳 2026-09-17 17:38:00 +08:00。本次读取了收据和两个新增测试源码，没有独立重跑。
- `test_execution.py:36-53` 明确载入 R2 C001 case0/14 rep0 的旧输入，并 monkeypatch `generate` 返回这些输入；断言只涉及完成状态、方法集合、R2 状态和 checkpoint 哈希，没有按 Power 选方法。它们是复用观察的接口检查，不是两个新的独立家庭。测试仍使用协议算法种子来产生参考随机数组；不得把接口测试误报成完整冻结执行器/真实新输入生成流程的验证，因为该 fixture 没有运行完整 `validate` 链。
- `analyze_confirm.py:27-30` 的新描述统计计算正确。`positive_discovery_family_fraction` 指 **R>0、至少有一个发现的家庭比例**，不是“正方向发现比例”或“至少一个真发现比例”；所有真假、两方向发现都计入 R。建议中文材料直接用“非零发现家庭比例”。
- 条件阈值为 m/(qR)=512/(.05R)=10240/R，单位是 **e-value**；仅在 R>0 的家庭中取 10%/50%/90% 分位数，全无发现时为 None。它不是 p-value 阈值，也不是总体无条件分位数。当前全部八个标签及两个 Delta 输出均以 eBH 作最终发现规则；V1 的 K_NR/B_strong 映射在冻结 `molecular_v1.py:93,127-129,152` 中核对过。
- 这些量没有调用 `ci()`、没有进入验收门，因此双侧区间仍为 289；只作描述时无需增加比较族，不能据它们作新的显著性或成功判定。

**本次新增改动通过，但整个预审尚不能关闭。** 末次实际读取中，`protocol-confirm.json` 和 `r3_experiment.py` 的哈希仍与首次预审相同；`analyze_confirm.py:55-60` 仍只有漂移检出信号，而 `r3_experiment.py:190-196` 仍只在 e 数组存在时重算。因此 B1、B2 尚未在被审查源码/冻结合同中落实。15 项通过收据不能替代这两项修复。待主任务落实后，只需对相应差分复核，不需要新实验或扩展科学任务。

本次变更文件/收据 SHA256：

```text
analyze_confirm.py     4158e7337108ef700b2500b9f009b447e68a92db2440763c89894e80d51644bf
CALIBRATION_MOMENTS.md 7e763088466bf7e225566b0147e130b7f7d1d08bfe628f12dda9b3e0c405f942
preflight-tests.xml    4aff57d4b53875e23c3d63de3b90f5a23e707f4296c28a0a0c166eb6140ac874
```

## 最终关闭复核：B1 / B2 已落实，可冻结

本轮仅核对用户要求的修复差分、测试源码和已有 XML 收据；没有运行任何测试、拟合或实验。末次检查 `C001` 尚不存在。

### B1 CLOSED

`protocol-confirm.json:27` 现明确：任一原始漂移配对 FDP 差下端 >0 即排除无条件完整成功；跨零未解决；两场景都支持非增大才能称用户第 15 条在这两个场景上获得经验支持。Delta5 不能抵销原始漂移恶化，不启动漂移救援扩展。

`analyze_confirm.py:55-66` 实现与该合同一致：按场景输出 `WORSENING_DETECTED` / `NONINCREASE_SUPPORTED_THIS_SCENE` / `UNRESOLVED`，另列 `full_user_success_excluded_by_drift` 与 `drift_nonincrease_demonstrated_in_both_scenarios`。它们只使用已有区间；289 个区间及 alpha 分配不变。三个 M0 统计门与边界、效用、贡献仍分别判读，不能对整份混合语义的 gates 字典机械求 all。

### B2 CLOSED

`r3_experiment.py:173-210` 新增完整输入/证据 schema，并在 `audit_rows:230-238` 的 new 路径上强制调用：

- input 精确五个 key，场景匹配的维度、float64/bool 类型、有限性和真实 nuisance 的基本合法性。
- completed 证据精确 21 个数组，漂移时 25 个；缺失必需 e/p/reference 或多出 key 均拒绝。
- 所有必需 e 均检查形状、float64、有限/非负并重算 eBH；p 范围与 shape、参考 bank 长度与单调排序均检查。
- R3 保守失败使用相应精确缩减 schema，要求四个 R3 e/decision 全零；R2 和 Delta 状态也有枚举及失败零证据检查，mandatory eBH 同时保证对应决策全零。
- 既有原始记录/数组哈希、精确任务集合、版本、种子、评分与 hard-failure 阻断仍保留。

因此此前“漏存 e 仍可能被接受”的路径已关闭。schema 校验不能替代生成种子重建或实际方法重放；这些证据等级仍按前文区分，但不构成此次冻结阻塞。

### 恢复与测试证据

`r3_experiment.py:141-145` 现在只把 PowerShell exit0/exit10 分别解释为存活/不存在，其它退出码直接拒绝恢复；此前查询错误可能被当作死进程的限定已修正。恢复前仍须主任务确认没有遗留 worker，不能仅凭 parent PID 消失擅自删除或重跑原任务。

已读取 `preflight-schema-tests.xml`：15 tests、0 failures、0 errors、0 skipped，9.068s，时间戳 2026-09-17 17:44:39 +08:00。`test_execution.py:36-60` 的两个旧输入接口 fixture 现在调用两种 schema，并显式断言删除 `e_R3_main` 后拒绝。没有独立重跑，也不把此收据称为穷尽所有失败分支的证明。

### 最终放行范围

**没有剩余阻塞缺陷。可按已审快照冻结 R3-GSR-1.0.0，执行单次固定 15×1024=15360 家庭确认，无 reserve、换候选、重抽种子或追加样本。** 当前模型和区间源码哈希与此前通过快照相同；本轮修复没有改变推断算法或比较族。

此处只放行冻结/执行准备。完成后的完整 index、严格 schema/评分审计、289 区间、五个预指定重放收据和实际边界/效用/贡献判断仍是最终接受所需证据；没有预判结果或替主任务执行冻结。

最终复核 SHA256：

```text
protocol-confirm.json      eedeeb64b5eec4f52a00043a233d7cf7d7d8a07cbe2668b13d26d95d78043e46
analyze_confirm.py         2c8fc1efcbfa893a35672eb8e74c6a583f3bc46bfadd159b21278ddd47fbcf02
r3_experiment.py           ade158608860b40e262b82095eeaf51c3bb5f2f99fcc7002f11e9afc73a1f43d
r3_model.py                45a1aecdf7de461987d5623a98864abdfa768f367ddad42cfe641e1ecf055447
bounded_intervals.py       d2143c54a5b9fdcc7175a1eb7712859ee66e1e20b718922baaca59f4ac946402
test_execution.py          d3df1518734b903ca6a1e70842d9f6b92c2cedfd6f34cdb12e1039c11897f196
preflight-schema-tests.xml 5d27a85ed3fcdf9896209af7431de4d04c8815fc5d240434d5df566c3a1f9f78
```
