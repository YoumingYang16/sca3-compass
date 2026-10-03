# SCA3 Compass：证据门户候选发布

此文件描述软件交付，不宣告 R5 科研目标完成。原冻结版本不变。

## 采用哪个版本

默认产品是**只读证据浏览器**，没有默认患者推断算法。R4-TCB-0.2.0
的正式确认保留；R5 A0.14 展示为开发候选，未晋升。R1 是历史阶段简称，
不能与 K-NR-1.0.0、所有 R2/R3 版本自动视为同一可比批次。

R5 尚无统一范围、同信息、冻结后的全版本确认比较，不能认定全面优于
R1–R4。D020 中连续效应 C7：A014 Power=21.323529%，R4 target=32.352941%，
R4 bridge=17.401961%，同升级固定混合=37.990196%。这是每场景两次旧输入
开发诊断，不是精确总体排名；它已足以否定“现有证据全面胜出”的表述。
X9 上界低估时 A014 平均 FDP=8.229814%，保持显示，不能当作安全收益。

来源：`research/r5-selection-aware-20260918/D020/summary.json`、全部20条索引
记录及其哈希。R4 每场景512次的 `C001/summary.json` 单独显示，绝不并入D020。

## 产品能力与边界

- `#research`：R4正式确认、R5同输入诊断、全部十场景、强参照、FDR区间、
  零假设 Power 未定义、失败边界、来源SHA与逐次标量指标下载。
- `#family`：模拟/生物研究/临床证据分层、证据阅读FAQ、原始资料入口。
- 静态公开构建不调用 FastAPI，不收集病历，不执行拟合，无登录或追踪脚本。
- 不声称提升患者生活质量、预测进展、诊断或推荐治疗；家庭教育效果未测量。
- 原本地工作台保留；新门户不依赖本机私有处理数据即可展示。
- GitHub 源码上传不等于网站公开部署。两项状态分别记录，不能伪报 URL。

## 本机已执行与可复现流程

从项目根目录：

```powershell
.venv/Scripts/python.exe -B scripts/export_product_evidence.py
.venv/Scripts/python.exe -B -m pytest tests/test_product_evidence.py -q -p no:cacheprovider
```

从 `apps/web`（已有 node_modules）：

```powershell
pnpm build:public
```

新环境需要 Node 与 pnpm，使用现有 `pnpm-lock.yaml` 的冻结安装：
`pnpm install --frozen-lockfile`。依赖不应改成一次无锁的 latest 重新解析。
本轮没有新拟合或正式确认。结果导出只读取已有记录并核对哈希。

本轮测试发现旧 `vite.config.js` 优先于 TypeScript 配置的遮蔽问题。正式命令
已明确 `--config vite.config.ts`；保留旧文件，不依赖临时环境变量来掩盖问题。

## 发布与许可

拟建 `YoumingYang16/sca3-compass`，首次按私有仓库准备。没有选择开源许可；
上传不自动授权第三方再分发。第三方依赖遵守各自许可。无账户访问时不上传。
公开网站只部署 `apps/web/dist-public` 的审查输出，不能把项目根目录、
研究数据目录、FastAPI全服务或虚拟环境挂到互联网。

GitHub/公网实际状态、最终测试与审查结果以本轮交付记录为准。
