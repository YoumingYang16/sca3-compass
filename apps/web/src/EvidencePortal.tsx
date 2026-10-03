import { useEffect, useState } from "react";
import { ArrowDownToLine, ArrowUpRight, CheckCircle2, Compass, FlaskConical, HeartHandshake, ShieldCheck } from "lucide-react";
import "./evidence-portal.css";

type Metric = { power: number | null; fdp: number; tp: number; fp: number; discoveries: number; FDR_interval?: { simultaneous_interval: number[] } };
type Scene = { name: string; ns: number; nt: number; D: number; distribution: string };
type Case = { case: number; scene: Scene; n: number; outside: boolean; methods: Array<Metric & { id: string; label: string }> };
type Data = {
  snapshot_date: string; kind: string; limitations: string[]; sources: { path: string; sha256: string }[];
  r4: { rows: { case: number; scene: Scene; n: number; methods: Record<string, Metric>; paired_power: Record<string, { mean: number; simultaneous_interval: number[] }> }[] };
  r5: { cases: Case[]; n: number; confirmation_count: number };
};
const pct = (v: number | null | undefined) => v == null ? "未定义" : `${(v * 100).toFixed(2)}%`;
const names = ["普通分布 · 匹配", "t5 重尾 · 小目标库", "t3 重尾 · 紧上界", "合法但宽的上界", "全局零假设", "单研究效应 · PC 零假设", "混合方向 · PC 零假设", "高比例连续效应", "固定预算 · 更多目标块", "上界低估 · 范围外"];
const key4 = ["target", "source_bound", "bridge", "strong_target", "strong_pool_bound"];
const label4: Record<string, string> = { target: "Target-only（沿用 R3）", source_bound: "Source-bound", bridge: "R4 固定桥", strong_target: "强参照 target · 仅实证", strong_pool_bound: "强参照 pool · 仅实证" };

function MetricBars({ methods }: { methods: { label: string; power: number | null; fdp: number }[] }) {
  return <div className="ep-bars" role="img" aria-label="各方法 Power 与平均 FDP，精确数值另见下方表格">
    <div className="ep-bar-header"><span>检出能力 / Power</span><span>0 — 100%</span></div>
    {methods.map((m, i) => <div className="ep-bar-row" key={m.label}><span>{m.label}</span><div className="ep-track"><div className={i === 0 ? "ep-fill primary" : "ep-fill"} style={{ width: `${Math.max(0, Math.min(100, (m.power ?? 0) * 100))}%` }} /></div><b>{pct(m.power)}</b></div>)}
    <p className="ep-muted">Power 越高表示模拟中的检出能力越强，但必须结合错误控制、区间和适用条件解释。</p>
  </div>;
}

function FamilyGuide() {
  const [opened, setOpened] = useState(0);
  const questions = [
    { title: "这些数字能告诉我家人的病情吗？", body: "不能。这里的算法比较来自有已知真值的计算机模拟，不是对家庭成员的检测、病程预测或疗效评估。我们用它检查分析方法何时可信、何时可能产生误导。" },
    { title: "为什么结果里会出现失败？", body: "失败能说明方法的使用边界。借用其他研究的信息有时提高检出能力，也可能因为研究条件不匹配而损害结果。公开负面结果，是帮助读者判断证据的一部分。" },
    { title: "这对 SCA3 家庭的实际价值是什么？", body: "当前提供的是研究透明度和证据阅读工具：分辨模拟、动物/细胞研究与人体证据；追溯来源；看懂不确定性。尚未证明本网站改善患者结局或照护效果。" },
    { title: "看到一条新研究，可以先核对什么？", body: "研究对象是什么？独立参与者有多少？结果是否经过另一批数据检验？差异有多大、范围多宽？是否公开局限与负面结果？这些问题不能代替医生对个人情况的判断。" },
  ];
  return <>
    <section className="ep-hero"><p className="ep-kicker">FOR PATIENTS & FAMILIES / 证据阅读</p><h1>更清楚地理解研究，<br /><em>而不是过早相信结论。</em></h1><p className="ep-lead">为 SCA3 患者与照护者提供一个透明的研究窗口：知道数字来自哪里，也知道它不能说明什么。</p><div className="ep-chips"><span>不收集病历</span><span>不做个体预测</span><span>不提供治疗推荐</span></div></section>
    <section className="ep-three" aria-label="三种证据的区别">
      {[['01', '方法模拟', '计算机生成的数据，用来检验统计方法。本站 R1–R5 的方法比较属于这一层，不是患者发现。'], ['02', '公开生物研究', '需要逐项核对物种、组织、独立样本和研究设计。动物、细胞或分子信号不能直接等同于人体获益。'], ['03', '临床与个人决策', '需要相应人体研究与专业评估。本网站不具备个人诊断、用药或疾病进程预测能力。']].map(([n,t,d]) => <article className="ep-card" key={n}><span className="ep-number">{n}</span><h2>{t}</h2><p>{d}</p></article>)}
    </section>
    <section className="ep-panel"><div className="ep-section-title"><div><p className="ep-kicker">READ WITH CONFIDENCE</p><h2>四个常见问题</h2></div><HeartHandshake size={28} /></div>{questions.map((q,i) => <div className="ep-faq" key={q.title}><button aria-expanded={opened === i} aria-controls={`answer-${i}`} onClick={() => setOpened(opened === i ? -1 : i)}>{q.title}<span>{opened === i ? '−' : '+'}</span></button><div id={`answer-${i}`} hidden={opened !== i}><p>{q.body}</p></div></div>)}</section>
    <section className="ep-panel"><p className="ep-kicker">ORIGINAL SOURCES</p><h2>从原始资料继续了解</h2><p>以下是外部信息入口，不是本项目的临床背书。内容与更新以原网站为准。</p><div className="ep-links"><a href="https://www.ncbi.nlm.nih.gov/books/NBK1196/" target="_blank" rel="noopener noreferrer">GeneReviews · Spinocerebellar Ataxia Type 3 <ArrowUpRight size={16} /></a><a href="https://www.ataxia.org/sca3/" target="_blank" rel="noopener noreferrer">National Ataxia Foundation · SCA3 <ArrowUpRight size={16} /></a></div></section>
  </>;
}

function ResearchView({ data }: { data: Data }) {
  const [batch, setBatch] = useState<'r4' | 'r5'>('r4');
  const [caseId, setCaseId] = useState(1);
  const row4 = data.r4.rows.find(r => r.case === caseId)!;
  const row5 = data.r5.cases.find(r => r.case === caseId)!;
  const scene = row4.scene;
  const rows = batch === 'r4' ? key4.map(k => ({ id: k, label: label4[k], ...row4.methods[k] })) : row5.methods;
  return <>
    <section className="ep-hero"><p className="ep-kicker">SCA3 COMPASS / EVIDENCE OBSERVATORY</p><h1>让每一个结果，<br /><em>都有可以追溯的证据。</em></h1><p className="ep-lead">从有限校准到安全信息借用。探索方法在什么条件下有效、付出什么代价，以及仍然没有解决的问题。</p><div className="ep-chips"><span>研究结果浏览器</span><span>只读 · 无在线拟合</span><span>模拟 ≠ 患者数据</span></div></section>
    <div className="ep-metrics"><div><strong>5,120</strong><span>R4 正式确认 family</span></div><div><strong>20</strong><span>本页 R5 配对开发 family</span></div><div><strong>0</strong><span>R5 正式独立确认</span></div><div><strong>未晋升</strong><span>R5 当前交付状态</span></div></div>
    <section className="ep-notice"><ShieldCheck size={23} /><div><b>版本更新，不等于全面胜出。</b><p>目前不能认定 R5 比 R1–R4 都更好。下方分开展示 R4 正式确认和 R5 同输入开发比较；没有将 R5 设为已验收的默认推断算法。</p></div></section>
    <section className="ep-panel"><div className="ep-section-title"><div><p className="ep-kicker">01 / COMPARE WITHIN A BATCH</p><h2>结果实验室</h2></div><span className="ep-tag">SIMULATION ONLY</span></div>
      <div className="ep-controls"><label>证据批次<select value={batch} onChange={e => setBatch(e.target.value as 'r4' | 'r5')}><option value="r4">R4 C001 · 正式确认 · 每场景 512 次</option><option value="r5">R5 D020 · 已见输入开发诊断 · 每场景 2 次</option></select></label><label>场景<select value={caseId} onChange={e => setCaseId(Number(e.target.value))}>{names.map((name,i) => <option value={i} key={name}>{i === 9 ? 'X9' : `C${i}`} · {name}</option>)}</select></label></div>
      <div className="ep-scene"><span>源库 <b>{scene.ns}</b> 块</span><span>目标库 <b>{scene.nt}</b> 块</span><span>分布 <b>{scene.distribution}</b></span><span>外部上界 D = <b>{scene.D}</b></span><span>名义 q = <b>5%</b></span></div>
      {caseId === 9 && <p className="ep-warning" role="status">范围外失败：漂移上界被低估。这些高 Power 不能作为安全借用有效的证据。</p>}
      {batch === 'r5' && <p className="ep-warning">每场景仅 2 次已见输入诊断；没有正式确认、没有可支持胜出结论的区间。关键强参照、校准升级端点和固定混合并列展示，不按结果挑选。</p>}
      <MetricBars methods={rows} />
      <div className="ep-scroll"><table><caption>{batch === 'r4' ? 'R4 C001：每场景 512 个完整 family，模拟不确定性区间来自预定比较族' : 'R5 D020：与 R4 子集同输入的开发比较，不能与上面 512 次均值混排'}</caption><thead><tr><th>方法</th><th>Power ↑</th><th>平均 FDP</th><th>{batch === 'r4' ? 'FDR 联合区间' : '平均正确发现'}</th><th>平均错误发现</th></tr></thead><tbody>{rows.map(r => <tr key={r.id}><td>{r.label}</td><td>{pct(r.power)}</td><td>{pct(r.fdp)}</td><td>{batch === 'r4' ? (r.FDR_interval?.simultaneous_interval.map(pct).join(' – ') ?? '—') : r.tp.toFixed(2)}</td><td>{r.fp.toFixed(2)}</td></tr>)}</tbody></table></div>
      <p className="ep-muted">{batch === 'r4' ? 'FDR 区间采用完整 family 为单位；预设上限 160 项、实际 133 项，联合覆盖至少 97.5%。它不是所有分布下的保证，强参照没有本次完整有限样本定理。' : 'D020 额外改变了校准处理与选择机制，不能把差异全部归因于自适应选择。两次结果中的平均 FDP 不是已验证的 FDR 上界。'} family 在此指整个多重检验批次，不是患者家庭。独立重复不是单个基因；全 PC 零假设场景的 Power 未定义。</p>
    </section>
    <section className="ep-panel"><p className="ep-kicker">02 / VERSION ≠ RANKING</p><h2>版本与证据边界</h2><div className="ep-timeline">{[
      ['V1 / K-NR-1.0.0', '限定实证', '历史经验验收保留；不是一般有限样本保证。R1 为历史阶段简称，此处以实际发布标识为准。'],
      ['R2 / PB-1.0.0', '有保证，但代价较大', '匹配模型下的有限校准控制；不能以理论成立替代检出效率。'],
      ['R3 / GSR-1.0.0', '限定模型效率提升', '原 M0 批次相对 R2 核心平均 +6.894 点，区间 [5.461, 8.201]；失配失败仍保留。不是与本页 R4/R5 的统一排名。'],
      ['R4 / TCB-0.2.0', '有条件有效', '紧上界时借用有利，合法但宽的上界可造成严重损失；目标校准与外部 D 是前提。'],
      ['R5 / A0.14（开发）', '未验收', '选择保护付出效率代价，连续效应诊断出现退化。尚未证明优于所有旧版，不提供患者推断接口。'],
    ].map(([v,s,b]) => <article key={v}><span className="ep-timeline-dot" /><div><strong>{v}</strong><span className="ep-version-state">{s}</span><p>{b}</p></div></article>)}</div></section>
    <section className="ep-panel"><div className="ep-section-title"><div><p className="ep-kicker">03 / TRANSPARENCY</p><h2>可以核查，也可以质疑</h2></div><a className="ep-download" href={`${import.meta.env.BASE_URL}research-evidence.json`} download><ArrowDownToLine size={16} /> 下载展示数据</a></div><ul className="ep-limits">{data.limitations.map(l => <li key={l}>{l}</li>)}</ul><details><summary>源文件与 SHA-256</summary><div className="ep-source-list">{data.sources.map(s => <div key={s.path}><span>{s.path}</span><code>{s.sha256}</code></div>)}</div></details><p className="ep-muted">导出包含全部 20 条 D020 的标量逐次指标，未包含原始矩阵、患者信息或未解封数据。下载不是完整研究代码包。外部同行评审：未进行。</p></section>
  </>;
}

export function EvidencePortal({ family, publicOnly }: { family: boolean; publicOnly: boolean }) {
  const [data, setData] = useState<Data>();
  const [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    fetch(`${import.meta.env.BASE_URL}research-evidence.json`, { signal: controller.signal })
      .then(r => { if (!r.ok) throw new Error(`Evidence HTTP ${r.status}`); return r.json(); })
      .then(setData).catch(e => { if (e.name !== 'AbortError') setError('已保存的研究结果暂不可用；不会用示例数字替代。'); });
    return () => controller.abort();
  }, []);
  return <div className="ep-root"><a className="ep-skip" href="#ep-content" onClick={e => { e.preventDefault(); document.getElementById('ep-content')?.focus(); }}>跳到主要内容</a><header className="ep-nav"><a href="#research" className="ep-brand"><Compass size={27} /><span>SCA3 <b>COMPASS</b><small>Evidence, in context.</small></span></a><nav aria-label="主导航"><a aria-current={!family ? 'page' : undefined} href="#research"><FlaskConical size={16} />研究结果</a><a aria-current={family ? 'page' : undefined} href="#family"><HeartHandshake size={16} />给患者与家庭</a>{!publicOnly && <a href="#overview">本地工作台 <ArrowUpRight size={14} /></a>}</nav><span className="ep-nav-note"><CheckCircle2 size={13} /> 只读证据视图</span></header>
    <main id="ep-content" tabIndex={-1}>{family ? <FamilyGuide /> : error ? <section className="ep-panel"><h1>研究证据未载入</h1><p role="alert">{error}</p><button onClick={() => window.location.reload()}>重新载入</button></section> : data ? <ResearchView data={data} /> : <p className="ep-loading" role="status">正在载入已核验的结果…</p>}</main>
    <footer className="ep-footer"><div><b>SCA3 / COMPASS</b><p>透明研究 · 清楚边界 · 尊重不确定性</p></div><p>结果快照 {data?.snapshot_date ?? '2026-10-03'}<br />非医疗器械，不作诊断或治疗推荐<br />无登录、无病历上传、无用户追踪脚本</p></footer>
  </div>;
}
