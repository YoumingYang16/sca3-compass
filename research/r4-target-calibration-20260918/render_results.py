"""Regenerate tables and vector figures from completed fixed R4 summaries.

Standard-library renderer; no model call, simulated input, smoothing or tuning.
"""
from pathlib import Path
import json,hashlib,html
PHASE=Path(__file__).resolve().parent

def digest(p):
    with p.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    src=PHASE/'C001/summary.json';s=json.loads(src.read_text(encoding='utf8'))
    if s['status']!='COMPLETE' or s['intervals_count']!=133: raise ValueError('completed registered summary required')
    out=PHASE/'generated';out.mkdir(exist_ok=True)
    table=['# Frozen R4-C001 results','',
        'Generated from `C001/summary.json`, SHA256 `'+digest(src)+'`. No new inference.',
        'Power/FDR percentages; paired differences in percentage points. Intervals are the registered97.5% simultaneous family (133inside cap160).',
        '','|case|method|Power %|FDR %|FDR simultaneous interval %|TP/family|FP/family|seconds/family|',
        '|---|---|---:|---:|---|---:|---:|---:|']
    for r in s['rows']:
        for k,m in r['methods'].items():
            power='N/A' if m['power'] is None else f"{100*m['power']:.4f}"
            lo,hi=m['FDR_interval']['simultaneous_interval']
            table.append(f"|{r['case']} {r['scene']['name']}|{k}|{power}|{100*m['fdp']:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|{m['tp']:.4f}|{m['fp']:.4f}|{m['seconds']:.4f}|")
    table+=['','## Registered paired Power comparisons','',
        '|case / group|comparison|difference pp|simultaneous interval pp|MCSE pp (descriptive)|',
        '|---|---|---:|---|---:|']
    for group,comparisons in [(str(r['case']),r['paired_power']) for r in s['rows']]+[
            ('equal-weight core',s['core_paired_power']),('fixed36 allocation',s['allocation_comparisons'])]:
        for k,v in comparisons.items():
            lo,hi=v['simultaneous_interval']
            table.append(f"|{group}|{k}|{100*v['mean']:.4f}|[{100*lo:.4f}, {100*hi:.4f}]|{100*v['mcse']:.4f}|")
    with (out/'RESULT_TABLES.md').open('x',encoding='utf8') as f: f.write('\n'.join(table)+'\n')
    legal=[r for r in s['rows'] if not r['scene'].get('outside_bound')]
    nonnull=[r for r in legal if r['methods']['target']['power'] is not None]
    gates={'source_summary_sha256':digest(src),'reference_draws':4095,'family_repeats':512,
        'legal_bridge_FDR_upper_max':max(r['methods']['bridge']['FDR_interval']['simultaneous_interval'][1] for r in legal),
        'legal_FDR_MC_gate':all(r['methods']['bridge']['FDR_interval']['simultaneous_interval'][1]<=.05 for r in legal),
        'local_positive_bridge_minus_target_cases':[r['case'] for r in nonnull if r['paired_power']['bridge_minus_target']['simultaneous_interval'][0]>0],
        'supported_negative_bridge_minus_target_cases':[r['case'] for r in nonnull if r['paired_power']['bridge_minus_target']['simultaneous_interval'][1]<0],
        'core_positive_gate':s['core_paired_power']['bridge_minus_target']['simultaneous_interval'][0]>0,
        'no_important_regression_gate':all(r['paired_power']['bridge_minus_target']['simultaneous_interval'][0]>=-.02 for r in nonnull),
        'theory':'MODEL_CONDITIONAL_EXACT_ARITHMETIC_REVIEWED','novelty':'PRIORITY_NOT_ESTABLISHED',
        'real_target_calibration':'NOT_CERTIFIED_AVAILABLE'}
    gates['overall_advantage_gate']=gates['core_positive_gate'] and gates['no_important_regression_gate']
    with (out/'GATES.json').open('x',encoding='utf8') as f: json.dump(gates,f,indent=2)
    # Forest diagram shows all nonnull scenes, including registered failure.
    plotted=[(f"C{r['case']}: {r['scene']['name']}",r['paired_power']['bridge_minus_target'],r['scene'].get('outside_bound',False))
             for r in s['rows'] if r['paired_power']]
    plotted.append(('Equal-weight legal core',s['core_paired_power']['bridge_minus_target'],False))
    width=1280;height=160+len(plotted)*66
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
         '<rect width="100%" height="100%" fill="white"/>',
         '<style>text{font-family:Arial,sans-serif;fill:#182436;font-size:14px}.title{font-size:22px;font-weight:bold}.small{font-size:12px;fill:#506070}</style>',
         '<text x="32" y="36" class="title">R4: protected borrowing minus target-only Power</text>',
         '<text x="32" y="60" class="small">512 whole-family paired repeats per scene; registered simultaneous intervals. Not a universal superiority claim.</text>']
    xmin=-70;xmax=80;x0=465;x1=1080
    def xx(v): return x0+(100*v-xmin)/(xmax-xmin)*(x1-x0)
    for tick in range(-60,81,20):
        x=xx(tick/100)
        svg.append(f'<line x1="{x}" y1="84" x2="{x}" y2="{height-65}" stroke="'+('#465666' if tick==0 else '#e3e8ec')+'"/>')
        svg.append(f'<text x="{x}" y="{height-42}" text-anchor="middle">{tick}</text>')
    for i,(label,v,outside) in enumerate(plotted):
        y=113+i*66;lo,hi=v['simultaneous_interval'];color='#a34820' if outside else '#215677'
        if not xmin<=100*lo<=100*hi<=xmax: raise ValueError('figure axis would clip an interval')
        svg.append(f'<text x="32" y="{y+5}">{html.escape(label)}</text>')
        svg.append(f'<line x1="{xx(lo)}" x2="{xx(hi)}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{xx(v["mean"])}" cy="{y}" r="5" fill="{color}"/>')
        svg.append(f'<text x="1100" y="{y+5}">{100*v["mean"]:+.2f} pp</text>')
        if outside: svg.append(f'<text x="32" y="{y+23}" class="small">OUTSIDE stated drift bound; high Power does not establish validity</text>')
    svg.append(f'<text x="770" y="{height-16}" text-anchor="middle" class="small">Paired Power difference (percentage points); horizontal axis includes all intervals</text></svg>')
    with (out/'paired-power.svg').open('x',encoding='utf8') as f: f.write('\n'.join(svg))
    import xml.etree.ElementTree as ET
    ET.parse(out/'paired-power.svg')
    print(json.dumps(gates,indent=2))

if __name__=='__main__': main()
