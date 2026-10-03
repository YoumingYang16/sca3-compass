"""Derive figures and human-readable tables only from frozen C001 summaries."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import read,sha,write

LABELS={'PB_grid':'R2 predictive + grid','PB_main':'R2 continuous calibration',
        'PB_grid_ordinary':'R2 ordinary','K_NR':'Frozen K-NR V1',
        'B_strong':'Practical strong reference','B_fair_conditional_eBH':'V1 fair ordinary',
        'R2_main':'Confidence envelope','BB_BY':'Classical BB + BY','BB_eBH':'BB learned (empirical only)'}
COLORS={'PB_grid':'#073b4c','PB_main':'#548c9d','PB_grid_ordinary':'#80a99e',
        'K_NR':'#bd7435','B_strong':'#783b69','B_fair_conditional_eBH':'#81878b'}


def interval_text(value):
    if value is None: return 'undefined'
    lo,hi=value['simultaneous_95_kl']
    return f"{100*value['mean']:.2f} [{100*lo:.2f}, {100*hi:.2f}]"


def main(source,out):
    if out.exists(): raise FileExistsError('preserve generated report')
    out.mkdir(); result=read(source/'summary.json'); rows=result['rows']
    methods=['PB_grid','PB_grid_ordinary','PB_main','K_NR','B_strong','B_fair_conditional_eBH','R2_main','BB_BY','BB_eBH']
    lines=['# Frozen C001 numerical results','',
           'All values are percent (paired differences: percentage points). Brackets are the predeclared',
           'simultaneous finite Chernoff–KL intervals, family cap512two-sided intervals, total error.05.',
           '512 whole independent families per scene. Model and mismatch tables remain separate.',
           'BB learned is EMPIRICAL_ONLY; BB+BY is model-valid. Undefined Power is not zero.','']
    for outside in [False,True]:
        lines+=['## '+('Outside-M0 sensitivity' if outside else 'Within M0'),'']
        for row in rows:
            if bool(row['case'].get('outside_M0'))!=outside: continue
            lines+=['### '+row['case']['name'],'',
                    '| Method | Power % [simultaneous CI] | FDR % [simultaneous CI] |',
                    '|---|---:|---:|']
            for key in methods+(['PB_Delta2','PB_Delta5'] if outside else []):
                m=row['methods'][key]
                lines.append(f"| {LABELS.get(key,key)} | {interval_text(m['power'])} | {interval_text(m['fdp'])} |")
            lines+=['','Paired R2 grid Power differences (pp):','',
                    '| Relative to | Difference [simultaneous CI] | Descriptive MCSE |','|---|---:|---:|']
            for key,value in row['paired_power_difference'].items():
                lines.append(f"| {LABELS[key]} | {interval_text(value)} | {100*value['mcse']:.3f} |")
            lines+=['',f"PB numerical failures: {row['numerical_failures']}; sensitivity failures: {row['sensitivity_failures']}; V1 fallback families: {row['v1_declared_fallback_families']}.",
                    f"Calibration-domain upper coverage: {interval_text(row['coverage_of_calibration_domain_only'])}. Upper/true-cal kappa quantiles(10/50/90%): {row['kappa_upper_ratio_quantiles']}.",
                    f"Mean computation seconds PB/V1/envelope: {row['runtime']['pb_seconds']['mean']:.3f} / {row['runtime']['v1_seconds']['mean']:.3f} / {row['runtime']['fc_seconds']['mean']:.3f}.",'']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    curve=[rows[i] for i in [1,2,3,4]]; x=np.array([4,12,32,128])
    for key in ['PB_grid','PB_main','PB_grid_ordinary','K_NR','B_strong']:
        values=np.array([r['methods'][key]['power']['mean']*100 for r in curve])
        axes[0].plot(x,values,'o-',label=LABELS[key],color=COLORS[key])
    axes[0].set(xscale='log',xticks=x,xticklabels=x,xlabel='Independent calibration blocks N',ylabel='Power (%)',ylim=(0,100),title='Matched t5, rho=.95, effect=3')
    axes[0].legend(fontsize=8)
    for key in ['PB_grid','B_strong','K_NR']:
        means=np.array([r['methods'][key]['fdp']['mean']*100 for r in curve]); intervals=np.array([r['methods'][key]['fdp']['simultaneous_95_kl'] for r in curve])*100
        axes[1].errorbar(x,means,yerr=np.vstack([means-intervals[:,0],intervals[:,1]-means]),fmt='o-',capsize=3,label=LABELS[key],color=COLORS[key])
    axes[1].axhline(5,color='#af3b3b',ls='--',lw=1,label='Nominal FDR5%')
    axes[1].set(xscale='log',xticks=x,xticklabels=x,xlabel='Independent calibration blocks N',ylabel='FDR (%)',title='Simultaneous finite intervals; not a proof')
    axes[1].legend(fontsize=8)
    fig.savefig(out/'calibration-efficiency.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for r,label,color in [(rows[13],'rho .80 to .90','#073b4c'),(rows[14],'rho .80 to .95','#bd7435')]:
        for metric,ax in [('power',axes[0]),('fdp',axes[1])]:
            values=[r['methods'][k][metric]['mean']*100 for k in ['PB_grid','PB_Delta2','PB_Delta5']]
            ax.plot([1,2,5],values,'o-',label=label,color=color)
    axes[1].axhline(5,color='#af3b3b',ls='--',lw=1)
    for ax,name in zip(axes,['Power','FDR']):
        ax.set(xlabel='Externally specified Delta (sensitivity only)',ylabel=name+' (%)',xticks=[1,2,5]); ax.legend()
    fig.suptitle('Outside matched M0: Delta5 covers design ratios2.2 and4.6; Delta1/2 do not',fontsize=10)
    fig.savefig(out/'mismatch-sensitivity.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    q=np.array([r['kappa_upper_ratio_quantiles'] for r in curve])
    ax.plot(x,q[:,1],'o-',color='#073b4c',label='Median upper / true kappa')
    ax.fill_between(x,q[:,0],q[:,2],color='#548c9d',alpha=.2,label='Empirical10th–90th percentile')
    ax.axhline(1,color='black',ls=':'); ax.set(xscale='log',xticks=x,xticklabels=x,yscale='log',xlabel='Independent calibration blocks N',ylabel='Upper bound / calibration kappa',title='99.5% one-sided coverage reference; not primary PB scale')
    ax.legend(fontsize=8); fig.savefig(out/'coverage-width.png',dpi=180); plt.close(fig)
    write(out/'manifest.json',{'source_summary_sha256':sha(source/'summary.json'),
          'script_sha256':sha(__file__),'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--source',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); main(Path(a.source).resolve(),Path(a.out).resolve())
