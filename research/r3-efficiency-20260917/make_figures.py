"""Rebuild descriptive plots/tables from the frozen completed summary only."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
COLORS={'R3_main':'#136a81','PB_grid':'#777a80','B_strong':'#ad6633','R3_ordinary':'#61538e'}
LABELS={'R3_main':'R3 geometric + independent shape','PB_grid':'R2 safe bridge','B_strong':'Strong (empirical only)','R3_ordinary':'R3 ordinary PC'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=BASE/'figures');args=parser.parse_args()
    source=BASE/'C001/summary.json';s=json.loads(source.read_text());rows=s['rows']
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.labelcolor':'#26313d','text.color':'#26313d'})
    ids=[r['case'] for r in rows if not r['design'].get('outside_M0') and r['methods']['R3_main']['power'] is not None]
    short={0:'Normal, N32',1:'t5, N4',2:'t5, N12',3:'t5, N32',4:'t5, N128',5:'t3, N12',6:'Continuous effects',7:'Dense90',8:'Global null',9:'Singleton null',10:'Mixed-sign PC null',11:'Effect2.625',12:'t1.5, N12',13:'Moderate drift',14:'Severe drift'}
    fig,axes=plt.subplots(1,2,figsize=(13,6.4),gridspec_kw={'width_ratios':[1.25,1]})
    yy=np.arange(len(ids))
    for offset,k in zip([-.17,0,.17],['PB_grid','R3_main','B_strong']):
        vals=np.array([rows[i]['methods'][k]['power']['mean']*100 for i in ids])
        ci=np.array([rows[i]['methods'][k]['power']['simultaneous_interval'] for i in ids])*100
        axes[0].errorbar(vals,yy+offset,xerr=[vals-ci[:,0],ci[:,1]-vals],fmt='o',ms=4,color=COLORS[k],capsize=2,label=LABELS[k])
    axes[0].set_yticks(yy,[short[i] for i in ids]);axes[0].invert_yaxis();axes[0].set_xlabel('Power (%)');axes[0].legend(fontsize=8,loc='lower right')
    vals=np.array([rows[i]['paired_power']['PB_grid']['mean']*100 for i in ids])
    ci=np.array([rows[i]['paired_power']['PB_grid']['simultaneous_interval'] for i in ids])*100
    axes[1].errorbar(vals,yy,xerr=[vals-ci[:,0],ci[:,1]-vals],fmt='o',color=COLORS['R3_main'],capsize=3)
    axes[1].axvline(0,color='#333333',lw=1);axes[1].axvline(-2,color='#ad6633',ls=':',label='-2pp regression guard')
    axes[1].set_yticks(yy,[]);axes[1].invert_yaxis();axes[1].set_xlabel('Paired R3 - R2 (percentage points)');axes[1].legend(fontsize=8)
    fig.suptitle('One frozen independent confirmation | 1,024 families per scene',fontsize=14)
    fig.text(.5,.01,'Simultaneous coverage >=97.5% over the preregistered comparison family; model M0 only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(out/'power-comparison.png',dpi=180);plt.close(fig)

    nn=np.array([4,12,32,128]);rr=[rows[i] for i in [1,2,3,4]]
    fig,ax=plt.subplots(2,2,figsize=(11,8))
    for k in ['PB_grid','R3_main','B_strong','R3_ordinary']:
        y=[r['methods'][k]['power']['mean']*100 for r in rr]
        ax[0,0].plot(nn,y,'o-',color=COLORS[k],label=LABELS[k])
        th=[r['methods'][k]['eBH_threshold_quantiles_when_positive'] for r in rr]
        ax[1,0].plot(nn,[np.nan if t is None else t[1] for t in th],'o-',color=COLORS[k])
    for name,color in [('geometric',COLORS['R3_main']),('median',COLORS['PB_grid'])]:
        q=np.array([r['scale_ratio_quantiles'][name] for r in rr])
        ax[0,1].plot(nn,q[:,1],'o-',color=color,label=name);ax[0,1].fill_between(nn,q[:,0],q[:,2],color=color,alpha=.12)
    ax[0,1].axhline(1,color='#333333',ls=':',lw=1)
    ax[1,1].plot(nn,np.sqrt((np.pi**2/3-1)/nn),'o-',color=COLORS['R3_main'],label='Exact SD log(B_geometric)')
    ax[0,0].set_ylabel('Power (%)');ax[0,0].legend(fontsize=8)
    ax[0,1].set_ylabel('Scale estimate / true kappa');ax[0,1].set_yscale('log');ax[0,1].legend(fontsize=8)
    ax[1,0].set_ylabel('Median final eBH threshold | R > 0');ax[1,0].set_yscale('log')
    ax[1,1].set_ylabel('Exact log-scale sampling SD');ax[1,1].legend(fontsize=8)
    for a in ax.flat:a.set_xscale('log',base=2);a.set_xticks(nn,[str(v) for v in nn]);a.set_xlabel('Independent calibration blocks N');a.grid(alpha=.12)
    fig.suptitle('Calibration-efficiency curve | t5, rho .95, effect3',fontsize=14)
    fig.text(.5,.012,'Bands: observed 10th/90th scale-error quantiles, NOT confidence coverage. Threshold conditions on any rejection.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(out/'calibration-efficiency.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(1,2,figsize=(13,6),gridspec_kw={'width_ratios':[1.3,1]})
    for offset,k in [(-.13,'PB_grid'),(.13,'R3_main')]:
        y=np.array([r['methods'][k]['fdp']['mean'] for r in rows[:13]])*100
        ci=np.array([r['methods'][k]['fdp']['simultaneous_interval'] for r in rows[:13]])*100
        ax[0].errorbar(y,np.arange(13)+offset,xerr=[y-ci[:,0],ci[:,1]-y],fmt='o',ms=3,color=COLORS[k],capsize=2,label=LABELS[k])
    ax[0].set_yticks(np.arange(13),[short[i] for i in range(13)]);ax[0].invert_yaxis();ax[0].axvline(5,color='#b64e4e',ls='--');ax[0].set_xlabel('FDR (%) | M0');ax[0].legend(fontsize=8)
    for offset,k,color in [(-.24,'PB_grid',COLORS['PB_grid']),(-.08,'R3_main',COLORS['R3_main']),(.08,'R2_Delta5','#b1b2b3'),(.24,'R3_Delta5','#63a4b3')]:
        vals=np.array([rows[i]['methods'][k]['fdp']['mean']*100 for i in [13,14]])
        ci=np.array([rows[i]['methods'][k]['fdp']['simultaneous_interval'] for i in [13,14]])*100
        ax[1].bar(np.arange(2)+offset,vals,width=.15,color=color,label=k,
                  yerr=[vals-ci[:,0],ci[:,1]-vals],capsize=2,error_kw={'elinewidth':.8})
    ax[1].axhline(5,color='#b64e4e',ls='--');ax[1].set_xticks([0,1],['Moderate drift','Severe drift']);ax[1].set_ylabel('FDR (%) | OUTSIDE M0');ax[1].legend(fontsize=8)
    fig.suptitle('Validity and explicit mismatch boundary',fontsize=14)
    fig.text(.5,.01,'Delta5 requires an external ratio bound; it is not a learned or universal drift correction.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.035,1,.96));fig.savefig(out/'validity-boundary.png',dpi=180);plt.close(fig)

    lines=['# Frozen R3 numerical tables','',f"Source: C001/summary.json SHA256 `{hashlib.sha256(source.read_bytes()).hexdigest()}`.",'',
           '1024 whole-family repeats/scene; Power/FDR in percent, differences in percentage points. All intervals belong to the prospectively fixed97.5%joint family.','',
           '| Scene | R2 Power | R3 Power | Strong Power | Paired R3-R2 interval | R3 FDR [interval] |','|---|---:|---:|---:|---|---|']
    for r in rows:
        def number(k):
            v=r['methods'][k]['power'];return 'undefined' if v is None else f"{100*v['mean']:.3f}"
        d=r['paired_power'].get('PB_grid');delta='undefined' if d is None else f"{100*d['mean']:+.3f} [{100*d['simultaneous_interval'][0]:+.3f}, {100*d['simultaneous_interval'][1]:+.3f}]"
        f=r['methods']['R3_main']['fdp'];ci=f['simultaneous_interval']
        lines.append(f"| {r['design']['name']} | {number('PB_grid')} | {number('R3_main')} | {number('B_strong')} | {delta} | {100*f['mean']:.3f} [{100*ci[0]:.3f},{100*ci[1]:.3f}] |")
    lines+=['','## Component2x2 (descriptive point means; independent confirmation)','','| Scene | A R2 | B scale only | C orientation+median | D full R3 | D-B-C+A |','|---|---:|---:|---:|---:|---:|']
    for i in ids:
        r=rows[i];a,b,c,d=[r['methods'][k]['power']['mean']*100 for k in ['PB_grid','R3_scale','R3_median','R3_main']]
        lines.append(f'| {short[i]} | {a:.3f} | {b:.3f} | {c:.3f} | {d:.3f} | {d-b-c+a:+.3f} |')
    lines+=['','Interaction is descriptive, not a new independently tested contrast. Individual main-minus-component intervals are in C001/summary.json.','',
            '## Actual FDR for every comparison', '',
            'Each entry is mean FDR% [simultaneous upper endpoint%], not a truth-adjusted nominal level. Full two-sided intervals are in the source summary.', '',
            '| Scene | R2 | R2 ordinary | R3 | Scale only | Orientation+median | R3 ordinary | Strong | V1 K_NR |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        values=[]
        for key in ['PB_grid','PB_grid_ordinary','R3_main','R3_scale','R3_median','R3_ordinary','B_strong','K_NR']:
            value=r['methods'][key]['fdp']
            values.append(f"{100*value['mean']:.3f} [{100*value['simultaneous_interval'][1]:.3f}]")
        lines.append('| '+r['design']['name']+' | '+' | '.join(values)+' |')
    lines.append('')
    with (out/'RESULT_TABLES.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    receipt={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
    with (out/'manifest.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
