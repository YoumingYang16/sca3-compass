"""Static scientific figures from completed results; no simulator or tuning."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
COLORS={'H':'#946454','F_safe':'#306A8A','F_empirical':'#338475'}
MAIN='R1B_guard_pilotc0.5_projection_gate_eBH'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--development',action='store_true');parser.add_argument('--project-root',type=Path,default=ROOT);a=parser.parse_args()
    folder=a.project_root.resolve()/'artifacts/robustness';src=folder/f"{a.run_id}-{'summary' if a.development else 'confirmation-analysis'}.json"
    data=json.loads(src.read_text(encoding='utf-8'))
    if not data['complete']:raise ValueError('Completed results only')
    if not a.development and data['phase'] not in ['C1','C2']:raise ValueError('Not a confirmation result')
    rows=data['rows'];core=[r for r in rows if r['case'].get('stratum')=='core'];out=folder/f'{a.run_id}-figures';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#263440','text.color':'#263440'})
    fig,axes=plt.subplots(2,2,figsize=(13.5,9.4),layout='constrained')
    ax=axes[0,0];strata={'core54':core,'normal27':[r for r in core if r['case']['distribution']=='normal'],'t5_27':[r for r in core if r['case']['distribution']=='t5']}
    for j,family in enumerate(['H','F_safe','F_empirical']):
        xx=np.arange(3)+(j-1)*.21
        if a.development:values=np.array([100*np.mean([r['comparisons'][family]['delta'] for r in rs]) for rs in strata.values()]);errors=None
        else:
            cr=[data['comparisons'][st+'/'+family] for st in strata]
            values=np.array([100*r['sample_envelope_difference'] for r in cr])
            errors=np.array([[values[i]-100*r['lower'] for i,r in enumerate(cr)],[100*r['upper']-values[i] for i,r in enumerate(cr)]])
        ax.errorbar(xx,values,yerr=errors,fmt='o',markersize=5,color=COLORS[family],capsize=3,label=family)
    ax.axhline(0,color='#8c979e',lw=.8);ax.set_xticks(range(3),['Core (54)','Normal (27)','t5 (27)']);ax.set_ylabel('Power difference (percentage points)');ax.set_title('A  Historical and equally repaired envelopes',loc='left');ax.legend(frameon=False,ncol=3,fontsize=8);ax.grid(axis='y',alpha=.15)
    ax=axes[0,1];indices=[r['case_index'] for r in rows];fdp=100*np.array([r['methods'][MAIN]['fdp'] for r in rows]);scope=[r['case'].get('assumption_class') in ['in_scope','matched_working_model'] for r in rows]
    if not a.development:
        lo=np.array([r['methods'][MAIN]['fdp_interval'][0]*100 for r in rows]);hi=np.array([r['methods'][MAIN]['fdp_interval'][1]*100 for r in rows]);ax.vlines(indices,lo,hi,color='#78828b',lw=.7)
    ax.scatter(np.array(indices)[scope],fdp[scope],s=13,color='#306A8A',label='Preclassified working scope')
    outside=~np.array(scope);ax.scatter(np.array(indices)[outside],fdp[outside],s=24,marker='x',color='#AE5757',label='Retained assumption stresses')
    ax.axhline(5,color='#AE5757',lw=1,ls='--',label='Nominal FDR 5%');ax.set_ylim(-1,max(10,fdp.max()+8));ax.set_xlabel('Frozen scenario index');ax.set_ylabel('Mean FDP / FDR estimate (%)');ax.set_title('B  No averaging away covariance-drift failures',loc='left');ax.legend(frameon=False,fontsize=7,loc='upper left');ax.grid(axis='y',alpha=.15)
    for r in rows:
        if r['methods'][MAIN]['fdp']>.1:ax.annotate(r['case']['name'].replace('normal_drift_null','moderate drift').replace('t5_severe_drift','severe t5 drift'),(r['case_index'],100*r['methods'][MAIN]['fdp']),xytext=(-8,8),textcoords='offset points',ha='right',fontsize=7)
    ax=axes[1,0]
    for dist,marker,color in [('normal','o','#306A8A'),('t5','s','#338475')]:
        rs=[r for r in core if r['case']['distribution']==dist]
        if a.development:baseline=[r['comparisons']['F_empirical']['sample_envelope_method'] for r in rs]
        else:baseline=[max(data['analysis_plan']['families']['F_empirical'],key=lambda m:r['methods'][m]['power']) for r in rs]
        x=[100*r['methods'][b]['power'] for r,b in zip(rs,baseline)];y=[100*r['methods'][MAIN]['power'] for r in rs]
        ax.scatter(x,y,s=24,alpha=.7,marker=marker,color=color,label=dist)
    ax.plot([0,100],[0,100],color='#8c979e',ls='--',lw=.8);ax.set_xlim(-1,102);ax.set_ylim(-1,102);ax.set_xlabel('Same-frontend simple envelope: sample mean Power (%)');ax.set_ylabel('R1 guard Power (%)');ax.set_title('C  Per-scenario utility, not a universal dominance claim',loc='left');ax.legend(frameon=False);ax.grid(alpha=.12)
    ax=axes[1,1];pairs=[('Guard cost: projection','R1B_guard_pilotc0.5_projection_eBH','R1B_plugin_pilotc0.5_projection_eBH','guard_cost_complex'),('Guard cost: Bonferroni','R1B_guard_pilotc0.5_ordinary_bonf_eBH','R1B_plugin_pilotc0.5_ordinary_bonf_eBH','guard_cost_simple'),('Projection - Bonf: guarded','R1B_guard_pilotc0.5_projection_eBH','R1B_guard_pilotc0.5_ordinary_bonf_eBH','projection_extra_guard'),('Projection - Bonf: plugin','R1B_plugin_pilotc0.5_projection_eBH','R1B_plugin_pilotc0.5_ordinary_bonf_eBH','projection_extra_plugin')]
    for i,(label,first,second,key) in enumerate(pairs):
        value=100*np.mean([r['methods'][first]['power']-r['methods'][second]['power'] for r in core]);err=None
        if not a.development:
            c=data['contributions']['core54/'+key];err=np.array([[value-100*c['lower']],[100*c['upper']-value]])
        ax.errorbar(value,i,xerr=err,fmt='o',color='#306A8A',capsize=3)
    ax.axvline(0,color='#8c979e',lw=.8);ax.set_yticks(range(4),[p[0] for p in pairs]);ax.invert_yaxis();ax.set_xlabel('Paired Power difference (percentage points)');ax.set_title('D  Frontend and projection contributions',loc='left');ax.grid(axis='x',alpha=.15)
    stage='DEVELOPMENT ONLY — no confirmation intervals' if a.development else 'FROZEN INDEPENDENT CONFIRMATION — simultaneous bounded intervals'
    fig.suptitle(f'Finite nuisance-bootstrap guard | {a.run_id}\n{stage}',fontsize=14,fontweight='medium')
    fig.savefig(out/'research-panels.png',dpi=180);fig.savefig(out/'research-panels.svg');plt.close(fig)
    if not a.development:
        rs=[r for r in rows if r['case_index'] in data['analysis_plan']['validity_scope_indices']]
        fig,ax=plt.subplots(figsize=(13,4.5),layout='constrained');x=np.array([r['case_index'] for r in rs]);m=np.array([r['methods'][MAIN]['fdp']*100 for r in rs]);lo=np.array([r['methods'][MAIN]['fdp_interval'][0]*100 for r in rs]);hi=np.array([r['methods'][MAIN]['fdp_interval'][1]*100 for r in rs])
        ax.errorbar(x,m,yerr=[m-lo,hi-m],fmt='o',color='#306A8A',markersize=3,capsize=2,lw=.7)
        ax.axhline(5,color='#AE5757',ls='--',label='Nominal FDR5%');ax.set_ylim(-.2,max(5.5,float(hi.max())+.5));ax.set_xlabel('Frozen scenario index');ax.set_ylabel('Mean FDP and simultaneous interval (%)');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
        ax.set_title(f'{a.run_id} | Preclassified working scope only\nFull-range panelB retains every assumption violation; this zoom does not replace it',loc='left')
        fig.savefig(out/'working-scope-fdr.png',dpi=180);fig.savefig(out/'working-scope-fdr.svg');plt.close(fig)
    shutil.copy2(__file__,out/'generate_figure_source.py')
    (out/'provenance.json').write_text(json.dumps({'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'stage':stage,'not_patient_data':True,'note':'Power panels use sample means; error bars only from frozen confirmation plan, when present. FDR display retains all assumption failures. Guard frontend also changes TRAIN pilot, so guard cost is not isolated variance-only effect.'},indent=2),encoding='utf-8')
    print(out/'research-panels.png')
if __name__=='__main__':main()
