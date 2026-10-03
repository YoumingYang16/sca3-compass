"""Regenerate selected DEVELOPMENT failure-mechanism figures from raw JSON.

Selected cases illustrate mechanisms and retained counterexamples. They are
not a complete benchmark, independent confirmation, or acceptance decision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=Path('artifacts/robustness'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    outputs=[args.output.with_suffix(s) for s in ['.png','.svg','.json']]
    if any(p.exists() for p in outputs):
        raise SystemExit('Output exists; select a new figure prefix')
    specifications=[
        ('R0031','t5_tail_shift_null','Calibration-tail mismatch','fdp_by_repetition',[
            ('capped_weighted_cone_f0.0_eBH','Transferred prior'),
            ('capped_target_only_weighted_cone_f0.0_eBH','Target residual prior')]),
        ('R0034','t5_reverse_covariance_signal','Severe covariance shift: useful power','power_by_repetition',[
            ('capped_target_only_weighted_cone_f0.0_eBH','Uncorrected (invalid FDP)'),
            ('capped_gated_point_target_only_weighted_cone_eBH','Quantile correction'),
            ('capped_gated_upper_target_only_weighted_cone_eBH','Conservative bound'),
            ('capped_gated_fcentral_target_only_weighted_cone_eBH','Central-F correction')]),
        ('R0036','normal_moderate_cov_null','Retained counterexample: moderate shift','fdp_by_repetition',[
            ('capped_target_only_weighted_cone_f0.0_eBH','Uncorrected'),
            ('capped_gated_fcentral_target_only_weighted_cone_eBH','Fourfold gate'),
            ('capped_always_fcentral_target_only_weighted_cone_eBH','Always-on central F')]),
        ('R0037','normal_rho0.95','Cost in a matched light-tail regime','power_by_repetition',[
            ('capped_target_only_weighted_cone_f0.0_eBH','Target residual prior'),
            ('capped_gated_fcentral_target_only_weighted_cone_eBH','Fourfold gate'),
            ('capped_always_fcentral_target_only_weighted_cone_eBH','Always-on central F')])]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none'})
    fig,axes=plt.subplots(2,2,figsize=(12,8.6),layout='constrained')
    evidence=[]
    colors=['#687582','#23647d','#ba7142','#42806a']
    for ax,(run,case_name,title,metric,methods) in zip(axes.flat,specifications):
        path=args.directory/f'{run}-screen.json'
        document=json.loads(path.read_text(encoding='utf-8'))
        assert document['settings']['phase']=='DEVELOPMENT_ONLY'
        case=next(s for s in document['scenarios'] if s['case']['name']==case_name)
        rows={row['method']:row for row in case['rows']}
        data=[]
        for i,(method,label) in enumerate(methods):
            raw=np.asarray(rows[method][metric],float)
            mean=float(raw.mean())
            ax.barh(i,mean,color=colors[i],height=.58)
            text_position=mean+.035
            info={'method':method,'label':label,'mean':mean,'mean_fdp':float(np.mean(rows[method]['fdp_by_repetition']))}
            if metric=='fdp_by_repetition':
                assert case['replicated_signed_truths']==0 and np.isin(raw,[0,1]).all()
                successes=int(raw.sum());n=len(raw)
                low=0 if successes==0 else float(beta.ppf(.025,successes,n-successes+1))
                high=1 if successes==n else float(beta.ppf(.975,successes+1,n-successes))
                ax.errorbar(mean,i,xerr=np.array([[mean-low],[high-mean]]),fmt='none',ecolor='#192a34',capsize=3)
                info.update(pointwise_95_clopper_pearson=[low,high])
                text_position=high+.022
                label_text=f'{mean:.3f}'
            else:
                label_text=f'{mean:.3f}  (FDP {info["mean_fdp"]:.3f})'
            ax.text(min(text_position,1.02),i,label_text,va='center',fontsize=9)
            data.append(info)
        ax.set_yticks(range(len(methods)),[label for _,label in methods])
        ax.invert_yaxis();ax.set_title(title,loc='left',pad=12)
        ax.set_xlim(0,1.4 if metric=='power_by_repetition' else .65)
        ax.set_xticks(np.arange(0,1.01,.25) if metric=='power_by_repetition' else [0,.05,.2,.4,.6])
        if metric=='fdp_by_repetition':
            ax.set_xticklabels(['0','.05','.2','.4','.6'])
        ax.set_xlabel('Mean signed power; FDP shown for context' if metric=='power_by_repetition' else 'Mean signed FDP under the global null')
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        if metric=='fdp_by_repetition':
            ax.axvline(.05,color='#a03d40',linestyle='--',lw=1,label='Nominal 0.05')
        evidence.append({'run':run,'case':case_name,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'repetitions':case['repetitions'],'metric':metric,'values':data})
    fig.suptitle('Robustness development: improvements and retained tradeoffs',fontsize=17,ha='left',x=.02)
    fig.supxlabel('Selected development cases only. Null error bars: pointwise 95% exact binomial intervals.\nPower bars are repetition means, not confidence intervals. No independent confirmation or acceptance claim.',fontsize=9)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(outputs[0],dpi=180);fig.savefig(outputs[1]);plt.close(fig)
    outputs[2].write_text(json.dumps({'phase':'DEVELOPMENT_ONLY','illustrative_selection':True,
        'independent_confirmation':False,'panels':evidence},indent=2),encoding='utf-8')
    print('\n'.join(str(p.resolve()) for p in outputs))


if __name__=='__main__':
    main()
