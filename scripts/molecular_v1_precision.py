"""Before-confirmation precision planning from EXPOSED DEV, not new results."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from molecular_v1_validation import read,validate_plan,sha
from sca3_compass.robustness_io import write_json
from sca3_compass.robustness_confirmation_bounds import kl_interval


def main():
    plan=read(ROOT/'configs/molecular_v1_confirmation.json');validate_plan(plan)
    source=ROOT/'artifacts/robustness/R0077-fair-diagnostic/result.json'
    data=read(source);delta=.02/(2*3*7);log=np.log(2/delta);rows={}
    for name,lo,hi in [('core',0,54),('normal',0,27),('t5',27,54)]:
        subset=[r for r in data['records'] if lo<=r['case']<hi]
        n=(hi-lo)*512
        rows[name]={}
        for mode in ['conditional_eBH','conditional_BY']:
            differences=np.array([r['candidate']['power']-r['metrics'][mode]['power'] for r in subset])
            var=float(differences.var(ddof=1))
            rows[name][mode]={'DEV_mean':float(differences.mean()),'DEV_pooled_variance':var,
                'planned_independent_families':n,
                'planning_radius_not_actual_CI':float(np.sqrt(2*var*log/n)+14*log/(3*(n-1)))}
    out=ROOT/'artifacts/robustness/V1-reference/precision-plan.json'
    if out.exists():raise FileExistsError('Preserve plan')
    write_json(out,{'kind':'PRE_CONFIRMATION_PLANNING_NOT_RESULTS','plan_sha256':sha(ROOT/'configs/molecular_v1_confirmation.json'),
        'DEV_source_sha256':sha(source),'paired':rows,
        'hypothetical_FDP_intervals':{str(n):{str(mean):list(kl_interval(mean,n,.02/(2*84*8))) for mean in [0.,.005,.01,.02]} for n in [512,1024]},
        'fixed_total':sum(plan['counts']),'compute_basis':'R0077 2688 joint families in134s on8 workers; roughly49min scaled, not deadline or guarantee'})
    print(out)


if __name__=='__main__':main()
