"""Exact algebraic replay of D015/D016_2; zero new samples or inference tuning."""
import gzip,json
import numpy as np
from r5_common import PHASE,read,write,sha
from focused_reference import components

def main():
    out=PHASE/'checks/focused-attribution.json'
    write(PHASE/'checks/focused-attribution-protocol.json',{'source':'ALL20 D016_2 andD015',
        'checks':['Emax=(1+Nselected)/(1+Nunion)*Eselected','Qminusboundary andboundaryscoreloss',
                  'conditional scalar dominance','actual decision identities'],
        'no_new_draws':True,'script_sha':sha(__file__)})
    rows=[]
    for r in read(PHASE/'D016_2/result.json')['rows']:
        c,rep=r['case'],r['rep'];p=PHASE/f'D015/raw/case-{c:02}/rep-{rep:05}.json.gz'
        old=json.load(gzip.open(p,'rt',encoding='utf8'));v=old['candidate'];ref=v['reference'];m=ref['meta'];q=ref['thresholds'];u=v['calibration']['observed_u']
        with np.load(PHASE/'D015'/old['evidence_path'],allow_pickle=False) as f:arr=dict(f)
        with np.load(PHASE/f'D016_2/case-{c:02}-rep-{rep:02}.npz',allow_pickle=False) as f:maxarr=dict(f)
        choice='variance_bridge' if np.log(q['variance_bridge'])+m['a']*u/2<=np.log(q['target']) else 'target'
        union=r['reference_counts']['union'];selected=ref['totals'][choice]
        factor=(1+selected)/(1+union)
        difference=float(np.max(np.abs(maxarr['e']-factor*arr['e_focused_'+choice])))
        assert np.allclose(maxarr['e'],factor*arr['e_focused_'+choice],rtol=3e-15,atol=1e-12)
        tt=arr['reference_pivot']*np.exp(-arr['reference_error_target']/2)
        uu=arr['reference_error_source']-arr['reference_error_target']
        b,a=components(tt,uu,0.,m['a'],m['c'],m['tau'],q['target'],q['variance_bridge'])
        boundary=float(np.sum(b+a));Q=ref['Q'];dominates=Q>=union
        if dominates:
            assert np.all(maxarr['e']+1e-10>=arr['e_focused_joint'])
            assert np.all(~arr['decision_focused_joint']|maxarr['decision'])
        rows.append({'case':c,'rep':rep,'chosen_endpoint':choice,'reference_selected':selected,
            'reference_union':union,'union_e_factor':factor,'max_identity_absolute_error':difference,
            'boundary_score_sum':boundary,'soft_boundary_saving':union-boundary,
            'profile_and_cover_cost':Q-boundary,'cover_excess_over_grid_lower':Q-ref['cover']['evaluated_lower_sum'],
            'net_reference_saving':union-Q,'soft_weight':v['calibration']['borrow_probability'],
            'structural_max_dominance':bool(dominates),
            'decisions_identical':bool(np.array_equal(maxarr['decision'],arr['decision_focused_joint']))})
    result={'status':'DETERMINISTIC_IDENTITIES_CHECKED_NOT_G1_OR_G3','rows':rows,
        'n_identical_decisions':sum(r['decisions_identical'] for r in rows),
        'n_structural_max_dominance':sum(r['structural_max_dominance'] for r in rows),
        'max_identity_absolute_error':max(r['max_identity_absolute_error'] for r in rows)}
    write(out,result)
    print({k:v for k,v in result.items() if k!='rows'},flush=True)
    print([r for r in rows if r['case']==1],flush=True)

if __name__=='__main__':main()
