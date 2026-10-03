"""D010 deterministic Rao-Blackwell attribution on existing D009 outputs."""
import gzip,json,numpy as np
from r5_common import PHASE,R4,read,write,sha,fc,exp

def main():
    source=PHASE/'D009';out=PHASE/'D010';out.mkdir()
    write(out/'protocol.json',{'stage':'DETERMINISTIC_COMPONENT_ATTRIBUTION_NOT_CONFIRMATION',
        'source_freeze_sha':sha(source/'freeze.json'),'source_index_sha':sha(source/'index.json'),
        'script_sha':sha(__file__),'units':'all80 existingD009families','stop':'exactly80, no sampling or fitting',
        'purpose':'Compare integrated-coin mixture to exact expected performance of its conceptual randomized selector; NOT standalone conditional-component validity.'})
    rows=[]
    for item in read(source/'index.json')['rows']:
        p=source/item['path']
        if sha(p)!=item['sha']:raise ValueError('raw hash')
        r=json.load(gzip.open(p,'rt',encoding='utf8'))
        if sha(source/r['evidence_path'])!=r['evidence_sha']:raise ValueError('evidence hash')
        with np.load(source/r['evidence_path'],allow_pickle=False) as f:e=f['e'];ce=f['component_e']
        a=r['source_artifacts']['diagnostic'];d=R4/'C001'/a['path']
        if sha(d)!=a['sha']:raise ValueError('truth diagnostic hash')
        with np.load(d,allow_pickle=False) as f:truth=f['truth']
        g=r['calibration']['borrow_probability']
        if not np.array_equal(e,g*ce[0]+(1-g)*ce[1]):raise ValueError('mixture replay mismatch')
        b=exp.score(fc.ebh(ce[0],.05),truth);t=exp.score(fc.ebh(ce[1],.05),truth)
        conditional_randomized={k:None if b[k] is None else g*b[k]+(1-g)*t[k] for k in b}
        # This is the conditional EXPECTATION of a random-coin procedure. Do not
        # average rejection sets or pretend component methods are valid alone.
        rows.append({'case':r['case'],'rep':r['rep'],'mix':r['metrics']['A0'],
          'randomized_conditional_expectation':conditional_randomized,'probability':g,
          'component_bridge_DIAGNOSTIC_ONLY':b,'component_target_DIAGNOSTIC_ONLY':t,
          'acceptance_bridge':r['reference_receipt']['branch_bridge']['conditional_pair_sampling']['rate'],
          'acceptance_target':r['reference_receipt']['branch_target']['conditional_pair_sampling']['rate'],
          'candidate_seconds':r['seconds']})
    write(out/'raw.json',rows)
    summary=[]
    for c in range(10):
        rr=[r for r in rows if r['case']==c];yes=rr[0]['mix']['power'] is not None
        x={'case':c,'n':8,'power_mix':np.mean([r['mix']['power'] for r in rr]) if yes else None,
          'power_randomized_expected':np.mean([r['randomized_conditional_expectation']['power'] for r in rr]) if yes else None,
          'fdp_mix_mean':np.mean([r['mix']['fdp'] for r in rr]),
          'fdp_randomized_expected_mean':np.mean([r['randomized_conditional_expectation']['fdp'] for r in rr]),
          'mean_borrow_probability':np.mean([r['probability'] for r in rr]),
          'mean_acceptance_target':np.mean([r['acceptance_target'] for r in rr]),
          'mean_acceptance_bridge':np.mean([r['acceptance_bridge'] for r in rr])}
        summary.append(x);print(json.dumps(x),flush=True)
    write(out/'summary.json',{'rows':summary,'all_saved_e_exactly_reproduced':True,
         'warning':'Only80distinctexistingDEVfamilies; no independent confirmation. Conditional components are not standalone valid methods.'})

if __name__=='__main__':main()
