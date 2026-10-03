"""Execute frozen analysis ONCE after a complete fixed-n confirmation batch.

Does not create simulations, select new candidates, or promote incomplete runs.
Outputs finite-n selection-aware H/F intervals and simultaneous per-scene FDP
intervals. Scientific validity is still scoped empirical, not inferred fromE.
"""
from pathlib import Path
import argparse,gzip,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from sca3_compass.robustness_confirmation_bounds import envelope_interval,kl_interval
from sca3_compass.robustness_confirmation_protocol import validate_plan
from sca3_compass.robustness_io import content_digest

def read(p):
    with (gzip.open(p,'rt',encoding='utf-8') if str(p).endswith('.gz') else Path(p).open(encoding='utf-8')) as s:return json.load(s)
def sha(p):
    with Path(p).open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()

def audit_arrays(record,names):
    with np.load(record['input_path']) as z:truth=z['truth'].ravel().astype(bool)
    with np.load(record['evidence_path']) as z:e=np.array([z[name].ravel() for name in names])
    if e.shape[1]!=truth.size or not np.isfinite(e).all() or np.any(e<0):raise ValueError('Invalid stored evidence')
    ranks=np.arange(1,truth.size+1);ordered=np.sort(e,axis=1)[:,::-1]
    k=np.where(ordered>=truth.size/(.05*ranks),ranks,0).max(1)
    cutoff=np.where(k>0,truth.size/(.05*np.maximum(k,1)),np.inf);rejected=e>=cutoff[:,None]
    total=rejected.sum(1);fp=(rejected&~truth).sum(1);tp=(rejected&truth).sum(1)
    for j,m in enumerate(names):
        expected={'discoveries':int(total[j]),'tp':int(tp[j]),'fp':int(fp[j]),'power':float(tp[j]/max(1,truth.sum())),'fdp':float(fp[j]/max(1,total[j]))}
        for key,value in expected.items():
            if record['metrics'][m][key]!=value:raise ValueError('Independent eBH/truth scoring mismatch:'+m+'/'+key)
    if int(truth.sum())!=record['n_true_signed']:raise ValueError('Truth count mismatch')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--project-root',type=Path,default=ROOT);args=p.parse_args();folder=args.project_root.resolve()/'artifacts/robustness'
    index=read(folder/f'{args.run_id}-results-index.json');protocol=index['settings']
    if content_digest(protocol)!=content_digest(read(folder/f'{args.run_id}-screen.protocol.json')):raise ValueError('Index/frozen protocol mismatch')
    if not index['complete'] or protocol['phase'] not in ['C1','C2']:raise ValueError('Complete fixed-n independent confirmation only')
    plan=protocol['analysis_plan']
    validate_plan(plan,protocol['cases'])
    if protocol.get('H_baselines') is not None and plan['families']['H']!=protocol['H_baselines']:raise ValueError('Historical envelope mismatch')
    if sum(plan['error_allocation'].values())>plan['reporting_error_budget']+1e-14:raise ValueError('Error overspend')
    if protocol['source_sha256']['scripts/robustness_r1_confirm.py']!=sha(__file__):raise ValueError('Analysis code changed since freeze; execute archived source')
    for rel,digest in protocol['source_sha256'].items():
        if sha(ROOT/rel)!=digest:raise ValueError('Frozen source mismatch:'+rel)
    dev=folder/f"{plan['development_run']}-summary.json"
    if sha(dev)!=plan['development_summary_sha256']:raise ValueError('DEV mapping source hash mismatch')
    grouped={i:[] for i in range(len(protocol['cases']))}
    for receipt in index['receipts']:
        if receipt['status']!='completed' or 'case_index' not in receipt or 'rep' not in receipt:raise ValueError('Full indexed confirmation receipts required')
        grouped[receipt['case_index']].append(receipt)
    for i,receipts in grouped.items():
        if len(receipts)!=protocol['repetition_counts'][i] or {r['rep'] for r in receipts}!=set(range(len(receipts))):raise ValueError('Incomplete or duplicate sample')
        receipts.sort(key=lambda r:r['rep'])
    names=plan['reported_methods'];main=plan['candidate'];methods={m:j for j,m in enumerate(names)}
    power={};fdp={};rows=[];array_bytes_checked=0
    delta_primary=plan['error_allocation']['FDR_candidate']/(2*len(grouped))
    delta_other=plan['error_allocation']['FDR_other']/(2*len(grouped)*(len(names)-1))
    for i,receipts in grouped.items():
        n=len(receipts);power[i]=np.empty((n,len(names)));fdp[i]=np.empty_like(power[i]);tp=np.empty_like(power[i]);fp=np.empty_like(power[i]);fallbacks=0;defined=None
        for pos,receipt in enumerate(receipts):
            if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Record checksum mismatch')
            r=read(receipt['path'])
            if r['status']!='completed' or r['case_index']!=i or r['rep']!=pos:raise ValueError('Wrong repetition record')
            identity={'run_id':args.run_id,'phase':protocol['phase'],'method_version':protocol['method_version'],
                'protocol_digest':content_digest(protocol),'seed_sequence':[protocol['seed'],i,pos]}
            if any(r.get(key)!=value for key,value in identity.items()):raise ValueError('Record belongs to different frozen experiment')
            for kind in ['input','evidence']:
                if sha(r[kind+'_path'])!=r[kind+'_sha256']:raise ValueError('Array checksum mismatch')
                array_bytes_checked+=Path(r[kind+'_path']).stat().st_size
            audit_arrays(r,names)
            for m,j in methods.items():
                metric=r['metrics'][m];power[i][pos,j]=metric['power'];fdp[i][pos,j]=metric['fdp'];tp[pos,j]=metric['tp'];fp[pos,j]=metric['fp']
                if not np.isfinite([metric['power'],metric['fdp']]).all() or not 0<=metric['power']<=1 or not 0<=metric['fdp']<=1:raise ValueError('Unbounded family metric')
            fallbacks+=sum(f['fallback'] for f in r['diagnostics_R1']['folds'])
            if defined is not None and defined!=bool(r['n_true_signed']):raise ValueError('Random undefined-power cases require separate estimand')
            defined=bool(r['n_true_signed'])
            if i in plan['strata']['core54'] and not defined:raise ValueError('Undefined power entered primary core')
        row={'case_index':i,'case':protocol['cases'][i],'n':n,'methods':{},'fallback_folds':fallbacks}
        for m,j in methods.items():
            interval=kl_interval(float(fdp[i][:,j].mean()),n,delta_primary if m==main else delta_other)
            row['methods'][m]={'power':float(power[i][:,j].mean()) if defined else None,
                'power_mcse':float(power[i][:,j].std(ddof=1)/np.sqrt(n)),
                'fdp':float(fdp[i][:,j].mean()),'fdp_interval':list(interval),'fdp_mcse':float(fdp[i][:,j].std(ddof=1)/np.sqrt(n)),
                'mean_tp':float(tp[:,j].mean()),'mean_fp':float(fp[:,j].mean())}
        rows.append(row)
        print('Verified complete case',i,n,flush=True)
    comparisons={};delta_side=plan['error_allocation']['envelopes']/(2*len(plan['strata'])*len(plan['families']))
    for stratum,indices in plan['strata'].items():
        for family,baseline_names in plan['families'].items():
            bindices=[methods[m] for m in baseline_names];differences=[power[i][:,methods[main],None]-power[i][:,bindices] for i in indices]
            fixed=[baseline_names.index(plan['development_selected_baselines'][family][str(i)]) for i in indices]
            comparisons[stratum+'/'+family]=envelope_interval(differences,fixed,delta_side,delta_side)
    contributions={};cdelta=plan['error_allocation']['contributions']/(2*len(plan['strata'])*len(plan['contribution_pairs']))
    for stratum,indices in plan['strata'].items():
        for label,(a,b) in plan['contribution_pairs'].items():
            x=[(power[i][:,methods[a]]-power[i][:,methods[b]])[:,None] for i in indices]
            contributions[stratum+'/'+label]=envelope_interval(x,[0]*len(x),cdelta,cdelta)
    result={'run_id':args.run_id,'phase':protocol['phase'],'method_version':protocol['method_version'],
        'index_sha256':sha(folder/f'{args.run_id}-results-index.json'),'analysis_plan':plan,
        'analysis_script_sha256':sha(__file__),'complete':True,'rows':rows,'comparisons':comparisons,
        'contributions':contributions,'whole_family_repetitions':len(index['receipts']),'input_evidence_bytes_verified':array_bytes_checked,
        'in_scope_candidate_fdr_all_upper_le_05':all(rows[i]['methods'][main]['fdp_interval'][1]<=.05 for i in plan['validity_scope_indices']),
        'provenance':protocol.get('provenance','TEST_FIXTURE'),'scope':'Limited simulation evidence only; bootstrap guard has no general finite-sample input-validity theorem',
        'not_claimed':['clinical effectiveness','novelty','general heavy-tail solution','protected historical2.4pp']}
    target=folder/f'{args.run_id}-confirmation-analysis.json'
    with target.open('x',encoding='utf-8') as s:json.dump(result,s,indent=2,ensure_ascii=False)
    for label,r in comparisons.items():print(label,{k:round(100*r[k],4) for k in ['sample_envelope_difference','lower','upper']})
    print('In-scope candidate FDR upper<=.05',result['in_scope_candidate_fdr_all_upper_le_05'],'fullfamilies',len(index['receipts']))
    print(target)

if __name__=='__main__':main()
