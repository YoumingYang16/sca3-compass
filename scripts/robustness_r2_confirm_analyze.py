"""Complete-only frozen C2 analysis; no simulations or sequential inference."""
from research_window import wait_start_gate,cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime,timezone
import argparse
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from sca3_compass.robustness_io import content_digest,write_json
from sca3_compass.robustness_confirmation_bounds import envelope_interval,kl_interval
from sca3_compass.robustness_paired_bounds import paired_interval
from robustness_r2_development import read,sha
from robustness_r2_analyze import rescore
from robustness_r2_confirmation import validate


def distribution(x):
    return {'sd':float(x.std(ddof=1)),'q05':float(np.quantile(x,.05)),
        'q50':float(np.median(x)),'q95':float(np.quantile(x,.95)),
        'q99':float(np.quantile(x,.99)),'max':float(x.max())}


def analyze(root):
    started=time.perf_counter();base=root/'artifacts/robustness'
    ip=base/'R0076-results-index.json';index=read(ip);protocol=read(base/'R0076-screen.protocol.json')
    plan=protocol['analysis_plan'];validate(plan)
    digest=content_digest(protocol)
    if not index['complete'] or len(index['receipts'])!=82800 or content_digest(index['settings'])!=digest:
        raise ValueError('Complete matching fixed C2 required; no interim analysis')
    for rel,check in protocol['source_sha256'].items():
        if sha(ROOT/rel)!=check:raise ValueError('Frozen analysis/method dependency changed:'+rel)
    for rel,check in plan['development_evidence_sha256'].items():
        if sha(root/rel)!=check:raise ValueError('DEV mapping evidence changed')
    target=base/'R0076-confirmation-analysis.json'
    if target.exists():raise FileExistsError('Preserve completed analysis')
    grouped={i:[] for i in range(84)}
    for receipt in index['receipts']:
        if receipt['status']!='completed':raise ValueError('Failed family cannot be dropped')
        grouped[receipt['case_index']].append(receipt)
    for i,records in grouped.items():
        records.sort(key=lambda r:r['rep'])
        if len(records)!=plan['repetition_counts'][i] or [r['rep'] for r in records]!=list(range(len(records))):
            raise ValueError('Missing/duplicated family')
    names=plan['reported_methods'];mi={m:j for j,m in enumerate(names)}
    main,ref=plan['candidate'],plan['reference'];alloc=plan['error_allocation']
    dprimary=alloc['FDR_candidate_and_reference']/(2*84*2)
    dother=alloc['FDR_other']/((len(names)-2)*84*2)
    dlocal=alloc['I_local']/(2*len(plan['power_defined_indices']))
    power={};rows=[];bytes_checked=0;cpu=wall=0.;failed_r1=[];failed_k=[]
    store=base/'R0076-analysis-arrays';store.mkdir(exist_ok=False)
    for i,receipts in grouped.items():
        if cooperative_stop():raise InterruptedError('Analysis finite resource stop; no final inference')
        n=len(receipts);p=np.empty((n,len(names)));f=np.empty_like(p);tp=np.empty_like(p);fp=np.empty_like(p)
        r1fails=kfails=patternfails=reused=0;defined=None
        rhos=[];gammas=[]
        for r,receipt in enumerate(receipts):
            if sha(receipt['path'])!=receipt['sha256']:raise ValueError('Record checksum mismatch')
            record=read(receipt['path'])
            expected={'run_id':'R0076','phase':'C2','method_version':protocol['method_version'],
                'protocol_digest':digest,'case_index':i,'rep':r,'seed_sequence':[protocol['seed'],i,r]}
            if record['status']!='completed' or any(record.get(k)!=v for k,v in expected.items()):
                raise ValueError('Scientific record identity conflict')
            for k in ['input','evidence']:
                if sha(record[k+'_path'])!=record[k+'_sha256']:raise ValueError('Array checksum mismatch')
                bytes_checked+=Path(record[k+'_path']).stat().st_size
            if set(record['metrics'])!=set(names):raise ValueError('Comparator set changed')
            rescore(record,names)
            is_defined=bool(record['n_true_signed'])
            if is_defined!=(i in plan['power_defined_indices']) or (defined is not None and defined!=is_defined):
                raise ValueError('Power estimand changed')
            defined=is_defined
            for m,j in mi.items():
                metric=record['metrics'][m]
                p[r,j]=metric['power'];f[r,j]=metric['fdp'];tp[r,j]=metric['tp'];fp[r,j]=metric['fp']
            for fold in record['diagnostics_R1']['folds']:
                if not fold['guard_success']:
                    r1fails+=1;failed_r1.append({'case':i,'rep':r,'fold':fold['fold'],'record':receipt})
            for fold in record['diagnostics_R2']['folds']:
                if not fold['K_success']:
                    kfails+=1;failed_k.append({'case':i,'rep':r,'fold':fold['fold'],'record':receipt})
                patternfails+=int(fold['pattern']['fallback'])
                rhos.append([fold['original']['rho'],fold['guarded']['rho']])
                gammas.extend(fold['gamma'])
            reused+=record['numerical_reuse']=='FRESH_SAME_FAMILY_R1_INTERMEDIATES'
            cpu+=record['cpu_seconds'];wall+=record['elapsed_seconds']
        array_path=store/f'case-{i:04}.npz'
        with array_path.open('xb') as stream:np.savez_compressed(stream,power=p,fdp=f,tp=tp,fp=fp,methods=names)
        row={'case_index':i,'case':protocol['cases'][i],'n':n,'power_defined':defined,'methods':{},
            'R1_failed_folds':r1fails,'K_failed_folds':kfails,'pattern_fallback_folds':patternfails,
            'same_family_memoized_families':reused,'analysis_array_path':str(array_path),'analysis_array_sha256':sha(array_path),
            'rho_mean_original_guarded':np.mean(rhos,axis=0).tolist(),'gamma_mean':float(np.mean(gammas))}
        for m,j in mi.items():
            row['methods'][m]={'power':float(p[:,j].mean()) if defined else None,
                'power_mcse':float(p[:,j].std(ddof=1)/np.sqrt(n)) if defined else None,
                'fdp':float(f[:,j].mean()),'fdp_interval':list(kl_interval(float(f[:,j].mean()),n,dprimary if m in [main,ref] else dother)),
                'fdp_mcse':float(f[:,j].std(ddof=1)/np.sqrt(n)),
                'mean_tp':float(tp[:,j].mean()),'mean_fp':float(fp[:,j].mean()),
                'no_discoveries_fraction':float(np.mean((tp[:,j]+fp[:,j])==0)),
                'FDP_gt_05_fraction':float(np.mean(f[:,j]>.05))}
            if m in [main,ref,'R1B_plugin_pilotc0.5_projection_gate_eBH']:
                row['methods'][m]['power_distribution']=distribution(p[:,j]) if defined else None
                row['methods'][m]['fdp_distribution']=distribution(f[:,j])
        row['I']=paired_interval([p[:,mi[main]]-p[:,mi[ref]]],dlocal) if defined else None
        power[i]=p;rows.append(row)
        write_json(base/'R0076-analysis-progress.json',{'verified_cases':len(rows),'planned_cases':84,
            'updated_utc':datetime.now(timezone.utc).isoformat(),'status':'RAW_HASH_AND_SCORE_AUDIT_ONLY'})
        print('Verified C2 case',i,n,flush=True)
    comparisons={};increments={};core_stats={}
    de=alloc['envelopes']/(2*len(plan['strata'])*len(plan['families']))
    di=alloc['I_strata']/(2*len(plan['strata']))
    for stratum,indices in plan['strata'].items():
        increments[stratum]=paired_interval([power[i][:,mi[main]]-power[i][:,mi[ref]] for i in indices],di)
        core_stats[stratum]={m:{'power':float(np.mean([rows[i]['methods'][m]['power'] for i in indices])),
            'mean_tp':float(np.mean([rows[i]['methods'][m]['mean_tp'] for i in indices]))} for m in names}
        for family,members in plan['families'].items():
            diffs=[power[i][:,mi[main],None]-power[i][:,[mi[m] for m in members]] for i in indices]
            fixed=[members.index(plan['development_selected_baselines'][family][str(i)]) for i in indices]
            comparisons[stratum+'/'+family]=envelope_interval(diffs,fixed,de,de)
    scope={}
    for label,indices in [('original68',plan['original_validity_scope_indices']),('expanded8',plan['expanded_matched_indices'])]:
        scope[label]={m:{'all_FDR_upper_le_05':all(rows[i]['methods'][m]['fdp_interval'][1]<=.05 for i in indices),
            'max_FDR':max(rows[i]['methods'][m]['fdp'] for i in indices),
            'max_FDR_upper':max(rows[i]['methods'][m]['fdp_interval'][1] for i in indices)} for m in [main,ref,'R1B_plugin_pilotc0.5_projection_gate_eBH']}
    incremental=(increments['core54']['lower']>0 and all(scope[s][main]['all_FDR_upper_le_05'] for s in scope))
    broad=all(comparisons[s+'/'+f]['lower']>0 for s in ['core54','t5_27'] for f in ['H','F_all_same_information']) and incremental
    result={'run_id':'R0076','phase':'C2','method_version':protocol['method_version'],'complete':True,
        'index_sha256':sha(ip),'protocol_digest':digest,'analysis_script_sha256':sha(__file__),
        'analysis_plan':plan,'rows':rows,'comparisons':comparisons,'I':increments,'stratum_methods':core_stats,
        'scoped_validity':scope,'incremental_statistical_criteria_pass':incremental,
        'broad_success_criteria_pass':broad,'retention_decision':'PENDING_REPLAY_AND_MAIN_FAILURE_COST_REVIEW',
        'whole_family_repetitions':82800,'whole_family_failures':0,'input_evidence_bytes_verified':bytes_checked,
        'R1_failed_folds':failed_r1,'K_failed_folds':failed_k,'pattern_fallback_folds':sum(r['pattern_fallback_folds'] for r in rows),
        'recorded_cpu_seconds':cpu,'summed_family_wall_seconds':wall,'generator_wall_seconds':index['elapsed_seconds'],
        'analysis_elapsed_seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'provenance':'SIMULATION_NOT_PATIENT_DATA','reporting_error_budget':.025,
        'not_claimed':['general estimated-nuisance validity','clinical benefit','novelty','all-heavy-tail success','universal baseline superiority'],
        'interval_note':'C2 endpoints jointly spend.025; original frozen C1 .025 unchanged. Quantiles/MCSE are descriptive, genes are not independent repetitions.'}
    with target.open('x',encoding='utf-8') as stream:
        import json
        json.dump(result,stream,indent=2,ensure_ascii=False)
    print('Complete C2',target,'incremental',incremental,'broad',broad,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=ROOT)
    analyze(parser.parse_args().project_root.resolve())
