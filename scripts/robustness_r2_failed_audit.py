"""All-attempted, failure-preserving DESCRIPTIVE audit of failed frozen C2.

No replacement, imputation-as-observation, new CI or confirmation promotion.
Original complete-only analyzer and all scientific files remain untouched.
"""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import gzip
import json
import shutil
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes())) if path.suffix == '.gz' else json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completion_bounds(observed, planned):
    """Algebraic sample-completion range only, NOT a population interval."""
    x = np.asarray(observed, float)
    if planned < len(x) or planned <= 0 or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError('Valid bounded observed scores and planned count required')
    total=float(x.sum())
    return [total/planned, (total+(planned-len(x)))/planned]


def paired_macro(arrays):
    if not arrays or any(len(x) < 2 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Only fully observed predetermined scenes')
    return {'mean_difference': float(np.mean([x.mean() for x in arrays])),
            'paired_mcse': float(np.sqrt(sum(x.var(ddof=1)/len(x) for x in arrays))/len(arrays)),
            'families': sum(len(x) for x in arrays),
            'role': 'DESCRIPTIVE_ONLY_NOT_COMPLETED_C2_CONFIRMATION'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path);args=parser.parse_args()
    started = time.perf_counter()
    base = ROOT/'artifacts/robustness'; snapshot = base/'R0076-source'
    protocol_path = base/'R0076-screen.protocol.json'; protocol = read(protocol_path)
    index_path = base/'R0076-results-index.json'; index = read(index_path)
    progress = read(base/'R0076-progress.json')
    if progress['completed']+progress['failed'] != 82800 or not progress['failed'] or len(index['receipts']) != 82800:
        raise ValueError('All fixed attempts including failures must have finished')
    if index['complete']:
        raise ValueError('Use original confirmation analyzer for a complete successful batch')
    for rel, digest in protocol['source_sha256'].items():
        if sha(snapshot/rel) != digest:
            raise ValueError('Frozen source changed: '+rel)
    sys.path[:0] = [str(snapshot/'src'), str(snapshot/'scripts')]
    from sca3_compass.robustness_io import content_digest, write_json
    from robustness_r2_analyze import rescore
    from robustness_r1_window import generate
    if content_digest(index['settings']) != content_digest(protocol):
        raise ValueError('Index/protocol mismatch')
    plan = protocol['analysis_plan']; names = plan['reported_methods']; pos = {m:j for j,m in enumerate(names)}
    candidate, reference = plan['candidate'], plan['reference']
    out = args.out.resolve() if args.out else base/'R0076-failed-audit'
    if not out.resolve().is_relative_to(base.resolve()) or out.exists():
        raise ValueError('NEW output inside artifacts/robustness required')
    out.mkdir(exist_ok=False)
    progress_path = out/'progress.json' if args.out else base/'R0076-descriptive-progress.json'
    shutil.copy2(__file__, out/'audit_source.py')
    arrays = out/'arrays'; arrays.mkdir()
    grouped = {i:[] for i in range(84)}
    for receipt in index['receipts']:
        grouped[receipt['case_index']].append(receipt)
    rows = []; powers = {}; soft_groups = {}; failures = []; soft_r1 = []; soft_k = []
    verified_bytes = 0; cpu = wall = 0.; total_success = 0
    for i in range(84):
        if cooperative_stop():
            raise InterruptedError('Bounded descriptive audit stop; preserve partial output')
        receipts = sorted(grouped[i], key=lambda r:r['rep']); n = plan['repetition_counts'][i]
        if len(receipts) != n or [r['rep'] for r in receipts] != list(range(n)):
            raise ValueError('Cannot omit or duplicate an attempted family')
        power = np.full((n,len(names)), np.nan); fdp = power.copy(); tp = power.copy(); fp = power.copy()
        ok = np.zeros(n,bool); groups = np.full(n,'hard_failure',dtype='<U20')
        for r, receipt in enumerate(receipts):
            path = Path(receipt['path'])
            if sha(path) != receipt['sha256']:
                raise ValueError('Original record changed')
            record = read(path)
            expected = {'run_id':'R0076','phase':'C2','method_version':protocol['method_version'],
                        'case_index':i,'rep':r,'seed_sequence':[protocol['seed'],i,r],
                        'protocol_digest':content_digest(protocol)}
            if any(record.get(k) != v for k,v in expected.items()) or record['status'] != receipt['status']:
                raise ValueError('Record identity changed')
            cpu += record['cpu_seconds']; wall += record['elapsed_seconds']
            if record['status'] != 'completed':
                ip = path.parent/f'rep-{r:06}-input.npz'
                with np.load(ip,allow_pickle=False) as saved:
                    expected_arrays = generate(protocol['seed'],i,r,protocol['cases'][i])
                    exact = all(np.array_equal(saved[k],a) for k,a in zip(['z','calibration','truth'],expected_arrays))
                if not exact:
                    raise ValueError('Failed input integrity mismatch')
                verified_bytes += ip.stat().st_size
                failures.append({'case':i,'rep':r,'record':str(path),'record_sha256':sha(path),
                    'input_path':str(ip),'input_sha256':sha(ip),'input_regeneration_exact':True,
                    'error':record['error'],'traceback':record['traceback'],
                    'missing_outputs':'Joint runner stopped; do not attribute failure to every comparator'})
                continue
            for key in ['input','evidence']:
                file = Path(record[key+'_path'])
                if sha(file) != record[key+'_sha256']:
                    raise ValueError('Original array changed')
                verified_bytes += file.stat().st_size
            if set(record['metrics']) != set(names):
                raise ValueError('Comparator membership changed')
            rescore(record,names)
            if bool(record['n_true_signed']) != (i in plan['power_defined_indices']):
                raise ValueError('Power definition changed')
            for m,j in pos.items():
                value=record['metrics'][m]
                power[r,j]=value['power'];fdp[r,j]=value['fdp'];tp[r,j]=value['tp'];fp[r,j]=value['fp']
            u = [f['fold'] for f in record['diagnostics_R1']['folds'] if not f['guard_success']]
            v = [f['fold'] for f in record['diagnostics_R2']['folds'] if not f['K_success']]
            soft_r1.extend({'case':i,'rep':r,'fold':fold} for fold in u)
            soft_k.extend({'case':i,'rep':r,'fold':fold} for fold in v)
            groups[r] = 'both_failure' if u and v else 'R1_failure_only' if u else 'K_failure_only' if v else 'both_success'
            ok[r] = True;total_success += 1
        ap = arrays/f'case-{i:04}.npz'
        with ap.open('xb') as stream:
            np.savez_compressed(stream,power=power,fdp=fdp,tp=tp,fp=fp,methods=names,observed=ok,failure_group=groups)
        row={'case_index':i,'case':protocol['cases'][i],'n_planned':n,'n_successful':int(ok.sum()),
             'n_failed':int((~ok).sum()),'power_defined':i in plan['power_defined_indices'],
             'array_path':str(ap),'array_sha256':sha(ap),'methods':{}}
        for m,j in pos.items():
            x=fdp[ok,j]; p=power[ok,j]
            row['methods'][m]={'fdp':float(x.mean()) if ok.all() else None,
                'fdp_success_only':float(x.mean()) if len(x) else None,
                'fdp_sample_completion_bounds_not_CI':completion_bounds(x,n),
                'fdp_mcse':float(x.std(ddof=1)/np.sqrt(n)) if ok.all() else None,
                'power':float(p.mean()) if ok.all() and row['power_defined'] else None,
                'power_mcse':float(p.std(ddof=1)/np.sqrt(n)) if ok.all() and row['power_defined'] else None,
                'mean_tp':float(tp[:,j].mean()) if ok.all() else None,
                'no_discoveries':float(np.mean(tp[:,j]+fp[:,j]==0)) if ok.all() else None,
                'FDP_gt_05_fraction':float(np.mean(x>.05)) if ok.all() else None}
        row['I']=paired_macro([power[:,pos[candidate]]-power[:,pos[reference]]]) if ok.all() and row['power_defined'] else None
        powers[i]=power;soft_groups[i]=groups;rows.append(row)
        write_json(progress_path,{'verified_cases':len(rows),'planned_cases':84,
                   'updated_utc':datetime.now(timezone.utc).isoformat(),'role':'DESCRIPTIVE_FAILED_C2_AUDIT'})
        print('Verified failed-C2 audit case',i,'successful',int(ok.sum()),'failed',int((~ok).sum()),flush=True)
    strata={}; comparisons={}; component={}
    for label, indices in plan['strata'].items():
        if any(rows[i]['n_failed'] for i in indices):
            strata[label]={'available':False,'reason':'Prespecified stratum includes missing output'}
            continue
        differences=[powers[i][:,pos[candidate]]-powers[i][:,pos[reference]] for i in indices]
        group_contrib={}
        for group in ['both_success','R1_failure_only','K_failure_only','both_failure']:
            weight=np.mean([np.mean(soft_groups[i]==group) for i in indices])
            contribution=np.mean([np.sum(d*(soft_groups[i]==group))/len(d) for i,d in zip(indices,differences)])
            group_contrib[group]={'weighted_fraction':float(weight),'contribution_to_I':float(contribution)}
        component[label]=group_contrib
        strata[label]={'available':True,'I':paired_macro(differences),
            'methods':{m:{'power':float(np.mean([rows[i]['methods'][m]['power'] for i in indices])),
                          'mean_tp':float(np.mean([rows[i]['methods'][m]['mean_tp'] for i in indices]))} for m in names}}
        for family,members in plan['families'].items():
            fixed=[plan['development_selected_baselines'][family][str(i)] for i in indices]
            difference=[powers[i][:,pos[candidate]]-powers[i][:,pos[b]] for i,b in zip(indices,fixed)]
            comparisons[label+'/'+family]={'sample_envelope_difference':float(np.mean([
                rows[i]['methods'][candidate]['power']-max(rows[i]['methods'][b]['power'] for b in members) for i in indices])),
                'DEV_fixed_comparator':paired_macro(difference),'members':members,
                'role':'DESCRIPTIVE_ONLY_NO_CONFIRMATION_INTERVAL'}
    if total_success != progress['completed'] or len(failures) != progress['failed']:
        raise ValueError('All attempts must reconcile with original progress')
    result={'run_id':'R0076','audit_complete':True,'all_fixed_attempts_retained':True,
        'independent_confirmation_complete':False,'role':'DESCRIPTIVE_ONLY_FAILED_FROZEN_C2',
        'successful_families':total_success,'failed_families':len(failures),'planned_families':82800,
        'plan':plan,'rows':rows,'strata':strata,'comparisons':comparisons,'component_partition_descriptive':component,
        'failures':failures,'R1_soft_failed_folds':soft_r1,'K_soft_failed_folds':soft_k,
        'original_index_sha256':sha(index_path),'protocol_sha256':sha(protocol_path),'script_sha256':sha(__file__),
        'input_evidence_bytes_verified':verified_bytes,'recorded_cpu_seconds':cpu,
        'summed_family_wall_seconds':wall,'original_generator_wall_seconds':index['elapsed_seconds'],
        'audit_elapsed_seconds':time.perf_counter()-started,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'provenance':'SIMULATION_NOT_PATIENT_DATA','new_formal_intervals_computed':False,
        'limits':['No removal/replacement or score imputation for failed families',
                  'Completion bounds are algebraic sample ranges, not confidence intervals or defined failure outputs',
                  'C2 stop rule restricts incomplete successful batches to descriptive evidence',
                  'No C2 promotion, no third confirmation; any numerical repair is post-C2 development']}
    write_json(out/'summary.json',result)
    print(out/'summary.json',flush=True)


if __name__ == '__main__':
    main()
