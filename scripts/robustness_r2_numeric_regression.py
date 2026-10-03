"""Bounded post-C2 repair regression on TEN fixed existing inputs. No scoring."""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import argparse
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    p=Path(path)
    return json.loads(gzip.decompress(p.read_bytes())) if p.suffix=='.gz' else json.loads(p.read_text(encoding='utf-8'))


@contextmanager
def repaired_aliases(modules,replacement):
    """Patch EVERY local-import alias in this isolated process, restore on error."""
    originals=[m.fit_calibration for m in modules]
    try:
        for module in modules:module.fit_calibration=replacement
        if not all(m.fit_calibration is replacement for m in modules):raise ValueError('Incomplete alias coverage')
        yield
    finally:
        for module,original in zip(modules,originals):module.fit_calibration=original


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    begun=time.perf_counter();base=ROOT/'artifacts/robustness'
    protocol=read(base/'R0076-screen.protocol.json');archive=base/'R0076-source'
    immutable={archive/rel:digest for rel,digest in protocol['source_sha256'].items()}
    immutable.update({base/name:sha(base/name) for name in ['R0076-screen.protocol.json','R0076-results-index.json','R0076-progress.json']})
    for path,digest in immutable.items():
        if sha(path)!=digest:raise ValueError('Frozen source mismatch')
    out=args.out.resolve()
    if not out.is_relative_to(base.resolve()) or out.exists():raise ValueError('NEW project-local regression output required')
    out.mkdir(exist_ok=False)
    source=ROOT/'src/sca3_compass/robustness_calibration_numeric_repair.py'
    shutil.copy2(source,out/'repair_source.py');shutil.copy2(__file__,out/'regression_source.py')
    sys.path[:0]=[str(archive/'src'),str(archive/'scripts')]
    from sca3_compass import robustness_methods as methods, robustness_bootstrap_guard as guard
    from sca3_compass import robustness_calibration_efficiency as efficiency
    candidate=efficiency.evaluate
    from sca3_compass.robustness_io import write_json
    if not Path(methods.__file__).resolve().is_relative_to(archive.resolve()):
        raise ValueError('Base implementation must be frozen')
    spec=importlib.util.spec_from_file_location('sca3_compass.robustness_calibration_numeric_repair',source)
    repair=importlib.util.module_from_spec(spec);sys.modules[spec.name]=repair;spec.loader.exec_module(repair)
    rows=[]
    fixed=[(80,465),(81,500)]+[(i,0) for i in [0,27,59,63,75,77,78,83]]
    old_method,old_guard=methods.fit_calibration,guard.fit_calibration
    with threadpool_limits(1):
        for case,rep in fixed:
            if cooperative_stop():raise InterruptedError('Preserve partial regression')
            record_path=base/'R0076-repetitions'/f'case-{case:04}'/f'rep-{rep:06}.json.gz'
            record=read(record_path);immutable[record_path]=sha(record_path)
            ip=record_path.parent/f'rep-{rep:06}-input.npz';immutable[ip]=sha(ip)
            if record['status']=='completed' and sha(ip)!=record['input_sha256']:raise ValueError('Input hash')
            with np.load(ip,allow_pickle=False) as data:
                z,cal=data['z'],data['calibration']  # DO NOT read truth / compute power or FDP
            frozen_fit=old_method(cal.reshape(-1,cal.shape[-1]))
            repaired_fit,fit_diagnostic=repair.fit_calibration_with_diagnostics(cal.reshape(-1,cal.shape[-1]))
            if not repaired_fit.converged:raise ArithmeticError('Repair did not produce an accepted original fit')
            seed=int(np.random.SeedSequence([protocol['seed'],case,rep,913]).generate_state(1)[0])
            passes=[];first_arrays=None;all_cal=[]
            def instrument(x):
                fit,diag=repair.fit_calibration_with_diagnostics(x)
                all_cal.append({'converged':fit.converged,'analytic':diag['gaussian_analytic_interior'],
                    'discarded_failed_start':any(not t['success'] for t in diag['student_starts']) and diag['student_any_success'],
                    'student_any_success':diag['student_any_success']})
                return fit
            with repaired_aliases([methods,guard,efficiency],instrument):
                for iteration in range(2):
                    start=time.perf_counter();cpu=time.process_time()
                    r1,d1=guard.evaluate(z,cal,draws=16,seed=seed)
                    k,dk=candidate(z,cal,draws=16,seed=seed,mode='K')  # fresh, no bank reuse
                    values={**r1,**k,**{'p_R1_'+m:v for m,v in d1['held_pvalues'].items()},
                        **{'p_R2_'+m:v for m,v in dk['held_pvalues'].items()}}
                    if len(values)!=132 or any(not np.isfinite(v).all() or np.any(v<0) for v in values.values()):
                        raise ValueError('Nonfinite/negative/missing evidence')
                    if any(np.any(v>1) for m,v in values.items() if m.startswith('p_')):raise ValueError('Invalid p value')
                    if first_arrays is None:
                        first_arrays={m:v.copy() for m,v in values.items()}
                        ap=out/f'case-{case:04}-rep-{rep:06}-evidence.npz'
                        with ap.open('xb') as stream:np.savez_compressed(stream,**values)
                    elif any(not np.array_equal(first_arrays[m],v) for m,v in values.items()):
                        raise ValueError('Repair regression not exactly reproducible')
                    passes.append({'elapsed_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                        'arrays':len(values),'R1_soft_failed_folds':[f['fold'] for f in d1['folds'] if not f['guard_success']],
                        'K_soft_failed_folds':[f['fold'] for f in dk['folds'] if not f['K_success']]})
            original_comparison=None
            if record['status']=='completed':
                ep=Path(record['evidence_path']);immutable[ep]=record['evidence_sha256']
                if sha(ep)!=immutable[ep]:raise ValueError('Original evidence changed')
                with np.load(ep,allow_pickle=False) as previous:
                    exact=sum(np.array_equal(previous[m],v) for m,v in first_arrays.items())
                    max_abs=max(float(np.max(np.abs(previous[m]-v))) for m,v in first_arrays.items())
                    max_norm=max(float(np.max(np.abs(previous[m]-v)/np.maximum(1,np.abs(previous[m])))) for m,v in first_arrays.items())
                original_comparison={'exact_arrays':exact,'total_compared':len(first_arrays),
                    'max_absolute_e_or_p_change':max_abs,'max_change_div_max_1_abs_old':max_norm,
                    'notice':'Numerical changes disclosed; no statistical scoring / superiority inference'}
            row={'case':case,'rep':rep,'original_status':record['status'],'input_path':str(ip),'input_sha256':sha(ip),
                'original_record_sha256':sha(record_path),'original_fit':vars(frozen_fit),
                'repaired_fit':vars(repaired_fit),'fit_diagnostic':fit_diagnostic,'passes':passes,
                'fresh_repeat_exact':True,'evidence_path':str(ap),'evidence_sha256':sha(ap),
                'original_comparison':original_comparison,'calibration_calls':len(all_cal),
                'calibration_failed_calls':sum(not a['converged'] for a in all_cal),
                'analytic_calls':sum(a['analytic'] for a in all_cal),
                'successful_start_preference_calls':sum(a['discarded_failed_start'] for a in all_cal)}
            rows.append(row);write_json(out/'progress.json',{'completed_inputs':len(rows),'planned_inputs':10,'rows':rows})
            print('Regression input',case,rep,'complete, exactly reproducible',flush=True)
    for path,digest in immutable.items():
        if sha(path)!=digest:raise ValueError('Original immutable file altered')
    result={'kind':'POST_C2_NUMERICAL_REPAIR_DEVELOPMENT_NOT_CONFIRMATION','complete':True,
        'new_samples':0,'truth_read':False,'power_fdr_computed':False,'independent_confirmation':False,
        'fixed_inputs':fixed,'rows':rows,'frozen_files_unchanged':True,
        'patched_aliases':['robustness_methods','robustness_bootstrap_guard','robustness_calibration_efficiency'],
        'coverage_correction':'v2 includes K calibration_rho_bank local import; first regression retained as partial alias coverage',
        'script_sha256':sha(__file__),'repair_sha256':sha(source),
        'elapsed_seconds':time.perf_counter()-begun,'finished_utc':datetime.now(timezone.utc).isoformat(),
        'limitations':['Ten existing inputs only, including observed failures; not random reliability coverage',
            'Does not replace original failed records or promote C2',
            'No globally certified Student optimum, parameter confidence set or FDR theorem',
            'All-Student-start failure and out-of-box Gaussian numerical failure remain explicit']}
    write_json(out/'summary.json',result)
    print(out/'summary.json',flush=True)


if __name__=='__main__':main()
