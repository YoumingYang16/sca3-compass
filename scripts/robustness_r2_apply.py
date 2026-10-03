"""Frozen K research adapter; no truth use or claim of measured application FDR.

Works on declared arrays, not raw patients; refuses sealed holdout or active
confirmation input. Availability of this adapter is NOT candidate promotion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
MAIN='R2K_pilotc0.5_projection_gate_eBH'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def check_input_path(path):
    if 'gse320100' in str(path).casefold():raise ValueError('Protected holdout remains sealed')
    for parent in path.parents:
        if parent.name.endswith('-repetitions'):
            run=parent.name.removesuffix('-repetitions');plan=parent.parent/f'{run}-screen.protocol.json'
            if plan.exists():
                protocol=json.loads(plan.read_text(encoding='utf-8'))
                if protocol['phase'] in ['C1','C2']:
                    ip=parent.parent/f'{run}-results-index.json'
                    if not ip.exists() or not json.loads(ip.read_text(encoding='utf-8'))['complete']:
                        raise ValueError('Do not read incomplete confirmation data')
            break


def main():
    from research_window import wait_start_gate
    wait_start_gate()
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--algorithm-seed',type=int,required=True)
    parser.add_argument('--provenance',choices=['SIMULATION_NOT_PATIENT_DATA','OBSERVATIONAL_RESEARCH_UNVALIDATED'],required=True)
    parser.add_argument('--acknowledge-research-only',action='store_true',required=True)
    args=parser.parse_args();root=args.project_root.resolve();base=root/'artifacts/robustness'
    path=base/'R0076-screen.protocol.json';protocol=json.loads(path.read_text(encoding='utf-8'))
    source=base/'R0076-source';plan=protocol['analysis_plan']
    if protocol['phase']!='C2' or plan['candidate']!=MAIN or plan['bootstrap_draws']!=16:
        raise ValueError('This adapter targets only the frozen K candidate')
    for rel,digest in protocol['source_sha256'].items():
        if sha(source/rel)!=digest:raise ValueError('Frozen source checksum mismatch:'+rel)
    data_path=args.data.resolve(strict=True);check_input_path(data_path)
    if args.out.exists():raise FileExistsError('Preserve previous application output')
    if args.algorithm_seed<0:raise ValueError('Nonnegative algorithm seed required')
    sys.path.insert(0,str(source/'src'));os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    import numpy as np
    from threadpoolctl import threadpool_limits
    import sca3_compass.robustness_calibration_efficiency as implementation
    from sca3_compass.robustness_pattern_test import _json_value
    from sca3_compass.molecular_methods import ebh
    if sha(implementation.__file__)!=plan['candidate_source_sha256']:raise ValueError('Wrong imported K version')
    with np.load(data_path,allow_pickle=False) as archive:
        z=np.array(archive['z'],dtype=float);cal=np.array(archive['calibration'],dtype=float)
        ignored_truth='truth' in archive.files # deliberately not read
    permitted_n=sorted({case['n'] for case in plan['cases']})
    if z.shape!=(256,4,6) or cal.ndim!=3 or cal.shape[0]!=4 or cal.shape[2]!=6 or cal.shape[1] not in permitted_n:
        raise ValueError('Adapter restricted to frozen G256/S4/K6 and studied calibration counts; dimensions alone do not establish applicability')
    if not np.isfinite(z).all() or not np.isfinite(cal).all():raise ValueError('Nonfinite input is not silently imputed')
    with threadpool_limits(1):
        start=time.perf_counter();cpu=time.process_time()
        evidence,diagnostics=implementation.evaluate(z,cal,draws=16,seed=args.algorithm_seed,mode='K')
        rejected=ebh(evidence[MAIN],.05)
        seconds,used=time.perf_counter()-start,time.process_time()-cpu
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);ep=out/'evidence.npz'
    with ep.open('xb') as stream:
        np.savez_compressed(stream,**evidence,**{'p_'+m:v for m,v in diagnostics.pop('held_pvalues').items()},primary_rejected=rejected)
    receipt={'purpose':'FROZEN_RESEARCH_APPLICATION_NOT_NEW_CONFIRMATION','provenance':args.provenance,
        'candidate':MAIN,'method_version':protocol['method_version'],'source_run':'R0076',
        'protocol_sha256':sha(path),'candidate_sha256':sha(implementation.__file__),'adapter_sha256':sha(__file__),
        'input_path':str(data_path),'input_sha256':sha(data_path),'evidence_sha256':sha(ep),
        'algorithm_seed':args.algorithm_seed,'bootstrap_draws':16,'algorithm_nominal_fdr':.05,
        'target_shape':list(z.shape),'calibration_shape':list(cal.shape),'truth_present_but_not_loaded':ignored_truth,
        'discoveries':int(rejected.sum()),'power':None,'actual_fdr':None,'wall_seconds_all36_labels':seconds,
        'cpu_seconds_all36_labels':used,'fresh_no_cached_nuisance':True,
        'scope_warning':'No general nuisance-coverage/FDR theorem; common pipeline means, transferable compound shape, independent calibration and training families are assumptions, not verified by array dimensions.',
        'not_claimed':['candidate passed C2','clinical utility','observed FDR without ground truth','publication readiness'],
        'diagnostics':_json_value(diagnostics)}
    with (out/'receipt.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ['candidate','discoveries','power','actual_fdr','scope_warning']},indent=2))
    print(out)


if __name__=='__main__':main()
