"""Bounded completion follower for the ONE frozen C2; no new research."""
from research_window import wait_start_gate,cooperative_stop
wait_start_gate()
from pathlib import Path
from datetime import datetime,timezone
import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from robustness_r1_postprocess import read,sha,save,fixed_completion
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--producer-job',type=Path,required=True);p.add_argument('--wait-seconds',type=float,default=21600)
    args=p.parse_args()
    if not os.environ.get('RESEARCH_STOP_FILE') or not math.isfinite(args.wait_seconds) or not 0<args.wait_seconds<=21600:
        raise ValueError('Finite owned C2 supervisor with wait<=21600s required')
    base=ROOT/'artifacts/robustness';protocol_path=base/'R0076-screen.protocol.json';protocol=read(protocol_path)
    if protocol['phase']!='C2' or sum(protocol['repetition_counts'])!=82800:
        raise ValueError('Only frozen R0076 C2')
    out=args.out.resolve()
    if not out.is_relative_to(base/'R1R2-20260916'):raise ValueError('Output outside task archive')
    for suffix in ['confirmation-analysis.json','analysis-arrays','replay.json','evidence-report']:
        if (base/f'R0076-{suffix}').exists():raise FileExistsError('Preserve existing C2 output:'+suffix)
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'postprocess_source.py')
    source=Path(protocol['source_snapshot'])
    commands=[('archived_analysis',[str(source/'scripts/robustness_r2_confirm_analyze.py'),'--project-root',str(ROOT)],3600),
              ('archived_fresh_replay',[str(source/'scripts/robustness_r2_confirm_replay.py'),'--project-root',str(ROOT)],600),
              ('tables_figures',['scripts/robustness_r2_evidence_report.py'],180)]
    helpers=[(ROOT/c[1][0]).resolve() for c in commands]+[ROOT/'scripts/robustness_r1_postprocess.py']
    state={'run_id':'R0076','purpose':'FIXED_C2_COMPLETION_ONLY_NO_NEW_EXPERIMENTS',
        'status':'WAITING','started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(protocol_path),
        'helper_sha256':{str(p):sha(p) for p in helpers},'steps':[]}
    path=out/'state.json';save(path,state);start=time.monotonic();last=-math.inf
    try:
        while True:
            if cooperative_stop():raise InterruptedError('Owned supervisor stop')
            if time.monotonic()-start>args.wait_seconds:raise TimeoutError('Fixed completion wait exhausted')
            progress=read(base/'R0076-progress.json');index_path=base/'R0076-results-index.json'
            if progress['completed']==82800 and progress['failed']==0 and index_path.exists():
                if not fixed_completion(progress,read(index_path),protocol):raise ValueError('Completion identity mismatch')
                break
            producer=read(args.producer_job)
            if producer['status']!='RUNNING':raise RuntimeError('Producer ended without complete sample:'+producer['status'])
            if time.monotonic()-last>=60:
                state.update(last_progress=progress,checked_utc=datetime.now(timezone.utc).isoformat())
                save(path,state);print('Waiting C2',progress['completed'],'/',82800,'failures',progress['failed'],flush=True);last=time.monotonic()
            time.sleep(5)
        if sha(protocol_path)!=state['protocol_sha256']:raise ValueError('Frozen protocol changed')
        for name,command,timeout in commands:
            for helper,digest in state['helper_sha256'].items():
                if sha(helper)!=digest:raise ValueError('Pinned helper changed:'+helper)
            if cooperative_stop():raise InterruptedError('Stop before postprocess step')
            step={'name':name,'command':[sys.executable,*command],'timeout_seconds':timeout,
                  'started_utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING'}
            state.update(status='RUNNING_'+name);state['steps'].append(step);save(path,state)
            begun=time.monotonic();print('C2 postprocessing',name,flush=True)
            completed=subprocess.run(step['command'],cwd=ROOT,timeout=timeout,check=False)
            step.update(exit_code=completed.returncode,elapsed_seconds=time.monotonic()-begun,
                        status='COMPLETED' if completed.returncode==0 else 'FAILED')
            save(path,state)
            if completed.returncode:raise RuntimeError('Postprocess failed; preserve outputs:'+name)
        state['status']='COMPLETED_PENDING_MAIN_REVIEW_NO_AUTO_PROMOTION'
    except BaseException as error:
        state.update(status='FAILED_OR_STOPPED',error=repr(error));raise
    finally:
        state.update(finished_or_checkpoint_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-start)
        save(path,state)
    print('C2 artifacts complete; main review required. No new methods or experiments launched.',flush=True)


if __name__=='__main__':main()
