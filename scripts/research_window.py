"""Small bounded-job supervisor. Windows Job Object owns only its child tree.

No scheduler, no automatic experiment restart. Closing the supervisor kills its
owned Windows job. A child waits at the start gate until assignment succeeds.
UTC and monotonic deadlines do not reset between batches/context restorations.
"""
from __future__ import annotations
import argparse
import ctypes as ct
from ctypes import wintypes as wt
from datetime import datetime,timezone,timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
WINDOW=ROOT/'artifacts/robustness/R1R2-20260916'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as s:json.dump(value,s,indent=2,ensure_ascii=False,allow_nan=False)
    os.replace(tmp,path)


def utc():return datetime.now(timezone.utc)


def initialize(t0):
    if (WINDOW/'window.json').exists():raise ValueError('Window already initialized; never reset T0')
    begun=datetime.fromisoformat(t0);now=utc();mono=time.monotonic()
    if begun.tzinfo is None or begun>now:raise ValueError('T0 must be observed past timezone-aware time')
    info={'t0_utc':begun.isoformat(),'timezone':'Asia/Hong_Kong','utc_anchor':now.isoformat(),
        'monotonic_anchor':mono,'t0_monotonic_estimate':mono-(now-begun).total_seconds(),
        'monotonic_notice':'Anchor measured at initialization; T0 inferred from initial observed wall-clock time, not directly sampled then.',
        'r1_priority_utc':(begun+timedelta(hours=4)).isoformat(),
        'no_new_mechanisms_utc':(begun+timedelta(hours=5)).isoformat(),
        'research_stop_utc':(begun+timedelta(hours=6,minutes=30)).isoformat(),
        'hard_stop_utc':(begun+timedelta(hours=7)).isoformat(),
        'confirmation_error_budget':{'C1':.025,'C2':.025,'C2_roles':['R2','repaired_R1.1_exclusive'],'third_attempt':False},
        'client_limit':'Model/session continuity is not guaranteed. Only actually launched bounded OS jobs persist; no claim of autonomous research after session ends.',
        'mechanisms_started':0,'status':'R1_INTAKE'}
    write(WINDOW/'window.json',info);return info


def remaining_seconds(info,*,packaging=False):
    mono=time.monotonic()
    if mono<info['monotonic_anchor']:raise RuntimeError('Monotonic anchor predates reboot; do not reset window')
    total=7*3600 if packaging else 6.5*3600
    wall=(datetime.fromisoformat(info['hard_stop_utc' if packaging else 'research_stop_utc'])-utc()).total_seconds()
    return min(wall,info['t0_monotonic_estimate']+total-mono)


def wait_start_gate():
    gate=os.getenv('RESEARCH_START_GATE')
    if not gate:return
    start=time.monotonic()
    while not Path(gate).exists():
        if time.monotonic()-start>15:raise RuntimeError('Supervisor did not assign safe job/start gate')
        time.sleep(.02)


def cooperative_stop():
    marker=os.getenv('RESEARCH_STOP_FILE')
    if marker and Path(marker).exists():return True
    window=os.getenv('RESEARCH_WINDOW_FILE')
    return bool(window and remaining_seconds(json.loads(Path(window).read_text(encoding='utf-8')))<=0)


class WindowsJob:
    def __init__(self):
        class Basic(ct.Structure):
            _fields_=[('PerProcessUserTimeLimit',ct.c_longlong),('PerJobUserTimeLimit',ct.c_longlong),
                ('LimitFlags',wt.DWORD),('MinimumWorkingSetSize',ct.c_size_t),('MaximumWorkingSetSize',ct.c_size_t),
                ('ActiveProcessLimit',wt.DWORD),('Affinity',ct.c_size_t),('PriorityClass',wt.DWORD),('SchedulingClass',wt.DWORD)]
        class IO(ct.Structure):
            _fields_=[(name,ct.c_ulonglong) for name in ['ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount']]
        class Extended(ct.Structure):
            _fields_=[('BasicLimitInformation',Basic),('IoInfo',IO),('ProcessMemoryLimit',ct.c_size_t),
                ('JobMemoryLimit',ct.c_size_t),('PeakProcessMemoryUsed',ct.c_size_t),('PeakJobMemoryUsed',ct.c_size_t)]
        self.lib=ct.WinDLL('kernel32',use_last_error=True)
        self.lib.CreateJobObjectW.argtypes=[ct.c_void_p,wt.LPCWSTR];self.lib.CreateJobObjectW.restype=wt.HANDLE
        self.lib.SetInformationJobObject.argtypes=[wt.HANDLE,ct.c_int,ct.c_void_p,wt.DWORD]
        self.lib.AssignProcessToJobObject.argtypes=[wt.HANDLE,wt.HANDLE]
        self.lib.TerminateJobObject.argtypes=[wt.HANDLE,wt.UINT]
        self.lib.CloseHandle.argtypes=[wt.HANDLE]
        self.lib.GetProcessTimes.argtypes=[wt.HANDLE,*([ct.POINTER(wt.FILETIME)]*4)]
        self.handle=self.lib.CreateJobObjectW(None,None)
        if not self.handle:raise ct.WinError(ct.get_last_error())
        limits=Extended();limits.BasicLimitInformation.LimitFlags=0x2000 # KILL_ON_JOB_CLOSE
        if not self.lib.SetInformationJobObject(self.handle,9,ct.byref(limits),ct.sizeof(limits)):
            self.close();raise ct.WinError(ct.get_last_error())

    def assign(self,process):
        handle=wt.HANDLE(int(process._handle))
        if not self.lib.AssignProcessToJobObject(self.handle,handle):raise ct.WinError(ct.get_last_error())
        values=[wt.FILETIME() for _ in range(4)]
        if not self.lib.GetProcessTimes(handle,*(ct.byref(x) for x in values)):raise ct.WinError(ct.get_last_error())
        return (values[0].dwHighDateTime<<32)|values[0].dwLowDateTime

    def terminate(self):
        if self.handle and not self.lib.TerminateJobObject(self.handle,124):raise ct.WinError(ct.get_last_error())

    def close(self):
        if getattr(self,'handle',None):self.lib.CloseHandle(self.handle);self.handle=None


def run(run_id,command,seconds,grace,continuation_authority=None):
    if not math.isfinite(seconds) or seconds<=0 or not math.isfinite(grace) or not 0<=grace<seconds:
        raise ValueError('Finite positive batch budget and a smaller nonnegative grace required')
    info=json.loads((WINDOW/'window.json').read_text(encoding='utf-8'))
    if not run_id.replace('-','').replace('_','').isalnum():raise ValueError('Unsafe run id')
    authority=None;job_root=WINDOW
    if continuation_authority is not None:
        authority_path=Path(continuation_authority).resolve(strict=True)
        if not authority_path.is_relative_to(WINDOW.resolve()):raise ValueError('Continuation record must stay within the research window archive')
        authority=json.loads(authority_path.read_text(encoding='utf-8'))
        actual=hashlib.sha256((WINDOW/'window.json').read_bytes()).hexdigest()
        legacy_continuation = authority.get('no_new_statistical_attempt') is True
        new_scoped_phase = (authority.get('new_user_authorized_phase') == 'V1_DELIVERY'
            and authority.get('confirmation_attempt_cap') == 1
            and authority.get('acceptance_path') == 'V1_ACCEPTANCE.md'
            and authority.get('user_soft_window_override') is True)
        if new_scoped_phase:
            expected = authority.get('acceptance_sha256')
            if hashlib.sha256((ROOT/'V1_ACCEPTANCE.md').read_bytes()).hexdigest() != expected:
                raise ValueError('New-phase acceptance changed after authority registration')
        if (authority.get('user_authorized_wallclock_extension') is not True
                or authority.get('original_t0_utc')!=info['t0_utc']
                or authority.get('original_window_sha256','').lower()!=actual
                or not (legacy_continuation or new_scoped_phase)):
            raise ValueError('Missing or inconsistent explicit continuation authority')
        job_root=authority_path.parent
    available=remaining_seconds(info) if authority is None else seconds
    if seconds<=0 or seconds>available:raise ValueError(f'Batch budget {seconds}s exceeds remaining {available:.1f}s')
    out=job_root/'jobs'/run_id;out.mkdir(parents=True,exist_ok=False)
    gate=out/'start.json';stop=out/'stop.json'
    env=os.environ.copy();env.update(RESEARCH_START_GATE=str(gate),RESEARCH_STOP_FILE=str(stop),
        RESEARCH_WINDOW_FILE=str(WINDOW/'window.json'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    if authority is not None:
        # Frozen children keep their marker-based cooperative stop. Do not
        # fabricate/reset the pre-reboot monotonic anchor or old T0.
        env.pop('RESEARCH_WINDOW_FILE',None)
    record={'run_id':run_id,'command':command,'cwd':str(ROOT),'started_utc':utc().isoformat(),
        'budget_seconds':seconds,'remaining_at_start':available,'stop_marker':str(stop),'status':'STARTING'}
    if authority is not None:
        record.update(continuation_authority=str(authority_path),
            continuation_authority_sha256=hashlib.sha256(authority_path.read_bytes()).hexdigest(),
            original_t0_utc=info['t0_utc'],deadline_policy='Explicit user extension; unchanged old window archive; finite per-job budget')
    begin=time.monotonic();job=WindowsJob() if os.name=='nt' else None;process=None
    write(out/'process.json',record)
    try:
        with (out/'stdout.log').open('xb') as stdout,(out/'stderr.log').open('xb') as stderr:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=os.name!='nt')
            record['pid']=process.pid
            if job:record['creation_filetime_100ns']=job.assign(process)
            record['status']='RUNNING';write(out/'process.json',record)
            write(gate,{'assigned_utc':utc().isoformat(),'pid':process.pid})
            requested=False
            while process.poll() is None:
                left=seconds-(time.monotonic()-begin)
                if authority is None:left=min(left,remaining_seconds(info))
                if left<=grace and not requested:
                    write(stop,{'reason':'bounded_deadline','utc':utc().isoformat()});requested=True
                if left<=0:
                    if job:job.terminate()
                    else:
                        import signal
                        os.killpg(process.pid,signal.SIGKILL)
                    record['forced_termination']=True;break
                time.sleep(min(.25,max(.01,left)))
            code=process.wait(timeout=10)
            record.update(status='TIMED_OUT' if requested else ('COMPLETED' if code==0 else 'FAILED'),
                exit_code=code,cooperative_stop_requested=requested,finished_utc=utc().isoformat(),elapsed_seconds=time.monotonic()-begin)
    except BaseException as error:
        record.update(status='SUPERVISOR_ERROR',error=repr(error))
        if process is not None and process.poll() is None:
            if job:job.terminate()
            else:process.terminate()
        raise
    finally:
        if job:job.close()
        write(out/'process.json',record)
    print(json.dumps(record));return 0 if record['status'] in ['COMPLETED','TIMED_OUT'] else 1


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    init=sub.add_parser('init');init.add_argument('--t0',required=True)
    job=sub.add_parser('run');job.add_argument('--id',required=True);job.add_argument('--seconds',type=float,required=True)
    job.add_argument('--continuation-authority',type=Path)
    job.add_argument('--grace',type=float,default=15);job.add_argument('command',nargs=argparse.REMAINDER)
    sub.add_parser('status')
    args=parser.parse_args()
    if args.action=='init':print(json.dumps(initialize(args.t0),indent=2));return
    if args.action=='status':
        info=json.loads((WINDOW/'window.json').read_text(encoding='utf-8'));print(json.dumps({**info,'remaining_research_seconds':remaining_seconds(info)},indent=2));return
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    if not command:raise ValueError('Command required')
    raise SystemExit(run(args.id,command,args.seconds,args.grace,args.continuation_authority))


if __name__=='__main__':main()
