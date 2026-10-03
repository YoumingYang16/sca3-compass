"""Bounded C1 completion follower; no new experiments, candidates or R2.

Run only under research_window.py's owned finite job. Wait for the original
fixed sample to finish, then execute already specified analysis/replay/report
commands. Existing outputs are never overwritten or silently treated as valid.
"""
from research_window import wait_start_gate, cooperative_stop
wait_start_gate()
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def retry_sharing_operation(operation, attempts=10, delay=0.1):
    """Bounded retry for transient Windows sharing denial, not corrupt input.

    Checkpoint writers use atomic replacement. A competing open/replace can
    briefly deny access. Persistent denial still fails; permissions are not
    changed and malformed JSON is never retried or treated as completion.
    """
    for attempt in range(attempts):
        try:
            return operation()
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(delay)


def read(path):
    return json.loads(retry_sharing_operation(
        lambda: Path(path).read_text(encoding='utf-8')))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    tmp = path.with_suffix('.pending.json')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    retry_sharing_operation(lambda: os.replace(tmp, path))


def fixed_completion(progress, index, protocol):
    count = sum(protocol['repetition_counts'])
    return (progress['run_id'] == protocol['run_id']
            and index['run_id'] == protocol['run_id']
            and progress['completed'] == count and progress['failed'] == 0
            and progress['planned'] == count and index['complete'] is True
            and index['settings'] == protocol and len(index['receipts']) == count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--wait-seconds', type=float, default=9000)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--producer-job', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('RESEARCH_STOP_FILE'):
        raise ValueError('Use the finite owned supervisor; no unbounded standalone waiter')
    if not math.isfinite(args.wait_seconds) or not 0 < args.wait_seconds <= 14400:
        raise ValueError('Finite wait <=4 hours required')
    folder = ROOT / 'artifacts/robustness'
    protocol_path = folder / f'{args.run_id}-screen.protocol.json'
    protocol = read(protocol_path)
    if protocol['phase'] != 'C1' or args.run_id != 'R0073':
        raise ValueError('This follower is scoped to the existing frozen R0073 C1 only')
    out = args.out.resolve()
    if not out.is_relative_to(folder / 'R1R2-20260916'):
        raise ValueError('Follower output must stay in this task archive')
    for suffix in ['confirmation-analysis.json', 'replay.json', 'contributions', 'figures', 'evidence-report']:
        if (folder / f'{args.run_id}-{suffix}').exists():
            raise FileExistsError('Pre-existing output needs explicit audit, not automatic overwrite: ' + suffix)
    out.mkdir(parents=True, exist_ok=False)
    shutil_source = out / 'postprocess_source.py'
    import shutil
    shutil.copy2(__file__, shutil_source)
    state = {'run_id': args.run_id, 'purpose': 'FOLLOW_FIXED_C1_ONLY_NO_R2_NO_NEW_SAMPLES',
             'protocol_sha256': sha(protocol_path), 'script_sha256': sha(__file__),
             'started_utc': datetime.now(timezone.utc).isoformat(), 'wait_limit_seconds': args.wait_seconds,
             'status': 'WAITING_FOR_COMPLETE_FIXED_SAMPLE', 'steps': []}
    helper_paths = [folder / f'{args.run_id}-source/scripts/robustness_r1_confirm.py'] + [
        ROOT / 'scripts' / name for name in ['robustness_r1_replay.py', 'robustness_r1_contributions.py',
                                             'robustness_r1_figures.py', 'robustness_r1_evidence_report.py']]
    state['helper_sha256_at_start'] = {str(p): sha(p) for p in helper_paths}
    state_path = out / 'state.json'
    save(state_path, state)
    start = time.monotonic()
    try:
        last_print = -math.inf
        while True:
            if cooperative_stop():
                raise InterruptedError('Supervisor requested a checkpoint stop')
            if time.monotonic() - start > args.wait_seconds:
                raise TimeoutError('Finite completion wait exhausted; no new sample plan')
            progress = read(folder / f'{args.run_id}-progress.json')
            index_path = folder / f'{args.run_id}-results-index.json'
            if progress['completed'] == sum(protocol['repetition_counts']) and progress['failed'] == 0 and index_path.exists():
                index = read(index_path)
                if not fixed_completion(progress, index, protocol):
                    raise ValueError('Completion/protocol/index mismatch')
                break
            producer = read(args.producer_job)
            if producer['status'] != 'RUNNING':
                raise RuntimeError('Producer ended without complete fixed sample: ' + producer['status'])
            if time.monotonic() - last_print >= 60:
                state['last_progress'] = progress
                state['checked_utc'] = datetime.now(timezone.utc).isoformat()
                save(state_path, state)
                print('Waiting for original complete C1:', progress['completed'], '/', progress['planned'],
                      'execution failures', progress['failed'], flush=True)
                last_print = time.monotonic()
            time.sleep(5)
        if sha(protocol_path) != state['protocol_sha256']:
            raise ValueError('Protocol changed during wait')
        commands = [
            ('archived_analysis', [str(folder / f'{args.run_id}-source/scripts/robustness_r1_confirm.py'),
                                   '--run-id', args.run_id, '--project-root', str(ROOT)], 3600),
            ('exact_replay', ['scripts/robustness_r1_replay.py', '--run-id', args.run_id,
                              '--cases', '0,27,59,63,75'], 180),
            ('contributions', ['scripts/robustness_r1_contributions.py', '--run-id', args.run_id], 180),
            ('figures', ['scripts/robustness_r1_figures.py', '--run-id', args.run_id], 180),
            ('evidence_report', ['scripts/robustness_r1_evidence_report.py', '--run-id', args.run_id], 180),
        ]
        for name, command, timeout in commands:
            if cooperative_stop():
                raise InterruptedError('Supervisor stop before next postprocessing step')
            source_path = (ROOT / command[0]).resolve()
            if sha(source_path) != state['helper_sha256_at_start'][str(source_path)]:
                raise ValueError('Postprocessing helper changed after launch: ' + name)
            state['status'] = 'RUNNING_' + name
            step = {'name': name, 'command': [sys.executable, *command],
                    'started_utc': datetime.now(timezone.utc).isoformat(), 'timeout_seconds': timeout,
                    'status': 'RUNNING'}
            state['steps'].append(step)
            save(state_path, state)
            print('Postprocessing:', name, flush=True)
            begun = time.monotonic()
            result = subprocess.run(step['command'], cwd=ROOT, timeout=timeout, check=False)
            step.update(exit_code=result.returncode, elapsed_seconds=time.monotonic()-begun,
                        status='COMPLETED' if result.returncode == 0 else 'FAILED')
            save(state_path, state)
            if result.returncode != 0:
                raise RuntimeError('Postprocessing failed; preserve all outputs: ' + name)
        state['status'] = 'COMPLETED_EVIDENCE_REQUIRES_MAIN_REVIEW'
    except BaseException as error:
        state.update(status='STOPPED_OR_FAILED', error=repr(error))
        raise
    finally:
        state['finished_or_checkpoint_utc'] = datetime.now(timezone.utc).isoformat()
        state['elapsed_seconds'] = time.monotonic()-start
        save(state_path, state)
    print('C1 artifacts generated; no R2 launch or scientific success declared.', flush=True)


if __name__ == '__main__':
    main()
