"""Bind a new guard record to an explicitly registered replacement process.

The original guard implementation and its AC-only loop stay frozen. Each CLI
process operates on one new job directory; no existing registration is rebound.
"""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, read_json, now, record_event, verify_hashes
from tools.train_next_v2.execution_guard import guard as engine
from tools.train_next_v2.research_inference.benchmark import process_alive
from tools.train_next_v2.run import Tee

LEGACY_JOB = 'f-validation-ac-guard-66428-v1'
TARGET = 'f-validation-v2-mixed-b01-cpu-v1'
TESTS = BASE/'analyses/f-bound-execution-guard-v1-tests/report.json'


def directory_for(job_id):
    if job_id == LEGACY_JOB or not re.fullmatch(r'f-validation-ac-guard-[a-zA-Z0-9_-]{1,56}', job_id):
        raise ValueError('A new scoped guard job ID is required')
    return BASE/'execution_guards'/job_id


def validate_binding(expected, current, alive, lease):
    if not isinstance(expected, dict) or set(expected) != {'pid', 'creation_filetime'}:
        raise ValueError('A complete PID creation identity is required')
    if not isinstance(expected['pid'], int) or expected['pid'] <= 0 or not str(expected['creation_filetime']).isdigit():
        raise ValueError('Invalid Windows process identity')
    if current.get('process_identity') != expected or alive is not True or lease is not True:
        raise ValueError('Exact bound target identity and live OS lease required')


def bind_engine(job_id):
    # This interpreter handles one job only. The frozen loop reads these two
    # registration fields; its target, policy, APIs and recovery checks are reused.
    directory = directory_for(job_id)
    engine.JOB, engine.DIRECTORY = job_id, directory
    return directory


def prepare(job_id, directory):
    if (directory/'launch.json').exists():
        raise ValueError('Preserve existing guard registration')
    tests = read_json(TESTS)
    if tests['status'] != 'passed':
        raise ValueError('Focused guard registration tests required')
    verify_hashes(tests['code'])
    target_dir = BASE/'comparison_jobs'/TARGET
    target_launch = read_json(target_dir/'launch.json')
    target_process = read_json(target_dir/'process.json')
    identity = target_process['process_identity']
    validate_binding(identity, target_process, process_alive(identity), is_running(TARGET))
    for active in read_json(PLAN)['active_jobs']:
        if active.get('kind') == 'temporary_ac_execution_guard' and is_running(active['run_id']):
            raise ValueError('Another execution guard is active')
    verify_hashes(target_launch['pinned_hashes'])
    pinned = dict(tests['code'])
    for path in (TESTS, target_dir/'launch.json', target_dir/'process.json'):
        pinned[path.relative_to(ROOT).as_posix()] = sha256(path)
    argv = [target_launch['python'], '-u', '-m', __package__+'.job', '--job-id', job_id,
            '--launch', str(directory/'launch.json')]
    launch = dict(job_id=job_id, registered_at=now(), python=target_launch['python'], cwd=str(ROOT),
                  environment=target_launch['environment'], original_argv=argv, resume_argv=argv+['--resume'],
                  pinned_hashes=pinned, target_job=TARGET, target_process_identity=identity,
                  target_launch=(target_dir/'launch.json').relative_to(ROOT).as_posix(),
                  os_lock=str(lock_path(job_id)), log=(directory/'console.log').relative_to(ROOT).as_posix(),
                  poll_seconds=15, status_module=__package__+'.job --job-id '+job_id+' --status',
                  policy=read_json(BASE/'execution_guards'/LEGACY_JOB/'launch.json')['policy'],
                  recovery='Resume this guard only for the exact registered target PID creation identity. '
                           'A replacement target requires a new job ID and registration; preserve all earlier records.')
    atomic_json(directory/'launch.json', launch)
    stages = read_json(PLAN)['stages']
    stages['F']['execution_guard'] = dict(job_id=job_id, launch=(directory/'launch.json').relative_to(ROOT).as_posix(),
        target_job=TARGET, target_process_identity=identity, temporary=True, status='registered')
    record_event(dict(event='F_new_bound_execution_guard_registered', job_id=job_id, target=TARGET, identity=identity),
        run=dict(run_id=job_id, stage='F', kind='temporary_ac_execution_guard', status='registered',
                 launch=(directory/'launch.json').relative_to(ROOT).as_posix()), updates=dict(stages=stages))
    print(json.dumps(dict(job_id=job_id, launch=str(directory/'launch.json'), target_process_identity=identity)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true')
    modes.add_argument('--status', action='store_true')
    modes.add_argument('--launch', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    directory = bind_engine(args.job_id)
    if args.status:
        print(json.dumps(engine.status())); return
    directory.mkdir(parents=True, exist_ok=True)
    with run_lease(args.job_id):
        if args.prepare:
            prepare(args.job_id, directory); return
        if args.launch.resolve() != directory/'launch.json':
            raise ValueError('Wrong bound guard registration path')
        launch = read_json(args.launch)
        with (directory/'console.log').open('a', encoding='utf-8', buffering=1) as stream:
            with redirect_stdout(Tee(sys.stdout, stream)), redirect_stderr(Tee(sys.stderr, stream)):
                try:
                    if launch['job_id'] != args.job_id or launch['target_job'] != TARGET:
                        raise ValueError('Guard registration identity mismatch')
                    if Path.cwd().resolve() != ROOT or Path(sys.executable).resolve() != Path(launch['python']).resolve():
                        raise ValueError('Wrong workspace or isolated Python')
                    if any(os.environ.get(k) != v for k, v in launch['environment'].items()):
                        raise ValueError('Frozen guard environment changed')
                    verify_hashes(launch['pinned_hashes'])
                    current = read_json(BASE/'comparison_jobs'/TARGET/'process.json')
                    identity = launch['target_process_identity']
                    validate_binding(identity, current, process_alive(identity), is_running(TARGET))
                    engine.run(launch, args.resume)
                except Exception as exc:
                    failure = dict(at=now(), error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc())
                    atomic_json(directory/f'failure-{time.time_ns()}.json', failure)
                    record_event(dict(event='F_bound_execution_guard_failed', job_id=args.job_id, **failure),
                                 run=dict(run_id=args.job_id, stage='F', status='failed'))
                    raise


if __name__ == '__main__':
    main()
