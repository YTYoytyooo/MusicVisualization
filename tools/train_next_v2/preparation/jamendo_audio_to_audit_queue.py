"""Finite dependency queue: verified255-track acquisition, then native audio audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, run_lease, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity

JOB = 'd-jamendo-audio-to-audit-v1'
DIRECTORY = BASE/'queues'/JOB
DOWNLOAD = BASE/'resources/20261004/jobs/d-jamendo-selected-audio-v1'
AUDIT = BASE/'analyses/d-jamendo-native-audit-v2'


def register():
    if DIRECTORY.exists():
        raise FileExistsError('Preserve existing finite queue')
    source = read_json(AUDIT/'launch.json')
    if not is_running('d-jamendo-selected-audio-v1') or is_running(source['job_id']):
        raise RuntimeError('Expected one active download and no active native audit')
    files = [Path(__file__), AUDIT/'launch.json', DOWNLOAD/'launch.json',
             ROOT/'tools/train_next_v2/common.py', ROOT/'tools/train_next/runtime.py', ROOT/'tools/train_next/core.py']
    argv = [source['python'], '-u', '-m', 'tools.train_next_v2.preparation.jamendo_audio_to_audit_queue', '--run']
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, cwd=str(ROOT), python=source['python'],
        environment=source['environment'], os_lock=str(lock_path(JOB)),
        protected_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in files},
        dependency_job='d-jamendo-selected-audio-v1', dependency_report=str((DOWNLOAD/'report.json').relative_to(ROOT)),
        audit_job=source['job_id'], audit_argv=source['original_argv'],
        log=str(DIRECTORY/'events.jsonl'), child_log=str(DIRECTORY/'audit-console.log'),
        maximum_dependency_wait_seconds=3600, poll_interval_seconds=10, gpu_used=False,
        recovery='One finite queue; no automatic retry after dependency or audit failure. Preserve launch/failure/child logs. Diagnose inactive child and use its registered --resume directly if justified.')
    DIRECTORY.mkdir(parents=True)
    atomic_json(DIRECTORY/'launch.json', launch)
    with run_lease('next-va-v2-d-ledger'):
        stages = read_json(PLAN)['stages']
        stages['D']['jamendo_pipeline_queue'] = dict(job_id=JOB, status='registered', launch=str((DIRECTORY/'launch.json').relative_to(ROOT)))
        record_event(dict(event='D_Jamendo_audio_to_audit_queue_registered', launch=str((DIRECTORY/'launch.json').relative_to(ROOT))), updates=dict(stages=stages))
    print(json.dumps(dict(launch=str(DIRECTORY/'launch.json'))))


def status():
    process = read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity = process.get('process_identity')
    state = read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else {}
    report = read_json(DIRECTORY/'report.json') if (DIRECTORY/'report.json').exists() else {}
    print(json.dumps(dict(job_id=JOB, checked_at=now(), os_lease_held=is_running(JOB),
        process_identity=identity, process_identity_matches=bool(identity and process_identity(identity['pid'])==identity),
        download_lease_active=is_running('d-jamendo-selected-audio-v1'), audit_lease_active=is_running('d-jamendo-native-audit-v2'),
        progress=state, report_status=report.get('status'))))


def run():
    with run_lease(JOB):
        launch = read_json(DIRECTORY/'launch.json')
        verify_hashes(launch['protected_hashes'])
        if Path(sys.executable).resolve()!=Path(launch['python']) or Path.cwd().resolve()!=ROOT:
            raise ValueError('Wrong Python or cwd')
        if (DIRECTORY/'process.json').exists():
            raise FileExistsError('Previous queue attempt exists; diagnose instead of repeating')
        identity = process_identity()
        atomic_json(DIRECTORY/'process.json', dict(started_at=now(), process_identity=identity))
        def publish(**fields):
            row = dict(updated_at=now(), process_identity=identity, **fields)
            atomic_json(DIRECTORY/'status.json', row)
            with (DIRECTORY/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
        try:
            started = time.monotonic()
            last_verified = None
            while is_running(launch['dependency_job']):
                state = read_json(DOWNLOAD/'status.json')
                if state.get('verified_tracks') != last_verified:
                    last_verified = state.get('verified_tracks')
                    publish(stage='waiting_for_verified_acquisition', verified_tracks=last_verified, total_tracks=255,
                            dependency_updated_at=state.get('updated_at'))
                if time.monotonic()-started > launch['maximum_dependency_wait_seconds']:
                    raise TimeoutError('Finite dependency wait expired; existing download remains untouched')
                time.sleep(launch['poll_interval_seconds'])
            if not (DOWNLOAD/'report.json').exists():
                raise RuntimeError('Acquisition exited without completed report; inspect its failure before recovery')
            completed = read_json(DOWNLOAD/'report.json')
            if completed['status']!='completed_selected_audio_verified' or completed['selected_tracks']!=255:
                raise ValueError('Acquisition completion contract failed')
            verify_hashes(launch['protected_hashes'])
            if is_running(launch['audit_job']) or (AUDIT/'process.json').exists():
                raise RuntimeError('Another audit attempt exists; refuse duplicate launch')
            environment = dict(os.environ)
            environment.update(launch['environment'])
            publish(stage='running_native_audio_audit', acquisition_sha256=sha256(DOWNLOAD/'report.json'))
            with (DIRECTORY/'audit-console.log').open('xb') as log:
                child = subprocess.Popen(launch['audit_argv'], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                atomic_json(DIRECTORY/'child-process.json', dict(pid=child.pid, process_identity=process_identity(child.pid), argv=launch['audit_argv']))
                code = child.wait()
            if code or not (AUDIT/'report.json').exists():
                raise RuntimeError(f'Native audit failed with exitcode={code}; preserve child log and do not retry')
            report = read_json(AUDIT/'report.json')
            if report['status']!='completed_native_audio_gate':
                raise ValueError('Native audit lacks completed report')
            result = dict(status='completed_acquisition_and_audio_audit', completed_at=now(),
                acquisition_report=str((DOWNLOAD/'report.json').relative_to(ROOT)), acquisition_sha256=sha256(DOWNLOAD/'report.json'),
                audit_report=str((AUDIT/'report.json').relative_to(ROOT)), audit_sha256=sha256(AUDIT/'report.json'),
                native_passed=report['native_passed'], eligible_categories=report['eligible_categories'],
                training_admitted=False, next_action='Review exclusions and freeze a matched category-only auxiliary training protocol.')
            atomic_json(DIRECTORY/'report.json', result)
            with run_lease('next-va-v2-d-ledger'):
                stages = read_json(PLAN)['stages']
                stages['D']['jamendo_pipeline_queue'].update(status=result['status'],report=str((DIRECTORY/'report.json').relative_to(ROOT)))
                record_event(dict(event='D_Jamendo_audio_to_audit_queue_completed', eligible_categories=report['eligible_categories']), updates=dict(stages=stages))
            publish(stage=result['status'], eligible_categories=report['eligible_categories'])
        except BaseException as exc:
            failure = dict(failed_at=now(), error=f'{type(exc).__name__}: {exc}')
            atomic_json(DIRECTORY/'failure.json', failure)
            publish(stage='failed', error=failure['error'])
            with run_lease('next-va-v2-d-ledger'):
                stages = read_json(PLAN)['stages']
                stages['D']['jamendo_pipeline_queue'].update(status='failed', error=failure['error'])
                record_event(dict(event='D_Jamendo_audio_to_audit_queue_failed', **failure), updates=dict(stages=stages))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--register',action='store_true')
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--status',action='store_true')
    args=parser.parse_args()
    register() if args.register else status() if args.status else run()
