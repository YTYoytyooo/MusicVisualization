"""Repair only the last3 archive caches, then let original --resume finalize."""
import argparse
import json
from pathlib import Path
import sys
import threading

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity
from tools.train_next_v2.jamendo_audio.ranges import Budget, acquire_archive
from .transport import BoundedTLSSession

JOB = 'd-jamendo-final-cache-recovery-v1'
DIRECTORY = BASE/'resources/20261004/jobs'/JOB
SOURCE = BASE/'resources/20261004/jobs/d-jamendo-selected-audio-v1'
RAW = ROOT/'data/datasets/mtg_jamendo/raw/selected-audio-v1'
SELECTED = ROOT/'data/datasets/mtg_jamendo/prepared/candidates-v1/selected.json'


def register():
    if DIRECTORY.exists() or is_running('d-jamendo-selected-audio-v1'):
        raise RuntimeError('Preserve registered task; source must be inactive')
    source = read_json(SOURCE/'launch.json')
    verify_hashes(source['protected_hashes'])
    rows = read_json(SELECTED)
    missing = []
    for row in rows:
        path = RAW/'audio'/row['low_quality_path']
        if path.exists():
            if sha256(path) != row['expected_audio_sha256']:
                raise ValueError('Existing audio changed')
        else:
            missing.append(row['track_id'])
    if len(missing) != 8:
        raise ValueError('Diagnosed recovery scope changed')
    folders = sorted({r['low_quality_path'].split('/')[0] for r in rows if r['track_id'] in missing})
    if folders != ['95', '98', '99']:
        raise ValueError('Unexpected recovery archives')
    files = list(Path(__file__).parent.glob('*.py')) + [SOURCE/'launch.json', SOURCE/'failure.json']
    protected = dict(source['protected_hashes'])
    protected.update({p.relative_to(ROOT).as_posix():sha256(p) for p in files})
    argv = [source['python'], '-u', '-m', 'tools.train_next_v2.jamendo_recovery.run', '--run']
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, cwd=str(ROOT), python=source['python'],
        environment=source['environment'], protected_hashes=protected,
        os_lock=str(lock_path(JOB)), source_cache_lock=str(lock_path(source['job_id'])),
        log=str(DIRECTORY/'events.jsonl'), missing_track_ids=missing, folders=folders, workers=1,
        maximum_connection_attempts=3, retry_delay_seconds=[2,5],
        network_budget_path=str((SOURCE/'network-budget.json').relative_to(ROOT)),
        maximum_bytes=source['maximum_bytes'],maximum_requests=source['maximum_requests'],
        scope='Reuse frozen ArchiveRanges/acquire_archive for3 interrupted caches and8 missing tracks only. Same TLS verification, ETag, Range, path, per-trackSHA and overall2GiB request reservation checks. Independent transport retries only the diagnosed connectionEOF; no monkeypatch or changes to original code/config.',
        next_action='After verifying all255 expected hashes and completing all92 archive receipts, execute original registered recovery argv once to revalidate cached artifacts and finalize without new network transfers. Then run native auditv2 directly; do not rerun failed dependency queue.',
        recovery='No automatic command restart. Preserve failure evidence and diagnose any new error.')
    DIRECTORY.mkdir(parents=True)
    atomic_json(DIRECTORY/'launch.json', launch)
    with run_lease('next-va-v2-d-ledger'):
        stages = read_json(PLAN)['stages']
        stages['D']['jamendo_cache_recovery'] = dict(job_id=JOB,status='registered',launch=str((DIRECTORY/'launch.json').relative_to(ROOT)))
        record_event(dict(event='D_Jamendo_final_cache_recovery_registered',missing=8,folders=folders),updates=dict(stages=stages))
    print(json.dumps(dict(launch=str(DIRECTORY/'launch.json'),missing=8,folders=folders)))


def run():
    # Source lock prevents a simultaneous original resume from writing these caches.
    with run_lease(JOB), run_lease('d-jamendo-selected-audio-v1'):
        launch = read_json(DIRECTORY/'launch.json')
        verify_hashes(launch['protected_hashes'])
        if Path(sys.executable).resolve()!=Path(launch['python']) or Path.cwd().resolve()!=ROOT:
            raise ValueError('Wrong Python or cwd')
        if (DIRECTORY/'process.json').exists():
            raise FileExistsError('Previous recovery exists; do not repeat')
        source = read_json(SOURCE/'launch.json')
        identity=process_identity()
        atomic_json(DIRECTORY/'process.json',dict(started_at=now(),process_identity=identity))
        budget=Budget(SOURCE/'network-budget.json',launch['maximum_bytes'],launch['maximum_requests'])
        before=dict(budget.state)
        def log(row):
            state=dict(updated_at=now(),process_identity=identity,**row)
            atomic_json(DIRECTORY/'status.json',state)
            with (DIRECTORY/'events.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(state)+'\n')
            if row.get('last_track') or row.get('completed') or row.get('event'):
                print(json.dumps(state),flush=True)
        rows={r['track_id']:r for r in read_json(SELECTED)}
        try:
            for item in source['archives']:
                if item['folder'] not in launch['folders']:
                    continue
                session=BoundedTLSSession(budget,log)
                acquire_archive(item['url'],item['folder'],[rows[k] for k in item['selected_track_ids']],
                    RAW/'audio',RAW/'index'/item['folder'],budget,threading.Event(),log,session)
            for row in rows.values():
                if sha256(RAW/'audio'/row['low_quality_path'])!=row['expected_audio_sha256']:
                    raise ValueError('Final frozen255-track checksum failed')
            for item in source['archives']:
                receipt=read_json(RAW/'index'/item['folder']/'report.json')
                if receipt['status']!='completed_selected_members_verified' or {r['track_id'] for r in receipt['records']}!=set(item['selected_track_ids']):
                    raise ValueError('Archive cache not complete')
            verify_hashes(launch['protected_hashes'])
            report=dict(status='completed_cache_repair_not_original_job_finalization',completed_at=now(),
                verified_tracks=len(rows),complete_archive_receipts=len(source['archives']),repaired_missing_tracks=8,
                budget_before=before,budget_after=budget.state,original_config_unchanged=True,original_code_unchanged=True,
                training_admitted=False,next_action=launch['next_action'])
            atomic_json(DIRECTORY/'report.json',report)
            with run_lease('next-va-v2-d-ledger'):
                stages=read_json(PLAN)['stages'];stages['D']['jamendo_cache_recovery'].update(status=report['status'],report=str((DIRECTORY/'report.json').relative_to(ROOT)))
                record_event(dict(event='D_Jamendo_final_cache_repaired',verified_tracks=len(rows)),updates=dict(stages=stages))
            print(json.dumps(report),flush=True)
        except BaseException as exc:
            failure=dict(failed_at=now(),error=f'{type(exc).__name__}: {exc}')
            atomic_json(DIRECTORY/'failure.json',failure)
            record_event(dict(event='D_Jamendo_final_cache_recovery_failed',**failure))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args();register() if args.register else run()
