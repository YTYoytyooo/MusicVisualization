"""Registered finite acquisition of the255 frozen Jamendo candidates only."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading
import time

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, verify_hashes, process_identity
from .ranges import Budget, acquire_archive

JOB = 'd-jamendo-selected-audio-v1'
DIRECTORY = BASE/'resources/20261004/jobs'/JOB
DEST = ROOT/'data/datasets/mtg_jamendo/raw/selected-audio-v1'
SELECTED = ROOT/'data/datasets/mtg_jamendo/prepared/candidates-v1/selected.json'
PROBE = BASE/'resources/20261004/jobs/d-jamendo-audio-probe-v1/report.json'
CHECKSUMS = ROOT/'data/datasets/mtg_jamendo/raw/source-details-v1/jamendo/data/download/autotagging_moodtheme_audio-low_sha256_tars.txt'


def register():
    if DIRECTORY.exists():
        raise FileExistsError('Preserve registered acquisition config')
    rows = read_json(SELECTED)
    if len(rows) != 255 or any(r['training_admitted'] or r['official_role'] != 'train' for r in rows):
        raise ValueError('Frozen training-only candidates changed')
    mirror = next(r for r in read_json(PROBE)['mirrors'] if r['viable'])
    prefix = mirror['url'].rsplit('/', 1)[0]
    tar_hashes = {name: digest for digest, name in (line.split() for line in CHECKSUMS.read_text().splitlines())}
    folders = sorted({r['low_quality_path'].split('/')[0] for r in rows})
    archives = []
    for folder in folders:
        name = f'autotagging_moodtheme_audio-low-{folder}.tar'
        archives.append(dict(folder=folder, url=prefix+'/'+name, published_tar_sha256=tar_hashes[name],
            selected_track_ids=[r['track_id'] for r in rows if r['low_quality_path'].startswith(folder+'/')]))
    previous = read_json(BASE/'analyses/d-jamendo-candidates-v1/launch.json')
    protected = dict(previous['protected_hashes'])
    files = [SELECTED, PROBE, CHECKSUMS] + list(Path(__file__).parent.glob('*.py'))
    protected.update({p.relative_to(ROOT).as_posix(): sha256(p) for p in files})
    python = str(Path(read_json(PLAN)['python']).resolve())
    argv = [python, '-u', '-m', 'tools.train_next_v2.jamendo_audio.download', '--run']
    launch = dict(job_id=JOB, registered_at=now(), python=python, cwd=str(ROOT), original_argv=argv,
        recovery_argv=argv+['--resume'], environment=previous['environment'], protected_hashes=protected,
        os_lock=str(lock_path(JOB)), log=str(DIRECTORY/'events.jsonl'), output=str(DEST.relative_to(ROOT)),
        maximum_bytes=2*1024**3, maximum_requests=40000, maximum_member_bytes=16*1024**2,
        workers=4, selected_tracks=len(rows), archives=archives,
        recovery='Require --resume and identical code/data/config. Reverify cached headers, pinned ETags and official per-track SHA256. Never retry automatically, overwrite a mismatching file, download a full tar, or loosen source checks.',
        training_admitted=False, scope='Selected official training tracks only; full-track weak tags remain separate from continuous targets. Audio admission requires later native decoder, timing and fingerprint audits.')
    DIRECTORY.mkdir(parents=True)
    atomic_json(DIRECTORY/'launch.json', launch)
    with run_lease('next-va-v2-d-ledger'):
        stages = read_json(PLAN)['stages']
        stages['D']['jamendo_selected_audio'] = dict(job_id=JOB, status='registered',
            launch=str((DIRECTORY/'launch.json').relative_to(ROOT)), selected_tracks=len(rows), training_admitted=False)
        record_event(dict(event='D_Jamendo_selected_audio_registered', tracks=len(rows), archives=len(archives)), updates=dict(stages=stages))
    print(json.dumps(dict(launch=str(DIRECTORY/'launch.json'), selected_tracks=len(rows), archives=len(archives))))


def status():
    process = read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity = process.get('process_identity')
    report = read_json(DIRECTORY/'report.json') if (DIRECTORY/'report.json').exists() else None
    state = read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else {}
    print(json.dumps(dict(checked_at=now(), job_id=JOB, os_lease_held=is_running(JOB),
        process_identity=identity, process_identity_matches=bool(identity and process_identity(identity['pid']) == identity),
        progress=state, report_status=report.get('status') if report else None)))


def run(resume=False):
    with run_lease(JOB):
        launch = read_json(DIRECTORY/'launch.json')
        verify_hashes(launch['protected_hashes'])
        if Path(sys.executable).resolve() != Path(launch['python']) or Path.cwd().resolve() != ROOT:
            raise ValueError('Wrong Python or cwd')
        if (DIRECTORY/'report.json').exists():
            raise FileExistsError('Completed acquisition must not be restarted')
        if (DIRECTORY/'process.json').exists() and not resume:
            raise ValueError('Previous attempt exists; investigate then use --resume')
        config_pin = DIRECTORY/'configuration-sha256.json'
        configuration_sha256 = sha256(DIRECTORY/'launch.json')
        if config_pin.exists():
            if read_json(config_pin)['sha256'] != configuration_sha256:
                raise ValueError('Registered launch changed since first attempt')
        elif resume:
            raise ValueError('Missing launch identity for recovery')
        else:
            atomic_json(config_pin, dict(sha256=configuration_sha256))
        DEST.mkdir(parents=True, exist_ok=True)
        identity = process_identity()
        atomic_json(DIRECTORY/'process.json', dict(started_at=now(), process_identity=identity, resumed=resume))
        budget = Budget(DIRECTORY/'network-budget.json', launch['maximum_bytes'], launch['maximum_requests'])
        rows = {r['track_id']: r for r in read_json(SELECTED)}
        stop, progress_lock, progress_by_archive = threading.Event(), threading.Lock(), {}
        started = time.monotonic()
        def progress(row):
            with progress_lock:
                progress_by_archive[row['folder']] = dict(row, updated_at=now())
                total_verified = sum(x['verified_tracks'] for x in progress_by_archive.values())
                state = dict(stage='acquiring_selected_audio', updated_at=now(), process_identity=identity,
                    verified_tracks=total_verified, total_tracks=launch['selected_tracks'],
                    completed_archives=sum(bool(x.get('completed')) for x in progress_by_archive.values()),
                    total_archives=len(launch['archives']), budget=dict(budget.state),
                    active=[x for x in progress_by_archive.values() if not x.get('completed')],
                    seconds_this_attempt=time.monotonic()-started)
                atomic_json(DIRECTORY/'status.json', state)
                with (DIRECTORY/'events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(dict(time=now(), **row))+'\n')
                if row.get('last_track') or row.get('completed'):
                    print(json.dumps(dict(total_verified_tracks=total_verified, total_tracks=launch['selected_tracks'], **row)), flush=True)
        atomic_json(DIRECTORY/'status.json', dict(stage='starting_selected_audio', updated_at=now(),
            process_identity=identity, verified_tracks=0, total_tracks=launch['selected_tracks'], resumed=resume))
        with run_lease('next-va-v2-d-ledger'):
            stages = read_json(PLAN)['stages']
            stages['D']['jamendo_selected_audio'].update(status='running', process_identity=identity,
                true_status_module='tools.train_next_v2.jamendo_audio.download --status')
            stages['D']['phase'] = 'Jamendo_selected_audio_acquisition'
            record_event(dict(event='D_Jamendo_selected_audio_started', job_id=JOB, resumed=resume,
                process_identity=identity), updates=dict(stages=stages))
        reports = []
        try:
            with ThreadPoolExecutor(max_workers=launch['workers']) as executor:
                futures = []
                for item in launch['archives']:
                    selected = [rows[k] for k in item['selected_track_ids']]
                    futures.append(executor.submit(acquire_archive, item['url'], item['folder'], selected,
                        DEST/'audio', DEST/'index'/item['folder'], budget, stop, progress))
                try:
                    for future in as_completed(futures):
                        reports.append(future.result())
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
            records = [r for item in reports for r in item['records']]
            if len(records) != len(rows) or {r['track_id'] for r in records} != set(rows):
                raise ValueError('Acquired selection differs from frozen candidates')
            for row in records:
                audio = DEST/'audio'/row['low_quality_path']
                if sha256(audio) != row['sha256'] or audio.stat().st_size != row['bytes']:
                    raise ValueError('Final independent file verification failed')
            verify_hashes(launch['protected_hashes'])
            report = dict(status='completed_selected_audio_verified', job_id=JOB, completed_at=now(),
                selected_tracks=len(records), archives=len(reports), records=sorted(records, key=lambda r:r['track_id']),
                budget=budget.state, actual_audio_bytes=sum(r['bytes'] for r in records),
                seconds_this_attempt=time.monotonic()-started, training_admitted=False,
                whole_archives_downloaded=False, whole_archive_sha256_verified=False,
                official_per_track_sha256_verified=True,
                next_action='Audit native duration, decoding consistency and audio fingerprint overlap before any features or category-only training.')
            atomic_json(DIRECTORY/'report.json', report)
            atomic_json(DIRECTORY/'status.json', dict(stage=report['status'], updated_at=now(),
                process_identity=identity, verified_tracks=len(records), total_tracks=launch['selected_tracks'],
                completed_archives=len(reports), total_archives=len(reports), budget=budget.state))
            with run_lease('next-va-v2-d-ledger'):
                stages = read_json(PLAN)['stages']
                stages['D']['jamendo_selected_audio'].update(status=report['status'], completed_tracks=len(records),
                    report=str((DIRECTORY/'report.json').relative_to(ROOT)))
                stages['D']['phase'] = 'Jamendo_native_audio_audit_pending'
                record_event(dict(event='D_Jamendo_selected_audio_completed', tracks=len(records), bytes=report['actual_audio_bytes']), updates=dict(stages=stages))
            print(json.dumps({k:v for k,v in report.items() if k!='records'}))
        except BaseException as exc:
            failure = dict(failed_at=now(), error=f'{type(exc).__name__}: {exc}', process_identity=identity,
                           budget=budget.state, resumed=resume)
            atomic_json(DIRECTORY/f'failure-{time.time_ns()}.json', failure)
            atomic_json(DIRECTORY/'failure.json', failure)
            with run_lease('next-va-v2-d-ledger'):
                stages = read_json(PLAN)['stages']
                stages['D']['jamendo_selected_audio'].update(status='failed', error=failure['error'])
                record_event(dict(event='D_Jamendo_selected_audio_failed', **failure), updates=dict(stages=stages))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--register', action='store_true')
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--status', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
    elif args.status:
        status()
    else:
        run(args.resume)
