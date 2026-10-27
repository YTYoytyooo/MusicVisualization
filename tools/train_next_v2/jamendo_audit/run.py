"""Freeze and run audio gates after the selected-only acquisition is complete."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import run_lease, is_running, lock_path
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, process_identity, read_json, record_event, verify_hashes
from tools.datasets.index_supplemental import FPCALC, MatchIndex
from tools.train_next_v2.auxiliary_data.merge_audio_audit import fingerprint
from tools.train_next_v2.auxiliary_data.merge_candidates import connected_groups
from .native import FFMPEG, inspect

JOB = 'd-jamendo-native-audit-v1'
DIRECTORY = BASE/'analyses'/JOB
OUTPUT = ROOT/'data/datasets/mtg_jamendo/prepared/native-audit-v1'
SELECTED = ROOT/'data/datasets/mtg_jamendo/prepared/candidates-v1/selected.json'
DOWNLOAD = BASE/'resources/20261004/jobs/d-jamendo-selected-audio-v1'
AUDIO = ROOT/'data/datasets/mtg_jamendo/raw/selected-audio-v1/audio'
MERGE = ROOT/'data/datasets/merge/prepared/audio-audit-v1'
MAIN = ROOT/'data/datasets/combined/prepared/v1/manifest.jsonl'


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def register():
    if DIRECTORY.exists() or OUTPUT.exists():
        raise FileExistsError('Preserve previous audit configuration')
    source = read_json(DOWNLOAD/'launch.json')
    selected = read_json(SELECTED)
    protected = dict(source['protected_hashes'])
    previous = read_json(BASE/'analyses/d-merge-audio-audit-v1/report.json')
    for path in (MERGE/'main-fingerprint-receipts.json', MERGE/'audio-inventory.json'):
        verify_hashes({rel(path): previous['outputs'][rel(path)]})
    files = [SELECTED, MAIN, DOWNLOAD/'launch.json', MERGE/'main-fingerprint-receipts.json',
             MERGE/'audio-inventory.json', BASE/'analyses/d-merge-audio-audit-v1/report.json', FPCALC, FFMPEG,
             ROOT/'tools/datasets/index_supplemental.py',
             ROOT/'tools/train_next_v2/auxiliary_data/merge_audio_audit.py',
             ROOT/'tools/train_next_v2/auxiliary_data/merge_candidates.py']
    files += list(Path(__file__).parent.glob('*.py'))
    protected.update({rel(p): sha256(p) for p in files})
    verify_hashes(protected)
    argv = [source['python'], '-u', '-m', 'tools.train_next_v2.jamendo_audit.run', '--run']
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, recovery_argv=argv+['--resume'],
        cwd=str(ROOT), python=source['python'], environment=source['environment'], protected_hashes=protected,
        os_lock=str(lock_path(JOB)), log=rel(DIRECTORY/'events.jsonl'),
        selected_tracks=len(selected), expected_audio={r['low_quality_path']:r['expected_audio_sha256'] for r in selected},
        required_acquisition_job=source['job_id'], required_acquisition_status='completed_selected_audio_verified',
        native_duration_tolerance_seconds=0.25, worker_count=1, gpu_used=False, targets_used=False,
        library_versions={k:importlib.metadata.version(k) for k in ('numpy','soundfile','requests')},
        fingerprint_cache=rel(ROOT/'data/datasets/.fingerprints'/sha256(FPCALC)[:16]),
        gates=['official per-track SHA256', 'strict ffmpeg decoding with no stderr',
               'full native finite samples; no libsndfile diagnostics', 'bulk/stream exact float32 sample equality',
               'native duration within0.25seconds of full-track metadata',
               'recording exclusion against pinned main/MERGE fingerprints and exact hashes',
               'within-pilot connected recording deduplication'],
        reference_limitations=['ManyMusic and official Jamendo heldout identities/artists were excluded by the pinned candidate screen. Audio fingerprints for their full external libraries are unavailable; no exhaustive acoustic exclusion is claimed.',
            'Five MERGE audio records had no valid fingerprint in the prior audit; their metadata and exact audio hashes remain excluded.',
            'Chromaprint is conservative screening, not proof against every cover, remix, alias or short overlap.'],
        recovery='After diagnosis and inactive OS lease, use identical --resume with recorded code/data. Reverify each saved native receipt and source SHA; never relax a failing audio gate.',
        next_gate='Freeze a matched category-only/control protocol and feature inputs. These weak uploader tags are not continuous VA labels.')
    DIRECTORY.mkdir(parents=True)
    atomic_json(DIRECTORY/'launch.json', launch)
    with run_lease('next-va-v2-d-ledger'):
        stages = read_json(PLAN)['stages']
        stages['D']['jamendo_native_audit'] = dict(job_id=JOB, status='registered_waiting_for_acquisition',
            launch=rel(DIRECTORY/'launch.json'), source_audio_expected=len(selected), training_admitted=False)
        record_event(dict(event='D_Jamendo_native_audit_registered', launch=rel(DIRECTORY/'launch.json')), updates=dict(stages=stages))
    print(json.dumps(dict(launch=rel(DIRECTORY/'launch.json'), status='registered_waiting_for_acquisition')))


def reference_values(receipt):
    fp = receipt['fingerprint']
    verify_hashes({fp['path']:fp['sha256']})
    values = read_json(ROOT/fp['path'])['fingerprint']
    if len(values) != fp['frames'] or len(values) < 80 or any(not isinstance(v, int) or not 0 <= v < 2**32 for v in values):
        raise ValueError('Invalid pinned reference fingerprint')
    return np.asarray(values, dtype=np.uint32)


def screen_recordings(selected, checks, graph):
    groups = connected_groups(graph)
    candidates = {r['song_id']:r for r in selected}
    if len(candidates) != len(selected) or len(checks) != len(candidates) or {r['song_id'] for r in checks} != set(candidates):
        raise ValueError('Native receipts do not cover the exact selected pool')
    verified = {r['song_id']:r for r in checks if r['passed']}
    eligible, excluded = [], []
    for sid, row in sorted(candidates.items()):
        group = groups[sid]
        reason = ('native_audio_gate_failed' if sid not in verified else
                  'recording_link_to_protected_or_rejected_audio' if any(x not in verified for x in group) else
                  'duplicate_candidate_recording' if sid != min(group) else None)
        if reason:
            excluded.append(dict(song_id=sid, reason=reason, recording_group=group))
        else:
            eligible.append(dict(**row, audio=verified[sid], recording_group=group, audio_gate_passed=True,
                auxiliary_task='whole_track_weak_category', fitting_enabled=False))
    return eligible, excluded


def run(resume=False):
    with run_lease(JOB):
        launch = read_json(DIRECTORY/'launch.json')
        verify_hashes(launch['protected_hashes'])
        if Path(sys.executable).resolve() != Path(launch['python']) or Path.cwd().resolve() != ROOT:
            raise ValueError('Wrong Python or cwd')
        for name, version in launch['library_versions'].items():
            if importlib.metadata.version(name) != version:
                raise ValueError('Audit library version changed')
        if is_running(launch['required_acquisition_job']) or not (DOWNLOAD/'report.json').exists():
            raise RuntimeError('Wait for completed acquisition; no partial-pool admission')
        if (DIRECTORY/'report.json').exists():
            raise FileExistsError('Completed audit cannot be restarted')
        if (DIRECTORY/'process.json').exists() and not resume:
            raise ValueError('Existing attempt requires diagnosed --resume')
        acquired = read_json(DOWNLOAD/'report.json')
        if acquired['status'] != launch['required_acquisition_status'] or acquired['selected_tracks'] != launch['selected_tracks']:
            raise ValueError('Acquisition not complete')
        acquired_map = {r['low_quality_path']:r['sha256'] for r in acquired['records']}
        if len(acquired['records']) != len(acquired_map) or acquired_map != launch['expected_audio']:
            raise ValueError('Acquisition differs from frozen candidates')
        pin = DIRECTORY/'configuration-sha256.json'
        config = dict(launch_sha256=sha256(DIRECTORY/'launch.json'), acquisition_report_sha256=sha256(DOWNLOAD/'report.json'))
        if pin.exists():
            if read_json(pin) != config:
                raise ValueError('Recovery config or acquisition changed')
        elif resume:
            raise ValueError('Missing recovery config identity')
        else:
            atomic_json(pin, config)
        identity = process_identity()
        atomic_json(DIRECTORY/'process.json', dict(started_at=now(), process_identity=identity, resumed=resume))
        (OUTPUT/'records').mkdir(parents=True, exist_ok=True)
        cache = ROOT/launch['fingerprint_cache']
        cache.mkdir(parents=True, exist_ok=True)
        def publish(**row):
            state = dict(updated_at=now(), process_identity=identity, **row)
            atomic_json(DIRECTORY/'status.json', state)
            with (DIRECTORY/'events.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(state)+'\n')
            print(json.dumps(state), flush=True)
        selected = read_json(SELECTED)
        checks = []
        try:
            publish(stage='native_audio_gates', completed=0, total=len(selected))
            for row in selected:
                path = AUDIO/row['low_quality_path']
                if sha256(path) != row['expected_audio_sha256']:
                    raise ValueError('Candidate audio changed: '+row['track_id'])
                saved = OUTPUT/'records'/(row['track_id']+'.json')
                if saved.exists():
                    receipt = read_json(saved)
                    if receipt['audio_sha256'] != row['expected_audio_sha256'] or receipt['config'] != config:
                        raise ValueError('Saved native receipt belongs to other inputs')
                    if receipt.get('fingerprint'):
                        reference_values(receipt)
                    verify_hashes(receipt['diagnostic_files'])
                else:
                    receipt = dict(song_id=row['song_id'], track_id=row['track_id'], audio_path=rel(path),
                        audio_sha256=row['expected_audio_sha256'], config=config, passed=False)
                    logdir = OUTPUT/'stderr'/row['track_id']/f'attempt-{time.time_ns()}'
                    try:
                        native = inspect(path, row['metadata_duration_seconds'], logdir)
                        fp, _ = fingerprint(path, row['expected_audio_sha256'], cache)
                        receipt.update(passed=True, native_audio=native, fingerprint=fp)
                    except Exception as exc:
                        receipt['error'] = f'{type(exc).__name__}: {exc}'
                    receipt['diagnostic_files'] = {rel(p):sha256(p) for p in logdir.glob('*.log')}
                    atomic_json(saved, receipt)
                checks.append(receipt)
                if len(checks) % 20 == 0 or len(checks) == len(selected):
                    publish(stage='native_audio_gates', completed=len(checks), total=len(selected),
                            passed=sum(r['passed'] for r in checks))
            main_receipts = read_json(MERGE/'main-fingerprint-receipts.json')
            main_rows = [json.loads(line) for line in MAIN.read_text(encoding='utf-8').splitlines()]
            main_expected = {r['song_id']:r['audio_sha256'] for r in main_rows}
            if len(main_receipts) != len(main_expected) or {r['song_id']:r['audio_sha256'] for r in main_receipts} != main_expected:
                raise ValueError('Pinned reference fingerprints do not cover the fixed main corpus')
            merge = read_json(MERGE/'audio-inventory.json')
            references = main_receipts + merge
            index, index_ids, graph = MatchIndex(), [], {}
            for i, row in enumerate(references):
                graph[row['song_id']] = dict(identity_keys=[('audio_sha256',row['audio_sha256']), ('fp_link',row['song_id'])])
                if row.get('fingerprint'):
                    index.add(reference_values(row))
                    index_ids.append(row['song_id'])
                if (i+1) % 1000 == 0 or i+1 == len(references):
                    publish(stage='pinned_reference_identity_index', completed=i+1, total=len(references), fingerprints=len(index_ids))
            links = []
            for i, receipt in enumerate(sorted(checks, key=lambda r:r['song_id'])):
                keys = [('audio_sha256',receipt['audio_sha256']), ('fp_link',receipt['song_id'])]
                if receipt['passed']:
                    values = reference_values(receipt)
                    for other, details in index.matches(values).items():
                        keys.append(('fp_link',index_ids[other]))
                        links.append(dict(song_id=receipt['song_id'], matched_song_id=index_ids[other], **details))
                    index.add(values)
                    index_ids.append(receipt['song_id'])
                graph[receipt['song_id']] = dict(identity_keys=keys)
                if (i+1) % 20 == 0 or i+1 == len(checks):
                    publish(stage='recording_overlap_screen', completed=i+1, total=len(checks), links=len(links))
            verified = {r['song_id']:r for r in checks if r['passed']}
            eligible, excluded = screen_recordings(selected, checks, graph)
            atomic_json(OUTPUT/'checks.json', checks)
            atomic_json(OUTPUT/'recording-links.json', links)
            atomic_json(OUTPUT/'recording-exclusions.json', excluded)
            atomic_json(OUTPUT/'eligible-category-candidates.json', eligible)
            verify_hashes(launch['protected_hashes'])
            report = dict(job_id=JOB, status='completed_native_audio_gate', completed_at=now(),
                checked=len(checks), native_passed=len(verified), eligible_categories=len(eligible), excluded=len(excluded),
                reference_audio_records=len(references), reference_fingerprints=sum(bool(r.get('fingerprint')) for r in references),
                exclusion_reasons=dict(Counter(r['reason'] for r in excluded)),
                selected_tag_counts=dict(Counter(tag for row in eligible for tag in row['selected_positive_tags'])),
                failures=[r for r in checks if not r['passed']], recording_links=len(links),
                limitations=launch['reference_limitations'], targets_used=False, gpu_used=False, training_admitted=False,
                next_action=launch['next_gate'], outputs={rel(p):sha256(p) for p in OUTPUT.glob('*.json')})
            atomic_json(DIRECTORY/'report.json', report)
            with run_lease('next-va-v2-d-ledger'):
                stages = read_json(PLAN)['stages']
                stages['D']['jamendo_native_audit'].update(status=report['status'], report=rel(DIRECTORY/'report.json'),
                    eligible_categories=len(eligible), native_passed=len(verified))
                stages['D']['phase'] = 'Jamendo_category_protocol_pending'
                record_event(dict(event='D_Jamendo_native_audio_gate_completed', eligible=len(eligible), excluded=len(excluded)), updates=dict(stages=stages))
            publish(stage=report['status'], eligible_categories=len(eligible), excluded=len(excluded))
        except BaseException as exc:
            failure = dict(failed_at=now(), error=f'{type(exc).__name__}: {exc}')
            atomic_json(DIRECTORY/f'failure-{time.time_ns()}.json', failure)
            atomic_json(DIRECTORY/'failure.json', failure)
            record_event(dict(event='D_Jamendo_native_audit_failed', **failure))
            raise


def status():
    process = read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity = process.get('process_identity')
    report = read_json(DIRECTORY/'report.json') if (DIRECTORY/'report.json').exists() else {}
    state = read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else {}
    print(json.dumps(dict(checked_at=now(), job_id=JOB, os_lease_held=is_running(JOB),
        process_identity=identity, process_identity_matches=bool(identity and process_identity(identity['pid']) == identity),
        saved_native_receipts=len(list((OUTPUT/'records').glob('*.json'))),
        progress=state, report_status=report.get('status'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--status', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.register:
        register()
    elif args.status:
        status()
    else:
        run(args.resume)
