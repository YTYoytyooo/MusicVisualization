"""Snapshot the already-audited Fabio exports without admitting training labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import is_running, lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, read_json, record_event, process_identity, verify_hashes
from tools.train_next_v2.research_inference.benchmark import process_alive


JOB = 'e-fabio-intake-f26ea031b964-v1'
DIRECTORY = BASE / 'annotation_pilot' / JOB
AUDIT = BASE / 'annotation_pilot/fabio-export-audit-f26ea031b964/report.json'
AUDIT_HASH = '31f4257b180f429020a1e04cc78cd09137d301e7c2d85fe56145a54da0acdc13'
AUDIO_ROOTS = ['assets/audio', 'data/datasets/external/fabio/audio']
EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'}


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def check_sources(sources):
    for source in sources:
        path = Path(source['path'])
        if path.stat().st_size != source['bytes'] or sha256(path) != source['sha256']:
            raise ValueError(f'External export changed: {path}; preserve this version and register a new audit.')


def write_once(path, data):
    """Never replace an existing committed artifact with different bytes."""
    path = Path(path)
    if not path.resolve().is_relative_to(DIRECTORY.resolve()):
        raise ValueError('Output must stay in this intake directory')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f'Existing artifact differs: {path}')
        return
    temporary = path.with_name(path.name + '.partial')
    # A previous incomplete file is evidence, not permission to overwrite it.
    if temporary.exists():
        if temporary.read_bytes() != data:
            raise ValueError(f'Incomplete artifact differs: {temporary}; inspect before recovery')
    else:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    os.rename(temporary, path)


def write_json_once(path, value):
    write_once(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def state(stage, **values):
    value = dict(job_id=JOB, stage=stage, updated_at=now(), **values)
    atomic_json(DIRECTORY/'status.json', value)
    line = json.dumps(value, ensure_ascii=False)
    with (DIRECTORY/'console.log').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')
    print(line, flush=True)


def prepare():
    if (DIRECTORY/'launch.json').exists():
        raise ValueError('Preserve registered intake configuration')
    verify_hashes({rel(AUDIT): AUDIT_HASH})
    audit = read_json(AUDIT)
    check_sources(audit['sources'])
    source_launch = read_json(BASE/'comparison_jobs/f-validation-v2-mixed-b01-cpu-v1/launch.json')
    files = list(Path(__file__).parent.glob('*.py')) + [AUDIT]
    files += [ROOT/name for name in ('tools/train_next/core.py', 'tools/train_next/runtime.py',
              'tools/train_next_v2/common.py', 'tools/train_next_v2/research_inference/benchmark.py')]
    argv = [source_launch['python'], '-u', '-m', __package__+'.intake', '--launch', str(DIRECTORY/'launch.json')]
    launch = dict(job_id=JOB, registered_at=now(), original_argv=argv, resume_argv=argv+['--resume'],
        verification_argv=argv+['--verify'], cwd=str(ROOT), python=source_launch['python'],
        environment=source_launch['environment'], code_and_data_sha256={rel(p):sha256(p) for p in files},
        source_exports=audit['sources'], audit=rel(AUDIT), audio_roots=AUDIO_ROOTS,
        os_lock=str(lock_path(JOB)), status_argv=[source_launch['python'], '-m', __package__+'.intake', '--status'],
        log=rel(DIRECTORY/'console.log'), output=rel(DIRECTORY), gpu_used=False,
        acceptance='Two byte-exact source snapshots, nine audio identities, original independent takes and masks retained; missing reviews remain unknown; no training or formal evaluation admission.',
        recovery='Confirm recorded process identity is dead and OS lease is free, diagnose failure, then original argv plus --resume. Reuse only identical committed files. Never replace mismatched artifacts, edit configuration, or promote labels.')
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    atomic_json(DIRECTORY/'launch.json', launch)
    record_event(dict(event='E_external_intake_registered', job_id=JOB, launch=rel(DIRECTORY/'launch.json')),
        run=dict(run_id=JOB, stage='E', status='registered', launch=rel(DIRECTORY/'launch.json')))
    print(json.dumps(dict(job_id=JOB, registered=True)))


def snapshot_audio(launch):
    """Match bytes in explicit local roots, never names or paths from an export."""
    path = DIRECTORY/'audio-inventory.json'
    if path.exists():
        return read_json(path)
    rows = []
    for relative in launch['audio_roots']:
        root = ROOT/relative
        if not root.resolve().is_relative_to(ROOT):
            raise ValueError('Audio search root escapes workspace')
        for audio in sorted(root.rglob('*')):
            if audio.is_file() and audio.suffix.lower() in EXTENSIONS:
                rows.append(dict(path=rel(audio), bytes=audio.stat().st_size, sha256=sha256(audio)))
    snapshot = dict(roots=launch['audio_roots'], files=rows,
        scope='Exact bytes only. Absence in these bounded roots does not establish absence elsewhere.')
    write_json_once(path, snapshot)
    return snapshot


def build_manifest(document, audit, inventory):
    audited = {s['song_id']:s for s in audit['songs']}
    if len(document['songs']) != len(audited):
        raise ValueError('Song count differs from pinned audit')
    songs = []
    for song in document['songs']:
        source = song['source']; prior = audited[source['song_id']]
        if any(source[k] != prior[k] for k in ('song_id', 'audio_sha256', 'audio_relative_path')):
            raise ValueError('Song identity differs from pinned audit')
        if song['completed'] != prior['completed'] or song['duration'] != prior['reported_duration_seconds']:
            raise ValueError('Review metadata differs from pinned audit')
        songs.append(dict(source=source, reviewer=song['reviewer'], revision=song['revision'],
            protocol=song['protocol'], label_scope=song['label_scope'], capture=song['capture'],
            reported_duration_seconds=song['duration'], native_duration_seconds=None,
            declared_completed=song['completed'], declared_artist=song['artist'], declared_style=song['style'],
            takes=[dict(id=t['id'], axes=t['axes'], point_count=len(t['points'])) for t in song['takes']],
            source_axis_audit=prior['axes'], confidence_summary_scope='Pinned per-axis audit statistics; blank confidence remains unrated.',
            exact_local_audio_matches=[r for r in inventory['files'] if r['sha256']==source['audio_sha256']],
            admission=dict(training=False, formal_evaluation=False), dataset_role='unassigned',
            pending_evidence=dict(native_time_origin_verified=None, recording_overlap_review=None,
                artist_group_review=None, raw_annotation_quality_review=None, completion_confirmed_by_reviewer=None,
                additional_independent_reviewers=None, split_allocation_frozen=None)))
    return dict(schema_version=1, job_id=JOB, status='source_snapshot_only_pending_audio_and_human_review',
        protocol=document['protocol'], sources=audit['sources'], source_audit=rel(AUDIT), songs=songs,
        rules=['Raw JSON and CSV snapshots are authoritative bytes; this manifest supplies no new labels.',
            'Never infer an audio match from filename, duration or title; exact SHA256 is only the first gate.',
            'Verify native time origin, decoding and near-duplicate/recording groups before assigning roles.',
            'Use separate native-time V/A masks and take boundaries; do not stretch time, shift labels or bridge missing intervals.',
            'Unrated confidence is unknown, not incorrect. No transition labels means unobserved, not negative.',
            'One reviewer can support a personal-preference study; 2-3 independent reviewers are required for the planned agreement study.',
            'Freeze song/recording-group allocation before training or target-domain model selection.',
            'No review, confidence rating, completion or split is generated by this preparation job.'])


def verify(launch):
    check_sources(launch['source_exports'])
    report = read_json(DIRECTORY/'report.json')
    if sha256(DIRECTORY/'audio-inventory.json') != report['audio_inventory_sha256']:
        raise ValueError('Audio inventory snapshot changed')
    for row in report['snapshots']:
        if sha256(ROOT/row['path']) != row['sha256'] or (ROOT/row['path']).read_bytes() != Path(row['source']).read_bytes():
            raise ValueError('Source snapshot is not byte-exact')
    manifest = read_json(DIRECTORY/'intake-manifest.json')
    for source in launch['source_exports']:
        if Path(source['path']).suffix == '.json':
            document = read_json(source['path'])
    expected = build_manifest(document, read_json(AUDIT), read_json(DIRECTORY/'audio-inventory.json'))
    if manifest != expected or sha256(DIRECTORY/'intake-manifest.json') != report['manifest_sha256']:
        raise ValueError('Manifest or metadata changed')
    for row in read_json(DIRECTORY/'audio-inventory.json')['files']:
        if sha256(ROOT/row['path']) != row['sha256']:
            raise ValueError('Audio inventory changed; a new intake snapshot is required')
    result = dict(job_id=JOB, status='passed', verified_at=now(), report_sha256=sha256(DIRECTORY/'report.json'),
        source_snapshots_byte_exact=True, nine_song_identity_metadata_exact=len(manifest['songs'])==9,
        raw_takes_masks_and_completion_flags_unchanged=True, human_review_performed=False,
        native_audio_timeline_verified=False, training_admitted=False, formal_evaluation_admitted=False)
    if not result['nine_song_identity_metadata_exact']:
        raise ValueError('Unexpected song count')
    if (DIRECTORY/'verification.json').exists():
        raise ValueError('Preserve prior completed verification')
    atomic_json(DIRECTORY/'verification.json', result)
    state('completed_verified', songs=9, source_snapshots=2)
    stages = read_json(PLAN)['stages']
    stages['E']['external_annotation_intake'] = dict(job_id=JOB, status='completed_verified_waiting_for_inputs',
        launch=rel(DIRECTORY/'launch.json'), manifest=rel(DIRECTORY/'intake-manifest.json'),
        report=rel(DIRECTORY/'report.json'), verification=rel(DIRECTORY/'verification.json'))
    record_event(dict(event='E_external_intake_verified', **result),
        run=dict(run_id=JOB, stage='E', status='completed_verified'), updates=dict(stages=stages))


def run(launch, resume=False):
    if (DIRECTORY/'report.json').exists():
        raise ValueError('Preparation already completed; verify it instead of rerunning')
    if (DIRECTORY/'status.json').exists() and not resume:
        raise ValueError('Existing work requires diagnosis and --resume')
    state('preparing_sources', resume=resume)
    check_sources(launch['source_exports'])
    snapshots = []
    for source in launch['source_exports']:
        path = Path(source['path']); data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != source['sha256']:
            raise ValueError('Source changed during copy')
        target = DIRECTORY/'sources'/path.name
        write_once(target, data)
        snapshots.append(dict(path=rel(target), sha256=source['sha256'], bytes=len(data), source=str(path)))
    inventory = snapshot_audio(launch)
    document = read_json(next(ROOT/r['path'] for r in snapshots if Path(r['path']).suffix=='.json'))
    manifest = build_manifest(document, read_json(AUDIT), inventory)
    write_json_once(DIRECTORY/'intake-manifest.json', manifest)
    report = dict(job_id=JOB, created_at=launch['registered_at'], status='snapshot_prepared_pending_verification',
        snapshots=snapshots, manifest_sha256=sha256(DIRECTORY/'intake-manifest.json'),
        audio_inventory_sha256=sha256(DIRECTORY/'audio-inventory.json'), songs=len(manifest['songs']),
        exact_local_audio_matched_songs=sum(bool(s['exact_local_audio_matches']) for s in manifest['songs']),
        inventoried_audio_files=len(inventory['files']), training_admitted=False, formal_evaluation_admitted=False,
        native_audio_timeline_verified=False, human_review_performed=False, gpu_used=False,
        final_test_targets_read=0, original_source_files_modified=False)
    write_json_once(DIRECTORY/'report.json', report)
    state('prepared_pending_verification', songs=report['songs'], exact_audio_matches=report['exact_local_audio_matched_songs'])
    record_event(dict(event='E_external_intake_prepared', **report),
        run=dict(run_id=JOB, stage='E', status='completed_pending_verification'))


def status():
    process = read_json(DIRECTORY/'process.json') if (DIRECTORY/'process.json').exists() else {}
    identity = process.get('process_identity')
    return dict(job_id=JOB, checked_at=now(), os_lease_active=is_running(JOB),
        process_identity=identity, process_alive=process_alive(identity) if identity else False,
        progress=read_json(DIRECTORY/'status.json') if (DIRECTORY/'status.json').exists() else None)


def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare', action='store_true'); group.add_argument('--status', action='store_true')
    group.add_argument('--launch', type=Path); parser.add_argument('--resume', action='store_true')
    parser.add_argument('--verify', action='store_true'); args=parser.parse_args()
    if args.status: print(json.dumps(status())); return
    if args.prepare: prepare(); return
    if args.launch.resolve() != (DIRECTORY/'launch.json').resolve(): raise ValueError('Wrong launch file')
    launch=read_json(args.launch); verify_hashes(launch['code_and_data_sha256'])
    if Path.cwd().resolve()!=ROOT or Path(sys.executable).resolve()!=Path(launch['python']).resolve():
        raise ValueError('Wrong working directory or Python')
    if any(os.environ.get(k)!=v for k,v in launch['environment'].items()):
        raise ValueError('Frozen environment mismatch')
    if args.verify and (DIRECTORY/'verification.json').exists():
        raise ValueError('Verification already completed; inspect --status instead')
    if not args.verify and (DIRECTORY/'report.json').exists():
        raise ValueError('Preparation already completed; verify it instead')
    previous=status()
    if previous['process_alive'] or previous['os_lease_active']:
        raise RuntimeError('Existing intake process is active')
    with run_lease(JOB):
        atomic_json(DIRECTORY/'process.json', dict(process_identity=process_identity(), started_at=now(),
            argv=sys.argv, verification=args.verify, resume=args.resume))
        record_event(dict(event='E_external_intake_started', job_id=JOB, verification=args.verify,
            process_identity=process_identity()), run=dict(run_id=JOB, stage='E', status='running'))
        try:
            if args.verify: verify(launch)
            else: run(launch, args.resume)
        except Exception:
            state('failed', error=traceback.format_exc())
            record_event(dict(event='E_external_intake_failed',job_id=JOB),
                run=dict(run_id=JOB, stage='E', status='failed'))
            raise


if __name__=='__main__': main()
