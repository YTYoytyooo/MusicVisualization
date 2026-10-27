"""Download the pinned MERGE archive with checked, explicitly resumable transfers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback
import urllib.request
import zipfile

from tools.train_next.core import atomic_json, sha256
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, process_identity, read_json, record_event, verify_hashes

JOB = 'd-merge-audio-download-v1'
FOLDER = BASE / 'analyses' / JOB
DESTINATION = ROOT / 'data/datasets/merge/raw/v1.1.0/MERGE_Audio_Complete.zip'


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def check_headers(status, headers, offset, size):
    if offset:
        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', headers.get('Content-Range', ''))
        if status != 206 or not match or tuple(map(int, match.groups())) != (offset, size - 1, size):
            raise ValueError('Server did not honor exact resume range; partial file preserved')
    elif status != 200:
        raise ValueError('Expected a complete initial response')
    length = headers.get('Content-Length')
    if length is not None and int(length) != size - offset:
        raise ValueError('Response size differs from pinned archive')
    if headers.get('Content-Encoding', 'identity').lower() != 'identity':
        raise ValueError('Encoded response is not byte-identical to the archive')
    if 'text/html' in headers.get('Content-Type', '').lower():
        raise ValueError('Received HTML instead of the archive')


def checksums(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            md5.update(block)
            sha.update(block)
    return dict(md5=md5.hexdigest(), sha256=sha.hexdigest())


def transfer(item, destination, publish, *, resume=False, opener=urllib.request.urlopen):
    partial = destination.with_name(destination.name + '.part')
    if destination.exists():
        sums = checksums(destination)
        if destination.stat().st_size != item['size'] or sums['md5'] != item['md5']:
            raise ValueError('Existing archive differs from pinned checksum; preserved')
        return dict(path=relative(destination), bytes=item['size'], **sums, reused=True)
    if partial.exists() and not resume:
        raise ValueError('Existing partial requires --resume after diagnosis')
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item['size']:
        raise ValueError('Partial exceeds pinned size; preserved')
    if offset < item['size']:
        headers = {'User-Agent': 'MusicVisualization-MERGE-research/1.0', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(item['url'], headers=headers)
        started, last_update = time.monotonic(), 0.
        with opener(request, timeout=30) as response:
            if not response.geturl().startswith('https://'):
                raise ValueError('HTTPS redirect required')
            check_headers(response.status, response.headers, offset, item['size'])
            publish(dict(stage='response_verified', offset=offset, http_status=response.status,
                         etag=response.headers.get('ETag'), content_length=response.headers.get('Content-Length')))
            with partial.open('ab' if partial.exists() else 'xb') as stream:
                received = offset
                while True:
                    if time.monotonic() - started > 1800:
                        raise TimeoutError('30-minute transfer budget exhausted; partial preserved')
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if received + len(block) > item['size']:
                        raise ValueError('Response exceeds pinned size; partial preserved')
                    stream.write(block)
                    received += len(block)
                    if time.monotonic() - last_update >= 10:
                        stream.flush()
                        publish(dict(stage='downloading', bytes=received, expected_bytes=item['size'],
                                     bytes_per_second=(received - offset) / max(.001, time.monotonic() - started)))
                        last_update = time.monotonic()
                stream.flush()
                os.fsync(stream.fileno())
    if partial.stat().st_size != item['size']:
        raise ValueError('Incomplete transfer; partial retained for explicit recovery')
    publish(dict(stage='verifying_archive', bytes=item['size'], expected_bytes=item['size']))
    sums = checksums(partial)
    if sums['md5'] != item['md5']:
        raise ValueError('Official archive MD5 mismatch; partial preserved, not admitted')
    partial.rename(destination)
    return dict(path=relative(destination), bytes=item['size'], **sums, reused=False)


def register():
    if FOLDER.exists():
        raise FileExistsError('Preserve previous download registration')
    metadata_launch = read_json(BASE / 'analyses/d-merge-package-metadata-v1/launch.json')
    item = metadata_launch['item']
    if item['url'] != 'https://zenodo.org/api/records/13939205/files/MERGE_Audio_Complete.zip/content':
        raise ValueError('Unexpected archive URL')
    if item['size'] != 1162604519 or item['md5'] != '167f7ba80ab8eca8eee5a8668498a419':
        raise ValueError('Unexpected source version')
    hashes = {key: value for key, value in metadata_launch['protected_hashes'].items()
              if not key.endswith('merge_audit.py')}
    for path in [Path(__file__), ROOT / 'tools/train_next_v2/common.py', ROOT / 'tools/train_next/core.py',
                 ROOT / 'tools/train_next/runtime.py', ROOT / metadata_launch['destination'] / 'archive-index.json']:
        hashes[relative(path)] = sha256(path)
    verify_hashes(hashes)
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', 'tools.train_next_v2.auxiliary_data.merge_download', '--run']
    launch = dict(job_id=JOB, stage='D', registered_at=now(), original_argv=argv,
        recovery_argv=argv + ['--resume'], cwd=str(ROOT), python=argv[0],
        environment=metadata_launch['environment'], protected_hashes=hashes, item=item,
        destination=relative(DESTINATION), archive_index=metadata_launch['destination'] + '/archive-index.json',
        os_lock=str(lock_path(JOB)), log=relative(FOLDER / 'console.log'), status_path=relative(FOLDER / 'status.json'),
        report=relative(FOLDER / 'report.json'), gpu_used=False, archive_extracted=False,
        declared_record_license='cc-by-nc-4.0', purpose='Local noncommercial research preparation; no audio redistribution.',
        recovery='No automatic retries. Check OS lease and process identity; diagnose failure, preserve attempts and partial, then use recovery_argv with exact hashes. A checksum-mismatched full partial must not be retried unchanged.',
        admission='Download verification only; labels, audio/time audit and recording overlap remain separate gates.')
    FOLDER.mkdir(parents=True)
    atomic_json(FOLDER / 'launch.json', launch)
    stages = read_json(PLAN)['stages']
    stages['D']['audio_download'] = dict(job_id=JOB, status='registered', launch=relative(FOLDER/'launch.json'),
                                        original_argv=argv, recovery_argv=argv+['--resume'])
    record_event(dict(event='D_MERGE_audio_download_registered', launch=relative(FOLDER/'launch.json')),
                 updates=dict(stages=stages))


def run(resume):
    launch = read_json(FOLDER / 'launch.json')
    verify_hashes(launch['protected_hashes'])
    if (FOLDER / 'report.json').exists():
        raise FileExistsError('Completed download must not be restarted')
    previous_attempts = sorted(FOLDER.glob('attempt-*.json'))
    if previous_attempts and not resume:
        raise ValueError('Prior attempt requires explicit --resume')
    identity = process_identity()
    attempt = dict(attempt=len(previous_attempts)+1, started_at=now(), process_identity=identity,
                   actual_argv=launch['original_argv']+(['--resume'] if resume else []))
    attempt_path = FOLDER / f"attempt-{attempt['attempt']:03d}.json"
    atomic_json(attempt_path, attempt)
    atomic_json(FOLDER / 'process.json', attempt)
    with (FOLDER / 'console.log').open('a', encoding='utf-8') as log:
        def publish(value):
            value = dict(**value, process_identity=identity, updated_at=now())
            atomic_json(FOLDER / 'status.json', value)
            line = json.dumps(value)
            log.write(line + '\n'); log.flush()
            print(line, flush=True)
        try:
            publish(dict(stage='starting', job_id=JOB))
            result = transfer(launch['item'], DESTINATION, publish, resume=resume)
            pinned = read_json(ROOT / launch['archive_index'])
            with zipfile.ZipFile(DESTINATION) as archive:
                members = archive.infolist()
                actual = {info.filename: (info.file_size, info.CRC) for info in members}
                expected = {row['name']: (row['bytes'], int(row['crc32'], 16)) for row in pinned}
                if len(actual) != len(members) or actual != expected:
                    raise ValueError('Archive directory differs from inspected metadata')
            report = dict(job_id=JOB, status='completed_archive_verified', completed_at=now(),
                process_identity=identity, archive=result, members=len(members),
                audio_members=sum(info.filename.lower().endswith('.mp3') for info in members),
                archive_index_matches=True, official_md5_verified=True, labels_used=False,
                training_admitted=False, archive_extracted=False, launch_sha256=sha256(FOLDER/'launch.json'))
            atomic_json(FOLDER / 'report.json', report)
            attempt.update(status=report['status'], completed_at=now())
            atomic_json(attempt_path, attempt)
            stages = read_json(PLAN)['stages']
            stages['D']['audio_download'].update(status=report['status'], report=relative(FOLDER/'report.json'),
                                                bytes=result['bytes'], sha256=result['sha256'])
            record_event(dict(event='D_MERGE_audio_archive_verified', bytes=result['bytes'], report=relative(FOLDER/'report.json')),
                         updates=dict(stages=stages))
            publish(dict(stage=report['status'], bytes=result['bytes'], audio_members=report['audio_members']))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            attempt.update(status='failed', completed_at=now(), error=error)
            atomic_json(attempt_path, attempt)
            log.write(traceback.format_exc() + '\n'); log.flush()
            publish(dict(stage='failed', error=error))
            record_event(dict(event='D_MERGE_audio_download_failed', error=error, attempt=relative(attempt_path)))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    with run_lease(JOB):
        register() if args.register else run(args.resume)
