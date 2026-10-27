"""Inspect official MERGE ZIP metadata through bounded HTTP ranges; no audio or fitting."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import traceback
import urllib.request
import zipfile

from tools.train_next.core import atomic_json, sha256, write_status
from tools.train_next.runtime import lock_path, run_lease
from tools.train_next_v2.common import BASE, ROOT, PLAN, now, process_identity, read_json, record_event, verify_hashes
from tools.train_next_v2.run import Tee

JOB_ID = 'd-merge-package-metadata-v1'
DIRECTORY = BASE / 'analyses' / JOB_ID
DESTINATION = ROOT / 'data/datasets/merge/raw/v1.1.0/package-metadata'
SOURCE = BASE / 'resources/20261003/pinned-metadata/merge.json'
MAX_TRANSFER = 32 * 1024 * 1024
MAX_MEMBER = 8 * 1024 * 1024
ALLOWED = {'.csv', '.tsv', '.json', '.txt', '.md', '.xlsx', '.xls', '.npy', '.npz', '.pkl', '.pickle'}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def validate_range_response(status, content_range, start, end, total, body):
    if status != 206 or content_range != f'bytes {start}-{end}/{total}':
        raise ValueError('Server did not honor the exact bounded byte range; no full-download fallback')
    if len(body) != end-start+1:
        raise ValueError('Incomplete or overlong range body')


class BoundedArchive(io.RawIOBase):
    def __init__(self, item, destination, progress):
        self.item, self.destination, self.progress = item, destination, progress
        self.position, self.transferred, self.sequence = 0, 0, 0
        self.etag, self.ranges = None, []

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value = offset + (0 if whence == 0 else self.position if whence == 1 else self.item['size'] if whence == 2 else -1)
        if whence not in (0, 1, 2) or not 0 <= value <= self.item['size']:
            raise ValueError('Invalid bounded ZIP seek')
        self.position = value
        return value

    def read(self, size=-1):
        size = self.item['size']-self.position if size < 0 else min(size, self.item['size']-self.position)
        if size == 0:
            return b''
        if size > MAX_MEMBER or self.transferred + size > MAX_TRANSFER:
            raise ValueError('ZIP metadata request exceeds the frozen network budget')
        start, end = self.position, self.position+size-1
        request = urllib.request.Request(self.item['url'], headers={
            'User-Agent': 'MusicVisualization-research-metadata/1.0',
            'Accept-Encoding': 'identity', 'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=25) as response:
            status, content_range = response.status, response.headers.get('Content-Range')
            # Refuse servers ignoring Range before reading an archive-sized body.
            if status != 206 or content_range != f'bytes {start}-{end}/{self.item["size"]}':
                raise ValueError(f'Exact Range unsupported: status={status}, content_range={content_range!r}')
            etag = response.headers.get('ETag')
            if self.etag is not None and etag != self.etag:
                raise ValueError('Archive HTTP identity changed between ranges')
            self.etag = etag
            body = response.read(size+1)
            validate_range_response(status, content_range, start, end, self.item['size'], body)
        self.sequence += 1
        path = self.destination / 'ranges' / f'{self.sequence:04d}-{start}-{end}.bin'
        if path.exists():
            raise FileExistsError('Preserve previous network evidence')
        path.write_bytes(body)
        row = dict(start=start, end=end, bytes=len(body), sha256=sha256(path), etag=etag, path=relative(path))
        self.ranges.append(row)
        self.transferred += len(body)
        self.position += len(body)
        self.progress(dict(stage='reading_archive_metadata_ranges', range_requests=self.sequence,
                           transferred_bytes=self.transferred, maximum_bytes=MAX_TRANSFER))
        atomic_json(self.destination / 'range-receipts.json', self.ranges)
        return body


def register():
    if (DIRECTORY / 'launch.json').exists() or DESTINATION.exists():
        raise FileExistsError('MERGE metadata audit already registered or has artifacts')
    record = read_json(SOURCE)
    source = next(row for row in record['files'] if row['key'] == 'MERGE_Audio_Complete.zip')
    item = dict(name=source['key'], size=source['size'], md5=source['checksum'].removeprefix('md5:'), url=source['links']['self'])
    env = dict(read_json(BASE / 'analyses/c-pilot-comparison-v1/launch.json')['environment'])
    protected = {relative(SOURCE): sha256(SOURCE), relative(Path(__file__)): sha256(Path(__file__)),
                 'models/emotion_model.pth': read_json(BASE / 'protocols/b-v1.json')['legacy_model_sha256']}
    verify_hashes(protected)
    DIRECTORY.mkdir(parents=True)
    argv = [str(Path(sys.executable).resolve()), '-u', '-m', 'tools.train_next_v2.auxiliary_data.merge_audit', '--launch', str(DIRECTORY / 'launch.json')]
    launch = dict(job_id=JOB_ID, registered_at=now(), original_argv=argv, cwd=str(ROOT), python=argv[0],
        environment=env, protected_hashes=protected, source_record=relative(SOURCE), item=item,
        declared_record_license=record['metadata']['license']['id'], destination=relative(DESTINATION),
        maximum_download_bytes=MAX_TRANSFER, maximum_member_bytes=MAX_MEMBER,
        os_lock=str(lock_path(JOB_ID)), log=str(DIRECTORY / 'console.log'), gpu_used=False,
        full_audio_download=False, remote_code_execution=False, deserialize_pickle=False,
        recovery='No automatic retry. Preserve partial range receipts and failure evidence; after diagnosis use a new versioned audit. Refuse servers that ignore Range; do not silently download the audio archive.',
        admission='Metadata inspection only. Whole-archive MD5, audio decoding, time origins and cross-dataset duplicate exclusion remain required before training.')
    atomic_json(DIRECTORY / 'launch.json', launch)
    stages = read_json(PLAN)['stages']
    stages['D'].update(status='in_progress', phase='MERGE_package_metadata_audit_registered',
        metadata_audit=dict(job_id=JOB_ID, status='registered', launch=relative(DIRECTORY / 'launch.json')))
    record_event(dict(event='D_MERGE_metadata_audit_registered', launch=relative(DIRECTORY / 'launch.json')),
                 updates=dict(stages=stages, current_stage='D'))
    print(json.dumps(dict(registered=True, launch=str(DIRECTORY / 'launch.json'), maximum_download_bytes=MAX_TRANSFER)))


def run(launch):
    verify_hashes(launch['protected_hashes'])
    for key, value in launch['environment'].items():
        if os.environ.get(key) != value:
            raise ValueError('Registered metadata-audit environment differs: ' + key)
    if DESTINATION.exists() or (DIRECTORY / 'report.json').exists():
        raise ValueError('Preserve existing audit artifacts; diagnose before any new attempt')
    identity = process_identity()
    atomic_json(DIRECTORY / 'process.json', dict(started_at=now(), process_identity=identity))
    DESTINATION.mkdir(parents=True)
    (DESTINATION / 'ranges').mkdir()
    (DESTINATION / 'members').mkdir()

    def progress(row):
        write_status(DIRECTORY / 'status.json', dict(**row, process_identity=identity))
        print(json.dumps(row), flush=True)

    progress(dict(stage='reading_ZIP_directory'))
    source = BoundedArchive(launch['item'], DESTINATION, progress)
    with zipfile.ZipFile(source) as archive:
        infos = archive.infolist()
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError('Duplicate archive member names')
        index = [dict(name=info.filename, bytes=info.file_size, compressed_bytes=info.compress_size,
                      crc32=f'{info.CRC:08x}', compression=info.compress_type, directory=info.is_dir()) for info in infos]
        atomic_json(DESTINATION / 'archive-index.json', index)
        selected = [info for info in infos if not info.is_dir() and
                    (Path(info.filename).suffix.lower() in ALLOWED or any(word in Path(info.filename).name.lower() for word in ('license', 'licence', 'readme', 'copying')))]
        if sum(info.file_size for info in selected) > MAX_TRANSFER:
            raise ValueError('Uncompressed metadata exceeds audit budget; select explicitly in a new version')
        members = []
        for ordinal, info in enumerate(selected, 1):
            if info.file_size > MAX_MEMBER or info.compress_size > MAX_MEMBER:
                raise ValueError('Metadata member exceeds bound: ' + info.filename)
            # Indexed destinations prevent path traversal; bytes are never executed/deserialized.
            data = archive.read(info)
            suffix = Path(info.filename).suffix.lower() or '.txt'
            path = DESTINATION / 'members' / f'{ordinal:03d}{suffix}'
            path.write_bytes(data)
            members.append(dict(archive_name=info.filename, path=relative(path), bytes=len(data),
                                sha256=hashlib.sha256(data).hexdigest(), zip_crc32_verified=True,
                                potentially_unsafe_serialization=suffix in ('.pkl', '.pickle')))
            atomic_json(DESTINATION / 'metadata-members.json', members)
            progress(dict(stage='metadata_members_saved', completed=ordinal, total=len(selected), transferred_bytes=source.transferred))
    verify_hashes(launch['protected_hashes'])
    report = dict(job_id=JOB_ID, status='completed_metadata_only', completed_at=now(),
        declared_record_license=launch['declared_record_license'], member_count=len(index),
        audio_members=sum(Path(row['name']).suffix.lower()=='.mp3' for row in index),
        metadata_members=members, transferred_bytes=source.transferred, maximum_bytes=MAX_TRANSFER,
        full_archive_md5_verified=False, audio_downloaded=False, labels_admitted_to_training=False,
        original_model_unchanged=True, remote_code_executed=False, pickle_deserialized=False,
        archive_index=relative(DESTINATION / 'archive-index.json'), archive_index_sha256=sha256(DESTINATION / 'archive-index.json'),
        remaining=['Inspect actual metadata fields and package terms without executing serialized code.',
                   'Verify label types and artist/title/recording identities; keep static tags separate from human continuous labels.',
                   'Only then plan checksummed audio acquisition, native duration/time-origin checks and cross-partition duplicate exclusion.'])
    atomic_json(DIRECTORY / 'report.json', report)
    progress(dict(stage=report['status'], transferred_bytes=source.transferred, metadata_members=len(members)))
    stages = read_json(PLAN)['stages']
    stages['D'].update(phase='MERGE_package_metadata_downloaded_pending_inspection',
        metadata_audit=dict(job_id=JOB_ID, status=report['status'], launch=relative(DIRECTORY / 'launch.json'), report=relative(DIRECTORY / 'report.json')))
    record_event(dict(event='D_MERGE_metadata_audit_completed', bytes=source.transferred, metadata_members=len(members)), updates=dict(stages=stages))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--register', action='store_true')
    mode.add_argument('--launch', type=Path)
    args = parser.parse_args()
    with run_lease(JOB_ID):
        if args.register:
            register()
            return
        if args.launch.resolve() != DIRECTORY / 'launch.json':
            raise ValueError('Use the recorded launch path')
        launch = read_json(args.launch)
        with (DIRECTORY / 'console.log').open('a', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                try:
                    run(launch)
                except BaseException as exc:
                    traceback.print_exc()
                    failure = dict(job_id=JOB_ID, status='failed', failed_at=now(), error=f'{type(exc).__name__}: {exc}')
                    atomic_json(DIRECTORY / 'failure.json', failure)
                    stages=read_json(PLAN)['stages']
                    stages['D'].update(phase='MERGE_metadata_audit_needs_diagnosis',
                        metadata_audit=dict(job_id=JOB_ID, status='failed', launch=relative(DIRECTORY / 'launch.json'), failure=relative(DIRECTORY / 'failure.json')))
                    record_event(dict(event='D_MERGE_metadata_audit_failed', error=failure['error']), updates=dict(stages=stages))
                    raise


if __name__ == '__main__':
    main()
