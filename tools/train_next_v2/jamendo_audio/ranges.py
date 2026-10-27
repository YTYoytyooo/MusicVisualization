"""Exact HTTP ranges, immutable source pins and selected-member tar reads."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import re
import tarfile
import threading
import time

import requests

from tools.train_next.core import atomic_json, sha256


class Budget:
    """Reserve worst-case bytes before I/O; interrupted requests still count."""
    def __init__(self, path, maximum_bytes, maximum_requests):
        self.path = Path(path)
        self.maximum_bytes = maximum_bytes
        self.maximum_requests = maximum_requests
        self.lock = threading.Lock()
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'bytes_reserved': 0, 'requests_reserved': 0}

    def reserve(self, size):
        with self.lock:
            new = dict(bytes_reserved=self.state['bytes_reserved'] + size,
                       requests_reserved=self.state['requests_reserved'] + 1)
            if new['bytes_reserved'] > self.maximum_bytes or new['requests_reserved'] > self.maximum_requests:
                raise ValueError('Frozen network budget exhausted; no request sent')
            atomic_json(self.path, new)
            self.state = new


def checked_headers(response, start, size, pinned=None):
    """Validate headers before touching any response body."""
    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
    if response.status_code != 206 or not match:
        raise ValueError(f'Exact HTTP206 range required; got status={response.status_code} '
                         f'Content-Range={response.headers.get("Content-Range")!r}; body not read')
    first, last, total = map(int, match.groups())
    if (first, last) != (start, start + size - 1) or not 1024 <= total <= 4*1024**3 or last >= total:
        raise ValueError('Range bounds or tar size mismatch; body not read')
    if response.headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('Encoded range rejected')
    if response.headers.get('Content-Length') != str(size):
        raise ValueError('Range content length mismatch')
    etag = response.headers.get('ETag', '')
    if not re.fullmatch(r'"[^"\r\n]+"', etag):
        raise ValueError('A strong ETag is required for resumable ranges')
    source = dict(url=response.url, bytes=total, etag=etag)
    if pinned is not None and source != pinned:
        raise ValueError('Archive identity changed; preserve cache and stop')
    return source


def parse_header(data, folder, offset, archive_size):
    if len(data) != 512:
        raise ValueError('Tar header must contain exactly512 bytes')
    if data == b'\0'*512:
        return None
    member = tarfile.TarInfo.frombuf(data, 'utf-8', 'strict')
    # No extraction API, links, extensions, absolute paths or dot traversal.
    if member.isdir():
        if member.name != folder or member.size != 0:
            raise ValueError('Unexpected tar directory')
    elif member.type in (tarfile.REGTYPE, tarfile.AREGTYPE):
        if not re.fullmatch(re.escape(folder) + r'/\d+\.low\.mp3', member.name):
            raise ValueError('Unexpected or unsafe tar member name')
        if member.size <= 0:
            raise ValueError('Empty audio member')
    else:
        raise ValueError('Unsupported tar member type; no links or extensions accepted')
    end = offset + 512 + ((member.size+511)//512)*512
    if end > archive_size or offset % 512:
        raise ValueError('Tar member outside pinned archive')
    return dict(name=member.name, bytes=member.size, type=member.type.decode('ascii'),
                header_offset=offset, data_offset=offset+512, next_offset=end,
                is_file=member.type in (tarfile.REGTYPE, tarfile.AREGTYPE))


class ArchiveRanges:
    def __init__(self, url, directory, budget, session=None):
        self.url = url
        self.directory = Path(directory)
        (self.directory/'headers').mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.session = session or requests.Session()
        self.source_path = self.directory/'source.json'
        self.source = json.loads(self.source_path.read_text()) if self.source_path.exists() else None
        if self.source and self.source['url'] != url:
            raise ValueError('Source URL differs from saved archive')
        self.network_bytes = 0
        self.header_hits = 0

    def fetch(self, start, size):
        if not 0 < size <= 16*1024**2 or start < 0:
            raise ValueError('Single range exceeds frozen bound')
        if self.source and start + size > self.source['bytes']:
            raise ValueError('Range beyond pinned source')
        headers = {'Range': f'bytes={start}-{start+size-1}', 'Accept-Encoding': 'identity',
                   'User-Agent': 'MusicVisualization-research/1.0'}
        if self.source:
            headers['If-Match'] = self.source['etag']
        self.budget.reserve(size)
        with self.session.get(self.url, headers=headers, stream=True, allow_redirects=False, timeout=(10, 30)) as response:
            if response.url != self.url:
                raise ValueError('Unexpected range URL')
            source = checked_headers(response, start, size, self.source)
            data = response.raw.read(size+1, decode_content=False)
            if len(data) != size:
                raise ValueError('Range body truncated or oversized')
        if self.source is None:
            atomic_json(self.source_path, source)
            self.source = source
        self.network_bytes += len(data)
        return data

    def header(self, offset):
        path = self.directory/'headers'/f'{offset:012d}.json'
        if path.exists():
            record = json.loads(path.read_text())
            data = base64.b64decode(record['data'], validate=True)
            if (record['source'] != self.source or record['offset'] != offset or len(data) != 512
                    or hashlib.sha256(data).hexdigest() != record['sha256']):
                raise ValueError('Cached tar header failed identity/hash check')
            self.header_hits += 1
            return data
        data = self.fetch(offset, 512)
        atomic_json(path, dict(offset=offset, source=self.source,
            data=base64.b64encode(data).decode('ascii'), sha256=hashlib.sha256(data).hexdigest()))
        return data

    def close(self):
        self.session.close()


def save_selected(data, destination, expected_sha256):
    """No final audio file is published unless its official digest matches."""
    destination = Path(destination)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError('Selected track does not match official SHA256')
    if destination.exists():
        if sha256(destination) != expected_sha256:
            raise ValueError('Existing track differs; refusing overwrite')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    if temporary.exists():
        raise FileExistsError('Inspect interrupted local write before recovery')
    with temporary.open('xb') as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    if sha256(temporary) != expected_sha256:
        raise ValueError('Local audio write hash mismatch')
    temporary.replace(destination)


def acquire_archive(url, folder, selected, destination, index_directory, budget, stop, progress, session=None):
    """Scan tar headers; request audio bodies only for the frozen selection."""
    if not re.fullmatch(r'\d{2}', folder):
        raise ValueError('Unexpected archive folder')
    wanted = {row['low_quality_path']: row for row in selected}
    if len(wanted) != len(selected) or not wanted:
        raise ValueError('Empty or duplicate selection')
    for name, row in wanted.items():
        if not re.fullmatch(folder+r'/\d+\.low\.mp3', name) or not re.fullmatch(r'[a-f0-9]{64}', row['expected_audio_sha256']):
            raise ValueError('Invalid frozen selected path or SHA256')
    archive = ArchiveRanges(url, index_directory, budget, session)
    seen, records, offset, headers = set(), [], 0, 0
    started = time.monotonic()
    try:
        while wanted:
            if stop.is_set():
                raise RuntimeError('Stopped after another archive failed; cache preserved')
            if headers >= 1200:
                raise ValueError('Tar header count limit reached')
            raw = archive.header(offset)
            member = parse_header(raw, folder, offset, archive.source['bytes'])
            headers += 1
            if member is None:
                raise ValueError('End of tar before selected members found')
            if member['name'] in seen:
                raise ValueError('Duplicate tar member in scanned range')
            seen.add(member['name'])
            if member['is_file'] and member['name'] in wanted:
                row = wanted[member['name']]
                if member['bytes'] > 16*1024**2:
                    raise ValueError('Selected member exceeds frozen16MiB bound')
                output = Path(destination)/member['name']
                if output.exists():
                    if output.stat().st_size != member['bytes'] or sha256(output) != row['expected_audio_sha256']:
                        raise ValueError('Existing selected file failed size/hash verification')
                else:
                    body = archive.fetch(member['data_offset'], member['bytes'])
                    save_selected(body, output, row['expected_audio_sha256'])
                receipt = dict(track_id=row['track_id'], low_quality_path=member['name'], bytes=member['bytes'],
                    sha256=row['expected_audio_sha256'], source=archive.source, data_offset=member['data_offset'],
                    whole_archive_sha256_verified=False, official_track_sha256_verified=True, training_admitted=False)
                atomic_json(Path(index_directory)/(row['track_id']+'.json'), receipt)
                records.append(receipt)
                del wanted[member['name']]
                progress(dict(folder=folder, headers=headers, verified_tracks=len(records), remaining=len(wanted),
                              last_track=row['track_id'], network_bytes_this_attempt=archive.network_bytes))
            elif headers % 10 == 0:
                progress(dict(folder=folder, headers=headers, verified_tracks=len(records), remaining=len(wanted),
                              network_bytes_this_attempt=archive.network_bytes))
            offset = member['next_offset']
        report = dict(folder=folder, status='completed_selected_members_verified', records=records,
            source=archive.source, scanned_headers=headers, cached_header_hits=archive.header_hits,
            network_bytes_this_attempt=archive.network_bytes, seconds=time.monotonic()-started,
            full_archive_downloaded=False, whole_archive_sha256_verified=False)
        atomic_json(Path(index_directory)/'report.json', report)
        progress(dict(folder=folder, headers=headers, verified_tracks=len(records), remaining=0, completed=True,
                      network_bytes_this_attempt=archive.network_bytes))
        return report
    finally:
        archive.close()
