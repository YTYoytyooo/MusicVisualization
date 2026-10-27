"""Boundary tests for selected-only downloading, source drift and safe recovery."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import threading
import unittest

from tools.train_next_v2.common import BASE
from tools.train_next_v2.jamendo_audio.ranges import (
    ArchiveRanges, Budget, acquire_archive, checked_headers, parse_header, save_selected,
)


URL = 'https://cdn.freesound.org/mtg-jamendo/test.tar'


class FakeResponse:
    def __init__(self, data, start, size, etag='"version1"', status=206):
        self.url, self.status_code = URL, status
        self.headers = {'Content-Range': f'bytes {start}-{start+size-1}/{len(data)}',
                        'Content-Length': str(size), 'ETag': etag}
        self.body = io.BytesIO(data[start:start+size])
        self.raw = self
        self.read_count = 0

    def read(self, size, decode_content=False):
        self.read_count += 1
        return self.body.read(size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeSession:
    def __init__(self, data, etag='"version1"', status=206):
        self.data, self.etag, self.status = data, etag, status
        self.requests, self.responses = [], []

    def get(self, url, headers, **kwargs):
        first, last = map(int, headers['Range'].removeprefix('bytes=').split('-'))
        self.requests.append((first, last+1, dict(headers)))
        response = FakeResponse(self.data, first, last-first+1, self.etag, self.status)
        self.responses.append(response)
        return response

    def close(self):
        pass


def fixture():
    stream = io.BytesIO()
    bodies = {'07/101.low.mp3': b'A'*1400, '07/102.low.mp3': b'B'*900, '07/103.low.mp3': b'C'*2300}
    with tarfile.open(fileobj=stream, mode='w', format=tarfile.USTAR_FORMAT) as tar:
        directory = tarfile.TarInfo('07')
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for name, data in bodies.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    return stream.getvalue(), bodies


class RangeTests(unittest.TestCase):
    def setUp(self):
        (BASE/'tmp').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=BASE/'tmp', prefix='jamendo-ranges-test-')
        self.root = Path(self.tmp.name)
        self.data, self.bodies = fixture()
        self.budget = Budget(self.root/'budget.json', 2*1024**2, 100)

    def tearDown(self):
        self.tmp.cleanup()

    def selected(self):
        name = '07/103.low.mp3'
        return [dict(track_id='track_103', low_quality_path=name,
                     expected_audio_sha256=hashlib.sha256(self.bodies[name]).hexdigest())]

    def test_only_selected_body_is_read_and_resume_uses_verified_cache(self):
        session = FakeSession(self.data)
        first = acquire_archive(URL, '07', self.selected(), self.root/'audio', self.root/'index',
                                self.budget, threading.Event(), lambda row: None, session)
        self.assertEqual(first['records'][0]['bytes'], 2300)
        self.assertEqual((self.root/'audio/07/103.low.mp3').read_bytes(), self.bodies['07/103.low.mp3'])
        self.assertFalse((self.root/'audio/07/101.low.mp3').exists())
        # Independent tarfile parser supplies member body intervals.
        with tarfile.open(fileobj=io.BytesIO(self.data)) as tar:
            for name in ('07/101.low.mp3', '07/102.low.mp3'):
                info = tar.getmember(name)
                for start, end, _ in session.requests:
                    self.assertTrue(end <= info.offset_data or start >= info.offset_data + info.size)
        resumed = FakeSession(self.data)
        acquire_archive(URL, '07', self.selected(), self.root/'audio', self.root/'index',
                        self.budget, threading.Event(), lambda row: None, resumed)
        self.assertEqual(resumed.requests, [])

    def test_ignored_range_is_rejected_before_body_read(self):
        session = FakeSession(self.data, status=200)
        client = ArchiveRanges(URL, self.root/'index', self.budget, session)
        with self.assertRaisesRegex(ValueError, 'HTTP206'):
            client.header(0)
        self.assertEqual(session.responses[0].read_count, 0)
        self.assertFalse((self.root/'index/source.json').exists())

    def test_source_drift_on_resume_fails_before_body_read(self):
        ArchiveRanges(URL, self.root/'index', self.budget, FakeSession(self.data)).header(0)
        changed = FakeSession(self.data, etag='"version2"')
        client = ArchiveRanges(URL, self.root/'index', self.budget, changed)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            client.header(512)
        self.assertEqual(changed.responses[0].read_count, 0)
        self.assertEqual(changed.requests[0][2]['If-Match'], '"version1"')

    def test_wrong_range_or_encoding_rejected_without_body(self):
        for key, value in [('Content-Range', 'bytes 0-510/10240'), ('Content-Length', '10240'),
                           ('Content-Encoding', 'gzip'), ('ETag', 'W/"weak"')]:
            response = FakeResponse(self.data, 0, 512)
            response.headers[key] = value
            with self.assertRaises(ValueError):
                checked_headers(response, 0, 512)
            self.assertEqual(response.read_count, 0)

    def test_bad_official_hash_never_creates_audio(self):
        destination = self.root/'audio/07/103.low.mp3'
        with self.assertRaisesRegex(ValueError, 'official SHA256'):
            save_selected(b'wrong', destination, '0'*64)
        self.assertFalse(destination.exists())
        self.assertFalse(destination.with_suffix('.mp3.part').exists())

    def test_links_traversal_and_corrupt_tar_headers_rejected(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('07/103.low.mp3', tarfile.SYMTYPE)]:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.size = 1
            with self.assertRaises(ValueError):
                parse_header(member.tobuf(tarfile.USTAR_FORMAT), '07', 0, len(self.data))
        bad = bytearray(self.data[:512])
        bad[1] ^= 1
        with self.assertRaises(tarfile.InvalidHeaderError):
            parse_header(bytes(bad), '07', 0, len(self.data))

    def test_cached_header_tampering_rejected(self):
        client = ArchiveRanges(URL, self.root/'index', self.budget, FakeSession(self.data))
        client.header(0)
        path = self.root/'index/headers/000000000000.json'
        record = json.loads(path.read_text())
        record['sha256'] = '0'*64
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'hash check'):
            client.header(0)

    def test_budget_reservation_survives_failed_request_and_restart(self):
        budget = Budget(self.root/'small-budget.json', 512, 1)
        session = FakeSession(self.data, status=200)
        with self.assertRaises(ValueError):
            ArchiveRanges(URL, self.root/'index', budget, session).header(0)
        resumed = Budget(self.root/'small-budget.json', 512, 1)
        with self.assertRaisesRegex(ValueError, 'budget exhausted'):
            ArchiveRanges(URL, self.root/'index', resumed, session).header(0)
        self.assertEqual(len(session.requests), 1)

    def test_existing_mismatching_audio_is_preserved(self):
        destination = self.root/'audio/07/103.low.mp3'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b'previous-evidence')
        with self.assertRaisesRegex(ValueError, 'Existing selected file'):
            acquire_archive(URL, '07', self.selected(), self.root/'audio', self.root/'index',
                            self.budget, threading.Event(), lambda row: None, FakeSession(self.data))
        self.assertEqual(destination.read_bytes(), b'previous-evidence')


if __name__ == '__main__':
    unittest.main()
