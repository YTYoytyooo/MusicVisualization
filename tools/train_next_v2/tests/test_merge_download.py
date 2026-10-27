"""Checks for byte ranges and preservation when a MERGE transfer is invalid."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest

from tools.train_next_v2.auxiliary_data.merge_download import check_headers, transfer
from tools.train_next_v2.common import BASE


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers

    def geturl(self):
        return 'https://zenodo.org/example.zip'


class DownloadTests(unittest.TestCase):
    def test_ignored_or_wrong_range_rejected(self):
        for status, value in [(200, ''), (206, 'bytes 0-9/10'), (206, 'bytes 4-9/11')]:
            with self.assertRaises(ValueError):
                check_headers(status, {'Content-Range': value}, 4, 10)
        check_headers(206, {'Content-Range': 'bytes 4-9/10', 'Content-Length': '6'}, 4, 10)

    def test_resume_appends_exact_bytes_and_checks_whole_archive(self):
        with tempfile.TemporaryDirectory(dir=BASE / 'tmp') as folder:
            path = Path(folder) / 'example.zip'
            path.with_name(path.name + '.part').write_bytes(b'abcd')
            item = dict(url='https://zenodo.org/example.zip', size=10, md5=hashlib.md5(b'abcdefghij').hexdigest())
            def open_response(request, timeout):
                self.assertEqual(request.get_header('Range'), 'bytes=4-')
                return Response(b'efghij', 206, {'Content-Range': 'bytes 4-9/10', 'Content-Length': '6'})
            result = transfer(item, path, lambda _: None, resume=True, opener=open_response)
            self.assertEqual(path.read_bytes(), b'abcdefghij')
            self.assertEqual(result['sha256'], hashlib.sha256(b'abcdefghij').hexdigest())

    def test_checksum_failure_preserves_partial(self):
        with tempfile.TemporaryDirectory(dir=BASE / 'tmp') as folder:
            path = Path(folder) / 'example.zip'
            item = dict(url='https://zenodo.org/example.zip', size=4, md5=hashlib.md5(b'good').hexdigest())
            with self.assertRaisesRegex(ValueError, 'MD5 mismatch'):
                transfer(item, path, lambda _: None, opener=lambda *a, **k: Response(b'bad!', 200, {'Content-Length': '4'}))
            self.assertFalse(path.exists())
            self.assertEqual(path.with_name(path.name + '.part').read_bytes(), b'bad!')


if __name__ == '__main__':
    unittest.main()
