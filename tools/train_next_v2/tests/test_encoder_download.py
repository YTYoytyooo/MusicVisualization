import unittest
import hashlib
import io
from pathlib import Path
import tempfile

from tools.train_next_v2.common import BASE
from tools.train_next_v2.preparation.fetch_encoder_weights import check_response, checked_destination, download


class EncoderDownloadGuards(unittest.TestCase):
    def test_exact_resume_required(self):
        check_response(206, {'Content-Range': 'bytes 10-99/100', 'Content-Length': '90'}, 10, 100)
        for status, headers in [(200, {'Content-Length': '100'}), (206, {'Content-Range': 'bytes 9-99/100'}), (206, {'Content-Range': 'bytes 10-99/101'})]:
            with self.assertRaises(ValueError):
                check_response(status, headers, 10, 100)

    def test_wrong_size_or_html_is_rejected(self):
        for headers in [{'Content-Length': '101'}, {'Content-Type': 'text/html'}]:
            with self.assertRaises(ValueError):
                check_response(200, headers, 0, 100)

    def test_output_path_cannot_escape(self):
        row = dict(repository='m-a-p/MERT-v1-95M', revision='a'*40, sha256='b'*64, bytes=100,
                   destination='../pytorch_model.bin', url='https://huggingface.co/m-a-p/MERT-v1-95M/resolve/'+'a'*40+'/pytorch_model.bin')
        with self.assertRaises(ValueError):
            checked_destination(Path.cwd() / 'weight-guard-test', row)

    def test_empty_partial_can_finish_and_verified_file_is_reused(self):
        payload = b'fixed public test weight'
        row = dict(name='test', repository='m-a-p/MERT-v1-95M', revision='a'*40,
                   sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                   destination='pytorch_model.bin', url='https://huggingface.co/m-a-p/MERT-v1-95M/resolve/'+'a'*40+'/pytorch_model.bin')
        class Response(io.BytesIO):
            status = 200
            headers = {'Content-Length': str(len(payload))}
            def geturl(self):
                return row['url']
        with tempfile.TemporaryDirectory(dir=BASE / 'tmp') as directory:
            root = Path(directory)
            (root / 'pytorch_model.bin.part').touch()
            result = download(root, row, lambda _: None, lambda *a, **k: Response(payload))
            self.assertEqual(result['status'], 'downloaded_verified')
            self.assertEqual((root / 'pytorch_model.bin').read_bytes(), payload)
            def no_network(*args, **kwargs):
                raise AssertionError('Verified file should be reused')
            self.assertEqual(download(root, row, lambda _: None, no_network)['status'], 'existing_verified')


if __name__ == '__main__':
    unittest.main()
