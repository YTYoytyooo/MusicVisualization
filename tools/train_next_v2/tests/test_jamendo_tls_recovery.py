from pathlib import Path
import tempfile
import unittest
import requests

from tools.train_next_v2.common import BASE
from tools.train_next_v2.jamendo_audio.ranges import Budget
from tools.train_next_v2.jamendo_recovery.transport import BoundedTLSSession


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=BASE/'tmp',prefix='jamendo-tls-test-')
        self.budget=Budget(Path(self.tmp.name)/'budget.json',4096,8)
        self.budget.reserve(512)
        self.calls=[];self.logs=[];self.sleeps=[]

    def tearDown(self):self.tmp.cleanup()

    def session(self,outcomes):
        sequence=iter(outcomes);calls=self.calls
        class Client:
            def get(self,url,**kwargs):
                calls.append(kwargs);outcome=next(sequence)
                if isinstance(outcome,Exception):raise outcome
                return outcome
            def close(self):pass
        return BoundedTLSSession(self.budget,self.logs.append,factory=Client,sleep=self.sleeps.append)

    def test_observed_eof_retries_are_bounded_and_reserved(self):
        error=requests.exceptions.SSLError('UNEXPECTED_EOF_WHILE_READING')
        token=object();client=self.session([error,error,token])
        result=client.get('https://cdn.freesound.org/file',headers={'Range':'bytes=0-511'},stream=True)
        self.assertIs(result,token);self.assertEqual(len(self.calls),3)
        self.assertEqual(self.budget.state,dict(bytes_reserved=1536,requests_reserved=3))
        self.assertEqual(self.sleeps,[2,5])

    def test_certificate_error_is_never_retried(self):
        client=self.session([requests.exceptions.SSLError('CERTIFICATE_VERIFY_FAILED')])
        with self.assertRaises(requests.exceptions.SSLError):
            client.get('https://cdn.freesound.org/file',headers={'Range':'bytes=0-511'})
        self.assertEqual(len(self.calls),1);self.assertEqual(self.sleeps,[])

    def test_three_eofs_stop_without_a_fourth_attempt(self):
        client=self.session([requests.exceptions.SSLError('UNEXPECTED_EOF_WHILE_READING')]*3)
        with self.assertRaises(requests.exceptions.SSLError):
            client.get('https://cdn.freesound.org/file',headers={'Range':'bytes=0-511'})
        self.assertEqual(len(self.calls),3)

    def test_non_tls_failures_and_http_responses_are_not_hidden(self):
        client=self.session([requests.exceptions.ReadTimeout('timeout')])
        with self.assertRaises(requests.exceptions.ReadTimeout):
            client.get('https://cdn.freesound.org/file',headers={'Range':'bytes=0-511'})
        self.assertEqual(len(self.calls),1)
        client=self.session([{'status_code':200}])
        self.assertEqual(client.get('https://cdn.freesound.org/file',headers={'Range':'bytes=0-511'}),{'status_code':200})
        # ArchiveRanges remains responsible for rejecting this response before reading its body.

    def test_tls_disable_request_rejected_without_network(self):
        client=self.session([])
        with self.assertRaises(ValueError):
            client.get('https://cdn.freesound.org/file',headers={'Range':'bytes=0-511'},verify=False)
        self.assertEqual(self.calls,[])


if __name__=='__main__':unittest.main()
