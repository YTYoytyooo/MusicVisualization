"""Real loopback HTTP boundary with isolated files and no heavy job execution."""
import http.client
import json
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlencode

from tools.train_next_v2.v3_studio import server, store
from support_v3 import StudioFixture


class HTTPContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = StudioFixture()
        self.httpd = server.make_server(port=0)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup_server)
        self.port = self.httpd.server_port
        status, body, _ = self.request('/api/state')
        self.assertEqual(status, 200)
        self.token = json.loads(body)['token']

    def cleanup_server(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)
        self.fixture.close()

    def request(self, path, method='GET', value=None, headers=None, raw=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.httpd.server_port, timeout=5)
        try:
            body = raw if raw is not None else json.dumps(value) if value is not None else None
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def mutation_headers(self, **extra):
        return {'Content-Type': 'application/json', 'X-V3-Token': self.token,
                'Origin': f'http://127.0.0.1:{self.port}', **extra}

    def test_loopback_state_and_project_identity(self):
        self.assertEqual(self.httpd.server_address[0], '127.0.0.1')
        status, body, headers = self.request('/api/state')
        payload = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(payload['library'][0]['id'], 'fixture-audio')
        self.assertIn('未通过', payload['model']['description'])
        self.assertIn("frame-ancestors 'none'", headers['Content-Security-Policy'])
        status, body, _ = self.request('/api/project?' + urlencode({'id': self.fixture.project_id}))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['result']['manifest_sha256'], self.fixture.manifest_sha)

    def test_foreign_host_and_origin_are_rejected_before_disclosure(self):
        for headers in ({'Host': 'attacker.invalid'}, {'Origin': 'https://attacker.invalid'},
                        {'Host': f'127.0.0.1:{self.port}.attacker.invalid'}):
            with self.subTest(headers=headers):
                status, body, _ = self.request('/api/state', headers=headers)
                self.assertEqual(status, 403)
                self.assertNotIn(self.token.encode(), body)

    def test_mutation_requires_token_and_same_origin_without_job_creation(self):
        body = {'audio_id': 'fixture-audio'}
        with patch.object(server.jobs, 'submit') as submit:
            for headers in ({'Content-Type': 'application/json'},
                            self.mutation_headers(**{'X-V3-Token': 'wrong'}),
                            self.mutation_headers(Origin='https://attacker.invalid')):
                with self.subTest(headers=headers):
                    self.assertEqual(self.request('/api/analyze', 'POST', body, headers)[0], 403)
            submit.assert_not_called()

    def test_authenticated_request_routes_without_starting_real_job(self):
        expected = {'job_id': 'j-contract', 'project_id': 'p-contract'}
        with patch.object(server.jobs, 'submit', return_value=expected) as submit:
            status, body, _ = self.request('/api/analyze', 'POST',
                {'audio_id': 'fixture-audio', 'device': 'cpu', 'include_v2': False}, self.mutation_headers())
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(body), expected)
            self.assertEqual(submit.call_args.args[0], 'analyze')
            self.assertEqual(submit.call_args.args[1]['audio_id'], 'fixture-audio')

    def test_malformed_body_content_type_or_size_is_rejected_before_job(self):
        cases = [('{not json', self.mutation_headers()),
                 ('[]', self.mutation_headers()),
                 ('{"audio_id":"fixture-audio","device":"cpu"}', self.mutation_headers(**{'Content-Type': 'text/plain'})),
                 ('{}', self.mutation_headers(**{'Content-Length': '65537'}))]
        with patch.object(server.jobs, 'submit') as submit:
            for raw, headers in cases:
                with self.subTest(raw=raw, headers=headers):
                    status, _, _ = self.request('/api/analyze', 'POST', headers=headers, raw=raw)
                    self.assertEqual(status, 400)
            submit.assert_not_called()

    def test_project_and_download_traversal_are_rejected(self):
        for project in ('..', '../outside', '..\\outside', '/absolute', 'C:\\Windows'):
            with self.subTest(project=project):
                self.assertEqual(self.request('/api/project?' + urlencode({'id': project}))[0], 400)
        query = urlencode({'id': self.fixture.project_id, 'file': '../project.json'})
        self.assertEqual(self.request('/download?' + query)[0], 400)
        self.assertEqual(self.request('/%2e%2e/project.json')[0], 404)

    def test_audio_ranges_return_actual_source_bytes_and_head_length(self):
        data = self.fixture.audio.read_bytes()
        path = '/audio?' + urlencode({'id': self.fixture.project_id})
        for requested, expected, bounds in (
                ('bytes=4-19', data[4:20], f'bytes 4-19/{len(data)}'),
                ('bytes=-16', data[-16:], f'bytes {len(data)-16}-{len(data)-1}/{len(data)}'),
                (f'bytes={len(data)-8}-', data[-8:], f'bytes {len(data)-8}-{len(data)-1}/{len(data)}')):
            with self.subTest(requested=requested):
                status, body, headers = self.request(path, headers={'Range': requested})
                self.assertEqual(status, 206)
                self.assertEqual(body, expected)
                self.assertEqual(headers['Content-Range'], bounds)
                self.assertEqual(int(headers['Content-Length']), len(expected))
        status, body, headers = self.request(path, method='HEAD')
        self.assertEqual(status, 200)
        self.assertEqual(body, b'')
        self.assertEqual(int(headers['Content-Length']), len(data))

    def test_unsatisfiable_ranges_return_416(self):
        path = '/audio?' + urlencode({'id': self.fixture.project_id})
        size = self.fixture.audio.stat().st_size
        for requested in ('bytes=9-2', f'bytes={size}-', 'bytes=-0', 'bytes=0-1,3-4', 'items=0-3'):
            with self.subTest(requested=requested):
                status, _, headers = self.request(path, headers={'Range': requested})
                self.assertEqual(status, 416)
                self.assertEqual(headers['Content-Range'], f'bytes */{size}')

    def test_invalid_export_settings_never_occupy_job_slot(self):
        base = {'project_id': self.fixture.project_id, 'start': 0., 'end': 1., 'smoothing_seconds': 0.}
        cases = ({'start': True}, {'end': float('nan')}, {'end': 3.}, {'start': 1., 'end': 1.},
                 {'smoothing_seconds': True}, {'smoothing_seconds': .25})
        with patch.object(server.jobs, 'submit') as submit:
            for change in cases:
                with self.subTest(change=change):
                    status, _, _ = self.request('/api/export', 'POST', {**base, **change}, self.mutation_headers())
                    self.assertEqual(status, 400)
            submit.assert_not_called()

    def test_download_does_not_follow_forged_export_directory(self):
        f = self.fixture
        outside = f.root / 'preview.mp4'; outside.write_bytes(b'not a video, path-security fixture')
        store.atomic(f.project_dir / 'exports/latest.json', {
            'directory': f.root.relative_to(store.ROOT).as_posix(), 'files': {'preview.mp4': store.digest(outside)}})
        query = urlencode({'id': f.project_id, 'file': 'preview.mp4'})
        status, body, _ = self.request('/download?' + query)
        self.assertEqual(status, 400)
        self.assertNotEqual(body, outside.read_bytes())


if __name__ == '__main__':
    unittest.main()
