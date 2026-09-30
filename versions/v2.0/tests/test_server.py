import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from studio.demo import create_demo
from studio.server import make_server, project_payload
from studio.store import atomic_json


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.project = create_demo(cls.root, 2.)
        cls.server = make_server(cls.root, 0, start_queue=False)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, path, method='GET', value=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=json.dumps(value) if value is not None else None,
                               headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def test_bootstrap_project_and_range(self):
        status, data, _ = self.request('/api/bootstrap')
        self.assertEqual(status, 200)
        bootstrap = json.loads(data)
        self.assertTrue(bootstrap['token'])
        self.assertEqual(bootstrap['projects'][0]['id'], self.project.name)
        status, data, _ = self.request('/api/projects/' + self.project.name)
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(data)['timeline']['source_raw']), 20)
        self.assertEqual(json.loads(data)['timeline']['step_us'], 100000)
        status, data, headers = self.request('/media/' + self.project.name + '/audio', headers={'Range': 'bytes=0-15'})
        self.assertEqual(status, 206)
        self.assertEqual(len(data), 16)
        self.assertIn('Content-Range', headers)

    def test_post_requires_token_and_rejects_external_origin(self):
        path = '/api/projects/' + self.project.name + '/preview'
        body = {'base_revision': 'r000000', 'edits': []}
        status, _, _ = self.request(path, 'POST', body, {'Content-Type': 'application/json'})
        self.assertEqual(status, 403)
        token = json.loads(self.request('/api/bootstrap')[1])['token']
        headers = {'Content-Type': 'application/json', 'X-Studio-Token': token}
        self.assertEqual(self.request(path, 'POST', body, headers)[0], 200)
        headers['Origin'] = 'https://evil.invalid'
        self.assertEqual(self.request(path, 'POST', body, headers)[0], 403)

    def test_path_traversal_and_unknown_host_rejected(self):
        self.assertEqual(self.request('/api/projects/..')[0], 400)
        self.assertEqual(self.request('/api/bootstrap', headers={'Host': 'evil.invalid'})[0], 403)

    def test_render_order_is_chronological_not_random_id(self):
        with tempfile.TemporaryDirectory() as root:
            project = create_demo(root, .5)
            for render_id, stamp in [('v-aaaa', '2026-09-20T00:00:00+00:00'),
                                     ('v-zzzz', '2026-09-21T00:00:00+00:00')]:
                atomic_json(project / 'renders' / render_id / 'render.json',
                            dict(id=render_id, created_at=stamp, revision='r000000',
                                 status='succeeded', video_file='output.mp4'))
            self.assertEqual([r['id'] for r in project_payload(project)['renders']], ['v-aaaa', 'v-zzzz'])

    def test_visual_plan_serves_real_cached_states_without_queue(self):
        from studio.pipeline import prepare_visual_plan
        prepare_visual_plan(self.project, 'r000000', [])
        token = json.loads(self.request('/api/bootstrap')[1])['token']
        status, data, _ = self.request('/api/projects/' + self.project.name + '/visual-plan',
                                      'POST', {'base_revision': 'r000000', 'edits': []},
                                      {'Content-Type': 'application/json', 'X-Studio-Token': token})
        payload = json.loads(data)
        self.assertEqual(status, 200)
        self.assertEqual(payload['status'], 'ready')
        self.assertIn('particle_speed', payload['plan']['states'][0])

    def test_visual_save_without_matching_plan_rejected(self):
        from studio.pipeline import prepare_visual_plan
        prepare_visual_plan(self.project, 'r000000', [])
        token = json.loads(self.request('/api/bootstrap')[1])['token']
        edits = [dict(id='emotion', layer='emotion', field='arousal', operation='point_target',
                      time_us=1000000, transition_in_us=500000, transition_out_us=500000, value=.8),
                 dict(id='visual', layer='visual', field='particle_speed', operation='point_target',
                      time_us=1000000, transition_in_us=500000, transition_out_us=500000, value=3.)]
        status, data, _ = self.request('/api/projects/' + self.project.name + '/revisions',
                                      'POST', {'base_revision': 'r000000', 'edits': edits},
                                      {'Content-Type': 'application/json', 'X-Studio-Token': token})
        self.assertEqual(status, 400)
        self.assertIn('规划', json.loads(data)['error'])


if __name__ == '__main__':
    unittest.main()
