"""HTTP contracts against real files and a queue with worker launch disabled."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from studio.demo import create_demo
from studio.jobs import JobManager
from studio.server import make_server, project_payload
from studio.store import digest, load_project


class UpgradeServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = create_demo(self.root, 2.)
        with patch('studio.jobs.threading.Thread'):
            self.manager = JobManager(self.root)
        with patch('studio.server.JobManager', return_value=self.manager):
            self.server = make_server(self.root, 0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.token = self.request('/api/bootstrap')[1]['token']

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.manager.close()
        self.temp.cleanup()

    def request(self, path, method='GET', value=None, headers=None):
        defaults = {'Content-Type': 'application/json'}
        if hasattr(self, 'token'):
            defaults['X-Studio-Token'] = self.token
        if headers:
            defaults.update(headers)
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=10)
        try:
            connection.request(method, path, json.dumps(value) if value is not None else None, defaults)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def endpoint(self, suffix):
        return '/api/projects/' + self.project.name + '/' + suffix

    def test_visual_plan_consumers_share_release_and_smoothing_is_frozen(self):
        smoothing = {'enabled': True, 'window_seconds': .3}
        body = {'base_revision': 'r000000', 'edits': [], 'consumer_id': 'page-a',
                'purpose': 'temporary', 'smoothing': smoothing}
        with patch('studio.server.timeline_payload', return_value={}) as timeline, \
                patch('studio.server.cached_visual_plan', return_value=None), \
                patch('studio.server.planning_key', return_value='same-key') as key:
            status, first = self.request(self.endpoint('visual-plan'), 'POST', body)
            self.assertEqual(status, 202)
            status, second = self.request(self.endpoint('visual-plan'), 'POST', {**body, 'consumer_id': 'page-b'})
        self.assertEqual(status, 202)
        self.assertEqual(first['job_id'], second['job_id'])
        self.assertEqual(key.call_args.kwargs['smoothing'], smoothing)
        self.assertEqual(timeline.call_args.kwargs['smoothing'], smoothing)
        self.assertEqual(self.manager.list()[0]['payload']['smoothing'], smoothing)
        status, released = self.request('/api/plan-release', 'POST', {'consumer_id': 'page-a'})
        self.assertEqual(status, 200)
        self.assertEqual(released['cancelled'], [])
        status, released = self.request('/api/plan-release', 'POST', {'consumer_id': 'page-b'})
        self.assertEqual(released['cancelled'], [first['job_id']])

    def test_ready_plan_releases_previous_consumer_and_rejects_bad_purpose(self):
        previous = self.manager.request_plan({'project_id': self.project.name}, 'old', 'page-a', 'temporary')
        body = {'base_revision': 'r000000', 'edits': [], 'consumer_id': 'page-a', 'purpose': 'temporary'}
        with patch('studio.server.timeline_payload', return_value={}), \
                patch('studio.server.cached_visual_plan', return_value={'key': 'ready', 'states': []}):
            status, result = self.request(self.endpoint('visual-plan'), 'POST', body)
            self.assertEqual(status, 200)
            self.assertEqual(result['status'], 'ready')
            self.assertEqual(self.request(self.endpoint('visual-plan'), 'POST', {**body, 'purpose': 'invalid'})[0], 400)
        old = next(j for j in self.manager.list() if j['id'] == previous['id'])
        self.assertEqual(old['status'], 'cancelled')

    def test_http_stale_plan_and_release_do_not_cancel_current_candidate(self):
        body = {'base_revision': 'r000000', 'edits': [], 'consumer_id': 'page-a',
                'purpose': 'temporary', 'consumer_seq': 2}
        with patch('studio.server.timeline_payload', return_value={}), \
                patch('studio.server.cached_visual_plan', return_value=None), \
                patch('studio.server.planning_key', return_value='same-key'):
            status, current = self.request(self.endpoint('visual-plan'), 'POST', body)
            self.assertEqual(status, 202)
            status, stale = self.request(self.endpoint('visual-plan'), 'POST', {**body, 'consumer_seq': 1})
        self.assertEqual(status, 200)
        self.assertEqual(stale['status'], 'superseded')
        status, stale = self.request('/api/plan-release', 'POST', {'consumer_id': 'page-a', 'consumer_seq': 1})
        self.assertEqual(status, 200)
        self.assertEqual(stale['status'], 'superseded')
        self.assertEqual(self.manager.list()[0]['id'], current['job_id'])
        self.assertEqual(self.manager.list()[0]['status'], 'queued')

    def test_preview_submits_frozen_snapshot_options_without_advancing_revision(self):
        before = load_project(self.project)['current_revision']
        snapshot = {'id': 's-frozen', 'options': {'start': .2, 'end': 1.5, 'fps': 30,
                                                'width': 640, 'height': 360, 'mode': 'analysis'}}
        smoothing = {'enabled': False}
        body = {'base_revision': before, 'edits': [], 'smoothing': smoothing,
                'edit_id': 'selected-edit', 'width': 320, 'height': 180}
        with patch('studio.pipeline.create_preview_snapshot', return_value=snapshot) as create:
            status, job = self.request(self.endpoint('previews'), 'POST', body)
        self.assertEqual(status, 202)
        self.assertEqual(job['kind'], 'preview')
        self.assertEqual(job['payload']['snapshot_id'], 's-frozen')
        self.assertNotIn('edits', job['payload'])
        self.assertEqual(job['payload']['options']['start'], .2)
        self.assertEqual(job['payload']['options']['end'], 1.5)
        self.assertEqual(job['payload']['options']['width'], 320)
        self.assertEqual(create.call_args.kwargs, {'smoothing': smoothing, 'edit_id': 'selected-edit'})
        self.assertEqual(load_project(self.project)['current_revision'], before)
        self.assertEqual(self.request(self.endpoint('previews'), 'POST', {**body, 'context_seconds': 10})[0], 400)

    def test_preview_and_save_forward_smoothing_and_detail_uses_revision(self):
        smoothing = {'enabled': False}
        body = {'base_revision': 'r000000', 'edits': [], 'smoothing': smoothing}
        with patch('studio.server.validate_preview', return_value={'times': []}) as validate, \
                patch('studio.server.save_revision', return_value={'id': 'r000001'}) as save:
            self.assertEqual(self.request(self.endpoint('preview'), 'POST', body)[0], 200)
            self.assertEqual(self.request(self.endpoint('revisions'), 'POST', body)[0], 201)
        self.assertEqual(validate.call_args.kwargs, {'smoothing': smoothing})
        self.assertEqual(save.call_args.kwargs, {'smoothing': smoothing})
        with patch('studio.server.timeline_payload', return_value={}) as timeline:
            project_payload(self.project, 'r000000')
        self.assertEqual(timeline.call_args.kwargs, {'revision': 'r000000'})

    def test_full_http_smoothing_roundtrip_preserves_raw_and_old_revision(self):
        raw_path = self.project / 'analysis' / 'predictions_raw.npz'
        original_hash = digest(raw_path)
        smoothing = {'enabled': True, 'window_seconds': .5}
        body = {'base_revision': 'r000000', 'edits': [], 'smoothing': smoothing}
        status, preview = self.request(self.endpoint('preview'), 'POST', body)
        self.assertEqual(status, 200, preview)
        self.assertEqual(preview['smoothing'], smoothing)
        self.assertNotEqual(preview['baseline'], preview['raw'])
        status, saved = self.request(self.endpoint('revisions'), 'POST', body)
        self.assertEqual(status, 201, saved)
        self.assertEqual(saved['smoothing'], smoothing)
        status, detail = self.request('/api/projects/' + self.project.name)
        self.assertEqual(status, 200, detail)
        self.assertEqual(detail['revision']['id'], saved['id'])
        self.assertEqual(detail['timeline']['smoothing'], smoothing)
        self.assertEqual(detail['timeline']['effective'], preview['effective'])
        status, old = self.request('/api/projects/' + self.project.name + '?revision=r000000')
        self.assertEqual(status, 200, old)
        self.assertFalse(old['timeline']['smoothing']['enabled'])
        self.assertEqual(old['timeline']['source_raw'], detail['timeline']['source_raw'])
        self.assertEqual(digest(raw_path), original_hash)

    def test_log_and_retry_endpoints_reject_running_retry_and_keep_payload(self):
        payload = {'project_id': self.project.name, 'revision': 'r000000', 'options': {'fps': 30}}
        job = self.manager.submit('render', payload)
        prefix = '/api/jobs/' + job['id']
        self.assertEqual(self.request(prefix + '/retry', 'POST', {})[0], 400)
        self.manager.cancel(job['id'])
        status, retry = self.request(prefix + '/retry', 'POST', {})
        self.assertEqual(status, 202)
        self.assertEqual(retry['payload'], payload)
        self.assertEqual(retry['parent_job_id'], job['id'])
        log_path = self.manager.folder / (job['id'] + '.log')
        log_path.write_text('safe error\npassword=private-value\n', encoding='utf-8')
        status, log = self.request(prefix + '/log?limit=256')
        self.assertEqual(status, 200)
        self.assertEqual(log['job_id'], job['id'])
        self.assertNotIn('private-value', log['text'])
        self.assertEqual(self.request(prefix + '/log?limit=999999')[0], 400)
        self.assertEqual(self.request('/api/jobs/j-unknown/log')[0], 404)

    def test_new_writes_require_token_and_object_body(self):
        self.assertEqual(self.request('/api/plan-release', 'POST', {'consumer_id': 'x'},
                                      {'X-Studio-Token': ''})[0], 403)
        self.assertEqual(self.request('/api/plan-release', 'POST', [1, 2])[0], 400)


if __name__ == '__main__':
    unittest.main()
