"""Deterministic queue contracts; no worker subprocesses are launched."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studio.jobs import JobManager, run_worker
from studio.store import atomic_json, read_json


class UpgradeJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        with patch('studio.jobs.threading.Thread'):
            self.manager = JobManager(self.root)
        self.addCleanup(self.manager.close)
        self.payload = {'project_id': 'p-one', 'base_revision': 'r000000', 'edits': [],
                        'smoothing': {'enabled': False}}

    def status(self, job):
        return read_json(self.manager.path(job['id']))['status']

    def temporary(self, key, consumer='page:temporary'):
        return self.manager.request_plan(self.payload, key, consumer, 'temporary')

    def test_latest_candidate_cancels_only_unreferenced_old_plan(self):
        first = self.temporary('one')
        second = self.temporary('two')
        self.assertNotEqual(first['id'], second['id'])
        self.assertEqual(self.status(first), 'cancelled')
        self.assertEqual(self.status(second), 'queued')
        same = self.temporary('two')
        self.assertEqual(same['id'], second['id'])

    def test_two_consumers_share_and_release_independently(self):
        first = self.temporary('shared', 'page-a')
        same = self.temporary('shared', 'page-b')
        self.assertEqual(first['id'], same['id'])
        self.temporary('new', 'page-a')
        self.assertEqual(self.status(first), 'queued')
        released = self.manager.release_plan('page-b')
        self.assertEqual(released['cancelled'], [first['id']])
        self.assertEqual(self.status(first), 'cancelled')
        self.assertEqual(self.manager.release_plan('page-b')['released'], [])

    def test_late_sequence_cannot_replace_or_cancel_newer_candidate(self):
        current = self.manager.request_plan(self.payload, 'new', 'page-a', 'temporary', 2)
        stale = self.manager.request_plan(self.payload, 'old', 'page-a', 'temporary', 1)
        self.assertEqual(stale['status'], 'superseded')
        self.assertEqual(len(self.manager.list()), 1)
        self.assertEqual(self.status(current), 'queued')
        self.assertEqual(self.manager.plan_refs['consumers']['page-a'], current['id'])
        self.assertEqual(self.manager.request_plan(self.payload, 'new', 'page-a', 'temporary', 2)['id'], current['id'])
        self.assertEqual(self.manager.request_plan(self.payload, 'different', 'page-a', 'temporary', 2)['status'], 'superseded')

    def test_release_tombstone_blocks_inflight_requests_and_survives_restart(self):
        first = self.manager.request_plan(self.payload, 'one', 'page-a', 'temporary', 1)
        self.manager.release_plan('page-a', 3)
        for seq in (1, 2, 3):
            self.assertEqual(self.manager.request_plan(self.payload, 'late', 'page-a', 'temporary', seq)['status'], 'superseded')
        self.assertEqual(self.status(first), 'cancelled')
        with patch('studio.jobs.threading.Thread'):
            restored = JobManager(self.root)
        self.addCleanup(restored.close)
        self.assertEqual(restored.request_plan(self.payload, 'late', 'page-a', 'temporary', 2)['status'], 'superseded')
        newer = restored.request_plan(self.payload, 'new', 'page-a', 'temporary', 4)
        self.assertEqual(newer['status'], 'queued')
        self.assertEqual(restored.release_plan('page-a', 3)['status'], 'superseded')
        self.assertEqual(self.status(newer), 'queued')

    def test_stale_cached_response_cannot_release_current_interest(self):
        current = self.manager.request_plan(self.payload, 'new', 'page-a', 'temporary', 2)
        result = self.manager.accept_cached_plan('old', 'page-a', 'temporary', 1)
        self.assertEqual(result['status'], 'superseded')
        self.assertEqual(self.status(current), 'queued')
        self.assertEqual(self.manager.plan_refs['consumers']['page-a'], current['id'])

    def test_sequence_must_be_nonnegative_integer(self):
        for seq in (-1, True, 1.5, '2'):
            with self.subTest(seq=seq), self.assertRaises(ValueError):
                self.manager.request_plan(self.payload, 'key', 'page-a', 'temporary', seq)

    def test_confirmed_plan_is_pinned_and_shared_with_temporary_consumer(self):
        first = self.temporary('shared')
        confirmed = self.manager.request_plan(self.payload, 'shared', 'page:confirmed', 'confirmed')
        self.assertEqual(first['id'], confirmed['id'])
        self.manager.release_plan('page:confirmed')
        self.manager.release_plan('page:temporary')
        self.assertEqual(self.status(first), 'queued')
        with patch('studio.jobs.threading.Thread'):
            restored = JobManager(self.root)
        self.addCleanup(restored.close)
        restored.release_plan('page:temporary')
        self.assertEqual(self.status(first), 'queued')

    def test_running_auto_cancel_is_derived_without_overwriting_worker_progress(self):
        job = self.temporary('one')
        running = dict(job, status='running', progress=.375, stage='planning')
        atomic_json(self.manager.path(job['id']), running)
        released = self.manager.release_plan('page:temporary')
        self.assertEqual(released['cancelled'], [job['id']])
        self.assertTrue(self.manager.cancel_path(job['id']).is_file())
        self.assertEqual(read_json(self.manager.path(job['id'])), running)
        shown = next(j for j in self.manager.list() if j['id'] == job['id'])
        self.assertEqual(shown['status'], 'cancelling')
        self.assertEqual(shown['progress'], .375)
        terminal = dict(running, status='succeeded')
        atomic_json(self.manager.path(job['id']), terminal)
        self.assertEqual(self.manager.list()[0]['status'], 'succeeded')

    def test_cached_confirmation_pins_worker_before_releasing_temporary_interest(self):
        job = self.temporary('published-key')
        atomic_json(self.manager.path(job['id']), dict(job, status='running'))
        self.manager.accept_cached_plan('published-key', 'page:temporary', 'confirmed')
        self.assertFalse(self.manager.cancel_path(job['id']).exists())
        self.assertEqual(self.status(job), 'running')
        self.assertIn(job['id'], self.manager.plan_refs['pinned'])

    def test_bad_consumer_is_rejected_before_any_job_is_written(self):
        for consumer in (None, '', '../bad', 2, 'x' * 201):
            with self.subTest(consumer=consumer), self.assertRaises(ValueError):
                self.temporary('test', consumer)
        self.assertEqual(self.manager.list(), [])

    def test_retry_copies_original_frozen_payload_and_requires_terminal_state(self):
        payload = {'project_id': 'p-one', 'revision': 'r000002', 'options': {'fps': 30}}
        expected = deepcopy(payload)
        job = self.manager.submit('render', payload)
        payload['revision'] = 'r000099'
        payload['options']['fps'] = 60
        with self.assertRaises(ValueError):
            self.manager.retry(job['id'])
        atomic_json(self.manager.path(job['id']), dict(job, status='failed'))
        retry = self.manager.retry(job['id'])
        self.assertNotEqual(retry['id'], job['id'])
        self.assertEqual(retry['parent_job_id'], job['id'])
        self.assertEqual(retry['payload'], expected)
        self.assertEqual(read_json(self.manager.path(job['id']))['status'], 'failed')

    def test_preview_retry_preserves_snapshot_id(self):
        original = self.manager.submit('preview', {'project_id': 'p-one', 'snapshot_id': 's-frozen',
                                                   'options': {'start': 3., 'end': 5.}})
        self.manager.cancel(original['id'])
        retry = self.manager.retry(original['id'])
        self.assertEqual(retry['kind'], 'preview')
        self.assertEqual(retry['payload'], original['payload'])

    def test_logs_are_bounded_sanitized_and_reject_path_input(self):
        job = self.manager.submit('render', {})
        path = self.manager.folder / (job['id'] + '.log')
        path.write_text('old\n' * 100 + '\x1b[31mERROR\x1b[0m\x00\n'
                        'Authorization: Bearer very-secret\napi_key=another-secret\n', encoding='utf-8')
        result = self.manager.log(job['id'], 256)
        self.assertEqual(result['job_id'], job['id'])
        self.assertTrue(result['truncated'])
        self.assertIn('ERROR', result['text'])
        self.assertNotIn('\x1b', result['text'])
        self.assertNotIn('\x00', result['text'])
        self.assertNotIn('very-secret', result['text'])
        self.assertNotIn('another-secret', result['text'])
        for limit in (0, 65537, True):
            with self.assertRaises(ValueError):
                self.manager.log(job['id'], limit)
        with self.assertRaises(ValueError):
            self.manager.log('../outside')
        with self.assertRaises(FileNotFoundError):
            self.manager.log('j-unknown')
        path.write_text('password=' + 'private' * 1000 + '\nsafe ending\n', encoding='utf-8')
        result = self.manager.log(job['id'], 64)
        self.assertNotIn('private', result['text'])
        self.assertIn('safe ending', result['text'])

    def test_worker_routes_preview_snapshot_without_loading_current_revision(self):
        job = self.manager.submit('preview', {'project_id': 'p-one', 'snapshot_id': 's-frozen',
                                             'options': {'start': 3., 'end': 5.}})
        (self.manager.folder / (job['id'] + '.started')).touch()
        fake_project = self.root / 'p-one'
        with patch('studio.jobs.resolve_project', return_value=fake_project), \
                patch('studio.pipeline.render_preview', return_value={'id': 'v-preview'}) as render:
            self.assertEqual(run_worker(self.root, job['id']), 0)
        self.assertEqual(render.call_args.args[:3], (fake_project, 's-frozen', job['payload']['options']))
        self.assertEqual(read_json(self.manager.path(job['id']))['result']['id'], 'v-preview')

    def test_worker_progress_accepts_details_without_changing_frozen_payload(self):
        job = self.manager.submit('render', {'project_id': 'p-one', 'revision': 'r000002'})
        (self.manager.folder / (job['id'] + '.started')).touch()
        details = {'done': 3, 'total': 10, 'unit': '帧'}

        def fake_render(project, revision, options, progress, cancel):
            self.assertIs(progress.accepts_details, True)
            progress('rendering', .3, details)
            saved = read_json(self.manager.path(job['id']))
            self.assertEqual(saved['details'], details)
            self.assertEqual(saved['payload'], job['payload'])
            progress('encoding', 0.)
            self.assertNotIn('details', read_json(self.manager.path(job['id'])))
            return {'id': 'v-result'}

        with patch('studio.jobs.resolve_project', return_value=self.root / 'p-one'), \
                patch('studio.pipeline.render', side_effect=fake_render):
            self.assertEqual(run_worker(self.root, job['id']), 0)


if __name__ == '__main__':
    unittest.main()
