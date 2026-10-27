"""Persistence/identity contracts; no inference or external worker is launched."""
import unittest
from unittest.mock import patch

from tools.train_next_v2.v3_studio import jobs, store
from support_v3 import StudioFixture


class JobContractTests(unittest.TestCase):
    def setUp(self):
        self.fixture = StudioFixture()
        self.addCleanup(self.fixture.close)

    def register(self):
        return jobs.submit('analyze', {'audio_id': 'fixture-audio', 'device': 'cpu', 'include_v2': False}, start=False)

    def test_registration_records_reproducible_inputs_before_execution(self):
        value = self.register()
        launch = store.read(jobs.directory(value['job_id']) / 'launch.json')
        self.assertEqual(launch['arguments']['audio_sha256'], self.fixture.row['sha256'])
        self.assertEqual(launch['manifest_sha256'], self.fixture.manifest_sha)
        self.assertEqual(launch['resume_argv'], launch['original_argv'] + ['--resume'])
        self.assertEqual(launch['cwd'], str(store.ROOT))
        self.assertEqual(launch['environment']['HF_HUB_OFFLINE'], '1')
        self.assertTrue(launch['code_hashes'])
        self.assertFalse(jobs.status(value['job_id'])['process_alive'])

    def test_pending_project_is_readable_before_worker_starts_without_faking_result(self):
        value=self.register()
        loaded=store.load_project(value['project_id'])
        self.assertIsNone(loaded['result'])
        self.assertEqual(loaded['project']['job_id'],value['job_id'])
        self.assertEqual(loaded['project']['audio']['sha256'],self.fixture.row['sha256'])
        self.assertFalse((store.project_path(value['project_id'])/'project.json').exists())
        self.assertEqual(next(p for p in store.projects() if p['id']==value['project_id'])['status'],'pending')

    def test_pending_metadata_cannot_impersonate_another_audio(self):
        value=self.register(); path=store.project_path(value['project_id'])/'pending.json'
        pending=store.read(path);pending['audio']['path']='another.mp3';store.atomic(path,pending)
        with self.assertRaises(ValueError):store.load_project(value['project_id'])

    def test_running_state_requires_pid_creation_identity(self):
        value = self.register(); directory = jobs.directory(value['job_id'])
        state = {'status': 'running', 'process_identity': {'pid': 987654, 'created': 'old'}}
        store.atomic(directory / 'status.json', state)
        for live in (None, {'pid': 987654, 'created': 'reused'}):
            with self.subTest(live=live), patch.object(store, 'identity', return_value=live):
                actual = jobs.status(value['job_id'])
                self.assertEqual(actual['status'], 'interrupted')
                self.assertFalse(actual['process_alive'])

    def test_verified_running_identity_remains_active(self):
        value = self.register(); ident = {'pid': 987654, 'created': 'matching'}
        store.atomic(jobs.directory(value['job_id']) / 'status.json', {'status': 'running', 'process_identity': ident})
        with patch.object(store, 'identity', return_value=ident):
            state = jobs.status(value['job_id'])
            self.assertTrue(state['process_alive'])
            self.assertEqual(state['status'], 'running')
            with self.assertRaises(ValueError):
                jobs.resume(value['job_id'])

    def test_spawned_but_early_dead_worker_is_recoverable(self):
        value = self.register(); directory = jobs.directory(value['job_id'])
        for ident in (None, {'pid': 987654, 'created': 'old'}):
            with self.subTest(ident=ident):
                store.atomic(directory / 'process.json', ident)
                with patch.object(store, 'identity', return_value=None):
                    state = jobs.status(value['job_id'])
                self.assertEqual(state['status'], 'interrupted')
                self.assertFalse(state['process_alive'])

    def test_cancelled_stale_pid_does_not_receive_cancel_request(self):
        value = self.register(); directory = jobs.directory(value['job_id'])
        store.atomic(directory / 'status.json', {'status': 'running', 'process_identity': {'pid': 987654, 'created': 'old'}})
        with patch.object(store, 'identity', return_value={'pid': 987654, 'created': 'new'}):
            with self.assertRaises(ValueError): jobs.cancel(value['job_id'])
        self.assertFalse((directory / 'cancel.json').exists())

    def test_resume_retains_original_config_and_archives_cancellation_request(self):
        value = self.register(); directory = jobs.directory(value['job_id'])
        original = (directory / 'launch.json').read_bytes()
        store.atomic(directory / 'status.json', {'status': 'cancelled'})
        store.atomic(directory / 'cancel.json', {'requested_at': 'synthetic'})
        with patch.object(jobs, 'spawn') as spawn:
            self.assertEqual(jobs.resume(value['job_id'])['job_id'], value['job_id'])
            self.assertEqual(spawn.call_args.args[1], True)
        self.assertEqual((directory / 'launch.json').read_bytes(), original)
        self.assertFalse((directory / 'cancel.json').exists())
        self.assertEqual(len(list(directory.glob('cancel-*.json'))), 1)

    def test_spawn_failure_never_strands_an_unrecoverable_registered_job(self):
        with patch.object(jobs.subprocess, 'Popen', side_effect=OSError('synthetic process creation failure')):
            with self.assertRaises(OSError):
                jobs.submit('analyze', {'audio_id': 'fixture-audio', 'device': 'cpu'}, start=True)
        rows = jobs.all_jobs()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['status'], 'failed')
        self.assertFalse(rows[0]['process_alive'])

    def test_registered_work_blocks_competing_submit(self):
        self.register()
        with self.assertRaises(RuntimeError):
            self.register()


if __name__ == '__main__':
    unittest.main()
