from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from studio.jobs import recover_interrupted
from studio.store import atomic_json, read_json


class JobRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'j-test.json'
        self.snapshot = {'id': 'j-test', 'status': 'running', 'pid': 123,
                         'process_identity': 'worker-created', 'progress': .5}
        atomic_json(self.path, self.snapshot)

    def test_worker_finishing_during_identity_check_keeps_result(self):
        completed = dict(self.snapshot, status='succeeded', progress=1.,
                         result={'project_id': 'p-completed'}, completed_at='finished')

        def finish_and_exit(pid):
            self.assertEqual(pid, 123)
            atomic_json(self.path, completed)
            return None

        with patch('studio.jobs.process_identity', side_effect=finish_and_exit):
            result = recover_interrupted(self.snapshot, self.path)
        self.assertEqual(result, completed)
        self.assertEqual(read_json(self.path), completed)

    def test_all_terminal_states_survive_stale_snapshot(self):
        for status in ('succeeded', 'failed', 'cancelled', 'interrupted'):
            with self.subTest(status=status):
                terminal = dict(self.snapshot, status=status, error='saved detail')
                atomic_json(self.path, terminal)
                with patch('studio.jobs.process_identity', return_value=None):
                    self.assertEqual(recover_interrupted(self.snapshot, self.path), terminal)
                self.assertEqual(read_json(self.path), terminal)

    def test_dead_worker_retains_latest_progress_when_marked_interrupted(self):
        atomic_json(self.path, dict(self.snapshot, progress=.9, stage='encoding'))
        with patch('studio.jobs.process_identity', return_value=None):
            result = recover_interrupted(self.snapshot, self.path)
        self.assertEqual(result['status'], 'interrupted')
        self.assertEqual(result['progress'], .9)
        self.assertEqual(result['stage'], 'encoding')
        self.assertIn('completed_at', result)
        self.assertEqual(read_json(self.path), result)

    def test_live_worker_and_newer_owner_are_not_overwritten(self):
        with patch('studio.jobs.process_identity', return_value='worker-created'):
            self.assertEqual(recover_interrupted(self.snapshot, self.path), self.snapshot)
        newer = dict(self.snapshot, pid=456, process_identity='new-worker')
        atomic_json(self.path, newer)
        with patch('studio.jobs.process_identity', return_value=None):
            self.assertEqual(recover_interrupted(self.snapshot, self.path), newer)
        self.assertEqual(read_json(self.path), newer)


if __name__ == '__main__':
    unittest.main()
