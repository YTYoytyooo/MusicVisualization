import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from tools.train_next_v2.common import process_identity
from tools.train_next_v2.refinement_20261028.observe import observe_process, observe


@unittest.skipUnless(os.name == 'nt', 'Windows process-object lifecycle')
class ObserveChecks(unittest.TestCase):
    def test_live_identity_and_reused_pid_guard(self):
        identity = process_identity()
        self.assertEqual(observe_process(os.getpid(), identity)['state'], 'alive')
        wrong = dict(pid=os.getpid(), creation_filetime='0')
        self.assertEqual(observe_process(os.getpid(), wrong)['state'], 'identity_mismatch')

    def test_retained_exited_process_is_not_alive_even_with_exit259(self):
        for exit_code in (0, 259):
            with self.subTest(exit_code=exit_code):
                child = subprocess.Popen([sys.executable, '-c', f'import sys; sys.exit({exit_code})'],
                                         cwd=Path(__file__).resolve().parents[4],
                                         creationflags=subprocess.CREATE_NO_WINDOW)
                self.assertEqual(child.wait(timeout=10), exit_code)
                # Popen still owns a handle, so the creation identity remains queryable.
                identity = process_identity(child.pid)
                self.assertIsNotNone(identity)
                result = observe_process(child.pid, identity)
                self.assertEqual(result['state'], 'exited')
                self.assertEqual(result['exit_code'], exit_code)
                with patch('tools.train_next_v2.refinement_20261028.observe.actual_status', return_value={
                    'process_identity': identity, 'same_process_alive': True, 'run_lock_held': False,
                    'active': False, 'indeterminate': True}):
                    status = observe('control', 20261003)
                    self.assertFalse(status['same_process_alive'])
                    self.assertFalse(status['active'])
                    self.assertFalse(status['indeterminate'])

    def test_missing_pid_is_inactive(self):
        self.assertEqual(observe_process(0)['state'], 'missing')
        self.assertEqual(observe_process(2147483647)['state'], 'missing')


if __name__ == '__main__':
    unittest.main()
