import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools.train_next import core, runtime


class StatusWriteTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows sharing regression')
    def test_reader_temporarily_blocks_replace(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'status.json'
            core.atomic_json(path, {'stage': 'old'})
            handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
            self.assertNotEqual(handle, ctypes.c_void_p(-1).value)
            timer = threading.Timer(.2, lambda: kernel.CloseHandle(handle))
            timer.start()
            try:
                core.atomic_json(path, {'stage': 'new'})
            finally:
                timer.join()
            self.assertEqual(json.loads(path.read_text())['stage'], 'new')

    def test_progress_write_failure_does_not_abort_training(self):
        with patch.object(core, 'atomic_json', side_effect=PermissionError('blocked')):
            self.assertFalse(core.write_status(Path('unused'), {'stage': 'extracting_features'}))

    def test_run_lock_is_visible_to_another_process_and_released(self):
        probe = ('from pathlib import Path; import sys; '
                 'from tools.train_next import runtime; '
                 'runtime.ROOT = Path(sys.argv[1]); '
                 'print(runtime.is_running("fixture"))')
        with tempfile.TemporaryDirectory() as tmp, patch.object(runtime, 'ROOT', Path(tmp)):
            def active():
                result = subprocess.run([sys.executable, '-c', probe, tmp],
                                        capture_output=True, text=True, check=True, timeout=30)
                return result.stdout.strip()
            with runtime.run_lease('fixture'):
                self.assertEqual(active(), 'True')
                with self.assertRaisesRegex(RuntimeError, 'already active'):
                    with runtime.run_lease('fixture'):
                        self.fail('Concurrent worker acquired the same run')
            self.assertEqual(active(), 'False')


if __name__ == '__main__':
    unittest.main()
