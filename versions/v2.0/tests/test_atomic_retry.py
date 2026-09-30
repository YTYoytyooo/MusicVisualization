import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from studio.store import atomic_json, read_json


class AtomicRetryTests(unittest.TestCase):
    def test_transient_reader_sharing_failure_retries_without_truncation(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'job.json';atomic_json(path,{'state':'old'})
            original=os.replace;calls=[]
            def replace(source,target):
                calls.append(1)
                if len(calls)<3:
                    self.assertEqual(read_json(path),{'state':'old'})
                    raise PermissionError(13,'sharing violation')
                return original(source,target)
            with patch('studio.store.os.replace',side_effect=replace),patch('studio.store.time.sleep'):
                atomic_json(path,{'state':'new'})
            self.assertEqual(len(calls),3)
            self.assertEqual(read_json(path),{'state':'new'})

    def test_persistent_permission_error_is_bounded_and_original_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'job.json';atomic_json(path,{'state':'old'})
            with patch('studio.store.os.replace',side_effect=PermissionError(13,'denied')) as call,patch('studio.store.time.sleep'):
                with self.assertRaises(PermissionError):atomic_json(path,{'state':'new'})
            self.assertLessEqual(call.call_count,8)
            self.assertEqual(read_json(path),{'state':'old'})
            self.assertEqual(list(Path(tmp).glob('.pending-*')),[])
