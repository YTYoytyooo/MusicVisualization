import tempfile
from pathlib import Path
import unittest

import numpy as np

from tools.train_next_v2.common import BASE
from tools.train_next_v2.feature_pilot.cache import read_shard, write_shard


class PilotCacheTests(unittest.TestCase):
    def setUp(self):
        (BASE / 'tmp').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=BASE / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'features.npz'
        self.times = np.array([0., .5, 1.])
        self.values = np.arange(6, dtype=np.float32).reshape(3, 2)

    def test_completed_cache_verifies_hash_and_rejects_wrong_clock(self):
        write_shard(self.path, self.times, self.values, 'key', 'audio', {})
        self.assertIsNotNone(read_shard(self.path, self.times, 'key', 'audio', 2))
        with self.assertRaises(AssertionError):
            read_shard(self.path, self.times + .5, 'key', 'audio', 2)
        with self.path.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            read_shard(self.path, self.times, 'key', 'audio', 2)

    def test_orphan_requires_exact_recomputation(self):
        with self.path.open('wb') as stream:
            np.savez_compressed(stream, times=self.times, features=self.values,
                                feature_key='key', audio_sha256='audio')
        self.assertIsNone(read_shard(self.path, self.times, 'key', 'audio', 2))
        with self.assertRaises(AssertionError):
            write_shard(self.path, self.times, self.values + .1, 'key', 'audio', {})
        write_shard(self.path, self.times, self.values, 'key', 'audio', {})
        self.assertIsNotNone(read_shard(self.path, self.times, 'key', 'audio', 2))

    def test_wrong_source_and_nonfinite_features_are_rejected(self):
        with self.assertRaises(ValueError):
            write_shard(self.path, self.times, self.values * np.nan, 'key', 'audio', {})
        write_shard(self.path, self.times, self.values, 'key', 'audio', {})
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            read_shard(self.path, self.times, 'key', 'different-audio', 2)


if __name__ == '__main__':
    unittest.main()
