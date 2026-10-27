"""Actual small NPZ reads exercise the frozen-cache branch, without extraction."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import numpy as np

from tools.train_next_v2.v3_inference import pipeline
from tools.train_next_v2.v3_studio import store
from support_v3 import StudioFixture


class HistoricalFeatureCacheTests(unittest.TestCase):
    def setUp(self):
        self.fixture = StudioFixture()
        self.addCleanup(self.fixture.close)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        f = self.fixture
        self.cache = f.root / 'synthetic-features.npz'
        self.protocol = f.root / 'comparison.json'
        self.b_protocol = f.root / 'b-protocol.json'
        store.atomic(self.b_protocol, {'feature_hashes': {}})
        self.stack.enter_context(patch.object(pipeline, 'COMPARISON', self.protocol))
        self.stack.enter_context(patch.object(pipeline, 'B_PROTOCOL', self.b_protocol))
        self.manifest = {'feature_key': 'synthetic-feature-contract'}
        self.audio = {**f.row, 'frames': 16000, 'channels': 1}
        self.write_cache()

    def write_cache(self, times=None, features=None, key='synthetic-feature-contract'):
        np.savez(self.cache, times=np.array([0., .5, 1., 1.5, 2.] if times is None else times, dtype=np.float64),
                 features=np.zeros((5, 531), dtype=np.float32) if features is None else features,
                 feature_key=np.array(key))
        relative = self.cache.relative_to(store.ROOT).as_posix()
        store.atomic(self.protocol, {'validation_songs': [{'audio_sha256': self.audio['sha256'],
                                                         'sequence_cache': relative}],
                                    'pinned_hashes': {relative: store.digest(self.cache)}})

    def read_cache(self):
        return pipeline._features(self.audio, self.manifest, 'cpu', None, None)

    def test_valid_cache_returns_original_unmodified_times_features_and_receipt(self):
        before = self.cache.read_bytes()
        times, features, receipt = self.read_cache()
        np.testing.assert_array_equal(times, [0., .5, 1., 1.5, 2.])
        self.assertEqual(features.shape, (5, 531))
        self.assertEqual(features.dtype, np.float32)
        self.assertTrue(receipt['reused'])
        self.assertEqual(receipt['sha256'], store.digest(self.cache))
        self.assertEqual(self.cache.read_bytes(), before)

    def test_byte_tampered_historical_cache_is_rejected_before_numpy_use(self):
        with self.cache.open('ab') as f: f.write(b'unregistered cache mutation')
        with self.assertRaises(ValueError): self.read_cache()

    def test_wrong_feature_contract_is_rejected_even_with_matching_container_hash(self):
        self.write_cache(key='different-feature-contract')
        with self.assertRaises(ValueError): self.read_cache()

    def test_shifted_clock_or_wrong_shape_or_dtype_is_rejected(self):
        cases = ({'times': [.1, .6, 1.1, 1.6, 2.1]},
                 {'features': np.zeros((5, 530), dtype=np.float32)},
                 {'features': np.zeros((5, 531), dtype=np.float64)},
                 {'features': np.full((5, 531), np.nan, dtype=np.float32)})
        for kwargs in cases:
            with self.subTest(kwargs=tuple(kwargs)):
                self.write_cache(**kwargs)
                with self.assertRaises(ValueError): self.read_cache()


if __name__ == '__main__':
    unittest.main()
