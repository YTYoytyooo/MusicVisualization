"""Comparison contracts: clocks, missing labels and unchanged legacy windows."""
import unittest

import numpy as np
import torch

from tools.train_next_v2.cross_version.adapters import (
    common_clock, legacy_windows, load_legacy_source, V2Head,
)


class CrossVersionTests(unittest.TestCase):
    def test_common_clock_excludes_unsupported_tail_for_all_models(self):
        times = np.arange(5, dtype=float) * .5
        legacy = np.arange(20, dtype=float) * .1
        mask = np.array([[1, 1], [0, 1], [1, 0], [1, 1], [1, 1]], bool)
        original = mask.copy()
        indices, shared = common_clock(times, legacy, mask)
        np.testing.assert_array_equal(indices, [0, 5, 10, 15, -1])
        np.testing.assert_array_equal(shared, [[1, 1], [0, 1], [1, 0], [1, 1], [0, 0]])
        np.testing.assert_array_equal(mask, original)

    def test_no_interpolation_or_time_shift(self):
        for legacy in (np.arange(20)*.1+.02, np.arange(20)*.1001, np.array([0., .1, .3])):
            with self.subTest(legacy=legacy):
                with self.assertRaises(ValueError):
                    common_clock(np.arange(3)*.5, legacy, np.ones((3, 2), bool))

    def test_invalid_clock_or_masks_rejected(self):
        for times, mask in (([0., 1.], np.ones((2,2), bool)),
                            ([0., .5], np.ones((2,2), float)),
                            ([0., .5], np.ones((2,1), bool))):
            with self.assertRaises(ValueError):
                common_clock(times, np.arange(10)*.1, mask)

    def test_legacy_left_zero_padding_and_last_frame_alignment(self):
        x = np.repeat(np.arange(1, 13, dtype=np.float32)[:, None], 512, axis=1)
        out = legacy_windows(x, 0, 12)
        np.testing.assert_array_equal(out[0, :, 0], [0]*9+[1])
        np.testing.assert_array_equal(out[8, :, 0], [0]+list(range(1,10)))
        np.testing.assert_array_equal(out[11, :, 0], list(range(3,13)))

    def test_legacy_window_boundaries_do_not_reset_at_batch(self):
        x = np.random.default_rng(17).normal(size=(23,512)).astype(np.float32)
        np.testing.assert_array_equal(legacy_windows(x, 8, 16), legacy_windows(x, 0, 23)[8:16])

    def test_bad_legacy_embeddings_rejected(self):
        for x in (np.zeros((0,512), np.float32), np.zeros((3,531), np.float32),
                  np.full((3,512), np.nan, np.float32), np.zeros((3,512), np.float64)):
            with self.assertRaises(ValueError):
                legacy_windows(x, 0, len(x))

    def test_cpu_head_matches_actual_v2_interface(self):
        _, source = load_legacy_source()
        torch.set_num_threads(2); torch.manual_seed(73)
        model = source.EmotionModel().eval()
        x = np.random.default_rng(23).normal(size=(23,512)).astype(np.float32)
        expected = np.stack([row['emotion'] for row in source.EmotionInterface(model).predict_sequence(x, batch_size=8)])
        actual = V2Head(model, 'cpu').predict(x, batch_size=8)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (23,5))


if __name__ == '__main__':
    unittest.main()
