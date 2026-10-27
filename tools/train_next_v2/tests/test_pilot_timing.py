"""Physical sample/frame alignment checks for the C encoder preflight."""
import unittest

import numpy as np

from tools.train_next_v2.feature_pilot.encoders import (
    central_indices, clipped_window, conv_geometry, frame_centers, frame_count,
)


class PilotTimingTests(unittest.TestCase):
    def test_mert_one_second_has_74_complete_frames(self):
        kernels = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]
        self.assertEqual(conv_geometry(kernels, strides), (320, 400))
        self.assertEqual(frame_count(24000, kernels, strides), 74)
        times = frame_centers(74, 3.0, 24000, 320, 400)
        self.assertAlmostEqual(times[0], 3 + 199.5 / 24000)
        self.assertAlmostEqual(times[-1], 3 + 23559.5 / 24000)

    def test_source_edges_are_clipped_without_left_padding(self):
        audio = np.arange(72000, dtype=np.float32)
        crop, offset = clipped_window(audio, 24000, 0, 2)
        self.assertEqual(offset, 0)
        np.testing.assert_array_equal(crop, audio[:24000])
        crop, offset = clipped_window(audio, 24000, 2.5, 2)
        self.assertEqual(offset, 36000)
        np.testing.assert_array_equal(crop, audio[36000:])

    def test_pooling_uses_physical_centers_not_stretched_frame_indices(self):
        centers = frame_centers(74, 0.0, 24000, 320, 400)
        selected = central_indices(centers, 0.0)
        np.testing.assert_array_equal(selected, np.arange(19))
        selected = central_indices(np.array([1.25, 1.5, 1.75]), 1.5)
        np.testing.assert_array_equal(selected, [0, 1])

    def test_empty_pool_fails_instead_of_synthesizing_a_frame(self):
        with self.assertRaisesRegex(ValueError, 'No valid central frame'):
            central_indices(np.array([0.0, 0.1]), 2.0)


if __name__ == '__main__':
    unittest.main()
