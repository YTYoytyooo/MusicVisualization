import unittest

import librosa
import numpy as np

from tools.train_next_v2.feature_pilot.resampling import resampled_length, sample_clock_metadata


class ResamplingLengthTests(unittest.TestCase):
    def test_exact_integer_boundary_matches_pinned_real_resampler(self):
        # The failing PMEmo recording contains exactly this many 44.1k samples.
        samples = np.zeros(2427264, dtype=np.float32)
        for rate, expected in ((48000, 2641921), (24000, 1320961)):
            actual = librosa.resample(samples, orig_sr=44100, target_sr=rate, res_type='soxr_hq')
            self.assertEqual(len(actual), expected)
            self.assertEqual(resampled_length(len(samples), 44100, rate), expected)
            clock = sample_clock_metadata(len(samples), 44100, rate, len(actual))
            self.assertEqual(clock['implementation_rounding_samples'], 1)
            self.assertAlmostEqual(clock['source_duration_seconds'], 55.04)
            with self.assertRaises(ValueError):
                sample_clock_metadata(len(samples), 44100, rate, len(actual)-1)

    def test_regular_and_fractional_lengths_keep_exact_checks(self):
        self.assertEqual(resampled_length(44100, 44100, 24000), 24001)
        self.assertEqual(resampled_length(22051, 44100, 24000), 12001)
        with self.assertRaises(ValueError):
            resampled_length(0, 44100, 24000)


if __name__ == '__main__':
    unittest.main()
