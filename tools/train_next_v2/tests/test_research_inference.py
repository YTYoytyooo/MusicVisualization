import unittest

import numpy as np

from tools.train_next.core import collate_songs
from tools.train_next_v2.research_inference.adapter import (
    acoustic_properties, combine_members, standardize, validate_clock,
)


class ResearchInferenceTests(unittest.TestCase):
    def test_normalization_matches_training_collator(self):
        rng = np.random.default_rng(94)
        x = rng.normal(size=(7, 531)).astype(np.float32)
        mean = rng.normal(size=531).astype(np.float32)
        std = rng.uniform(.01, 2, size=531).astype(np.float32)
        reference, _, _, _ = collate_songs([
            dict(x=x, y=np.zeros((7, 2), np.float32), mask=np.zeros((7, 2), bool))
        ], mean, std)
        np.testing.assert_array_equal(standardize(x, mean, std), reference[0].numpy())

    def test_invalid_features_are_rejected(self):
        for x in (np.zeros((2, 530), np.float32), np.full((2, 531), np.nan, np.float32),
                  np.zeros((0, 531), np.float32), np.zeros((2, 531), np.float64)):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(ValueError):
                standardize(x, np.zeros(531, np.float32), np.ones(531, np.float32))

    def test_invalid_scaler_is_rejected_without_refitting(self):
        x = np.zeros((2, 531), np.float32)
        for std in (np.zeros(531), np.full(531, np.inf), np.ones(530)):
            with self.assertRaises(ValueError):
                standardize(x, np.zeros(531, np.float32), std)

    def test_native_half_second_clock(self):
        validate_clock(np.array([0., .5, 1.]), 3, 1.2)
        validate_clock(np.array([0., .5, 1.]), 3, 1.)
        validate_clock(np.array([0.]), 1, .2)

    def test_clock_cannot_be_shifted_stretched_or_truncated(self):
        for t, n, d in (([.1, .6, 1.1], 3, 1.2), ([0., .6, 1.2], 3, 1.2),
                        ([0., .5], 2, 1.2), ([0., .5, 1.], 3, .9),
                        ([0., .5, 1.], 3, float('nan'))):
            with self.subTest(t=t, d=d), self.assertRaises(ValueError):
                validate_clock(np.asarray(t), n, d)

    def test_raw_mean_and_sample_disagreement_keep_units(self):
        members = np.array([[[-.5, -.8]], [[0., -.2]], [[.5, .4]]], np.float32)
        result = combine_members(members)
        np.testing.assert_allclose(result['raw_va'], [[0., -.2]], atol=1e-7)
        np.testing.assert_allclose(result['seed_disagreement_std'], [[.5, .6]], atol=1e-7)
        self.assertEqual(result['calibration']['status'], 'unavailable_failed_audit')
        self.assertIsNone(result['calibration']['intervals'])
        self.assertIsNone(result['display']['values'])
        self.assertIsNone(result['tension']['values'])
        self.assertFalse(result['disagreement_is_calibrated_confidence'])

    def test_invalid_ensemble_is_rejected(self):
        for values in (np.zeros((2, 5, 2)), np.ones((3, 5, 2))*1.01,
                       np.full((3, 5, 2), np.nan), np.zeros((3, 0, 2))):
            with self.assertRaises(ValueError):
                combine_members(values)

    def test_acoustic_units_are_not_emotion_scores(self):
        x = np.zeros((2, 531), np.float32)
        x[:, 512] = np.log(.25 + 1e-7)
        x[:, 514] = np.log1p(1000.)
        values = acoustic_properties(x)
        np.testing.assert_allclose(values['rms_amplitude'], .25, rtol=1e-6)
        np.testing.assert_allclose(values['rms_dbfs'], -12.0411998266, atol=1e-5)
        np.testing.assert_allclose(values['spectral_centroid_hz'], 1000., rtol=1e-6)

    def test_silence_floor_is_explicit_and_finite(self):
        x = np.zeros((1, 531), np.float32)
        x[:, 512] = np.log(1e-7)
        values = acoustic_properties(x)
        self.assertTrue(np.isfinite(values['rms_dbfs']).all())
        self.assertEqual(values['dbfs_floor_amplitude'], 1e-8)
        np.testing.assert_array_equal(values['spectral_centroid_hz'], [0.])

    def test_invalid_acoustic_magnitudes_do_not_become_plausible_outputs(self):
        x = np.zeros((1, 531), np.float32)
        x[:, 514] = -1
        with self.assertRaises(ValueError):
            acoustic_properties(x)


if __name__ == '__main__':
    unittest.main()
