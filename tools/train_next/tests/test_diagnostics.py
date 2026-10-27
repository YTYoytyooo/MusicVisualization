import unittest

import numpy as np

from tools.train_next.diagnostics import axis_diagnostics, lag_profile, select_representatives


class DiagnosticTests(unittest.TestCase):
    def test_positive_lag_means_prediction_is_late(self):
        times = np.arange(0, 120, .5)
        signal = lambda t: .35 * np.sin(t * .37) + .2 * np.sin(t * .73)
        truth = signal(times)
        for shift in (-2., 2.):
            result = lag_profile(times, truth, signal(times - shift), np.ones(len(times), bool))
            self.assertEqual(result['best_seconds'], shift)
            self.assertAlmostEqual(result['best_correlation'], 1.)
            self.assertTrue(result['credible'])

    def test_native_one_hz_mask_and_missing_gap_are_not_interpolated(self):
        times = np.arange(0, 100, .5)
        valid = (times % 1 == 0) & ((times < 30) | (times > 50))
        truth = .4 * np.sin(times * .37)
        prediction = .4 * np.sin((times - 3) * .37)
        result = lag_profile(times, truth, prediction, valid)
        self.assertEqual(result['best_seconds'], 3.)
        np.testing.assert_allclose(np.diff(result['search_seconds']), 1.)
        changed = truth.copy()
        changed[~valid] = 999
        self.assertEqual(result, lag_profile(times, changed, prediction, valid))

    def test_constant_truth_does_not_claim_a_delay(self):
        times = np.arange(0, 60, .5)
        result = lag_profile(times, np.zeros(len(times)), np.sin(times), np.ones(len(times), bool))
        self.assertIsNone(result['best_seconds'])
        self.assertFalse(result['credible'])

    def test_mean_and_amplitude_only_use_joint_observed_positions(self):
        times = np.arange(8.)
        truth = np.array([-1., 99., 0., 99., 1., 99., 0., 99.])
        prediction = np.array([-.4, -99., .1, -99., .6, -99., .1, -99.])
        valid = np.array([True, False] * 4)
        result = axis_diagnostics(times, truth, prediction, valid)
        self.assertEqual(result['values'], 4)
        self.assertAlmostEqual(result['truth_mean'], 0.)
        self.assertAlmostEqual(result['prediction_mean'], .1)
        self.assertAlmostEqual(result['std_ratio'], .5)
        self.assertAlmostEqual(result['span90_ratio'], .5)

    def test_selection_is_deterministic_and_contains_six_distinct_strata(self):
        rows = [dict(song_id=f'x:{i:02}', duration_seconds=i + 20,
                     valence=dict(truth_std=i / 100, ccc=i / 20),
                     arousal=dict(truth_std=i / 100, ccc=i / 20)) for i in range(20)]
        result = select_representatives(rows)
        self.assertEqual(result, select_representatives(rows[::-1]))
        self.assertEqual(len({r['song_id'] for r in result}), 6)
        self.assertEqual(result[-1]['category'], 'long')
        self.assertEqual(result[-1]['song_id'], 'x:19')


if __name__ == '__main__':
    unittest.main()
