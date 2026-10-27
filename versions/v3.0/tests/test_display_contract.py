"""Independent display smoothing oracles and immutable raw exports."""
import copy
import csv
import io
import unittest

from tools.train_next_v2.v3_studio import render


class DisplayContractTests(unittest.TestCase):
    def test_smoothing_known_impulse_uses_time_window_and_preserves_raw(self):
        times = [0., .5, 1.]
        raw = [[0., 0.], [1., -1.], [0., 0.]]
        before = copy.deepcopy(raw)
        smoothed = render.smooth(times, raw, 1.)
        expected = [[.5, -.5], [1 / 3, -1 / 3], [.5, -.5]]
        self.assertEqual(smoothed, expected)
        self.assertEqual(raw, before)
        smoothed[0][0] = 9.
        self.assertEqual(raw, before)

    def test_disabled_smoothing_returns_an_independent_copy(self):
        raw = [[-.2, .3], [.2, -.3]]
        out = render.smooth([0., .5], raw, 0.)
        self.assertEqual(out, raw)
        out[0][0] = 1.
        self.assertEqual(raw[0][0], -.2)

    def test_missing_axes_have_independent_contiguous_segments(self):
        raw = [[0., -1.], [None, 1.], [1., None], [-1., -.5]]
        out = render.smooth([0., .5, 1., 1.5], raw, 2.)
        self.assertEqual(out, [[0., 0.], [None, 0.], [0., None], [0., -.5]])
        self.assertIsNone(raw[1][0])
        self.assertIsNone(raw[2][1])

    def test_time_gap_does_not_join_separate_segments(self):
        out = render.smooth([0., .5, 2., 2.5], [[0., 1.], [0., 1.], [1., -1.], [1., -1.]], 2.)
        self.assertEqual(out, [[0., 1.], [0., 1.], [1., -1.], [1., -1.]])

    def test_non_monotonic_nonfinite_or_mismatched_clock_is_rejected(self):
        for t, values in (([0., 0.], [[0., 0.]] * 2), ([.5, 0.], [[0., 0.]] * 2),
                          ([0., float('nan')], [[0., 0.]] * 2), ([0.], [])):
            with self.subTest(t=t), self.assertRaises(ValueError):
                render.smooth(t, values, 1.)

    def test_unapproved_window_or_invalid_values_are_rejected(self):
        for seconds in (True, -1., .25, float('nan'), float('inf')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                render.smooth([0.], [[0., 0.]], seconds)
        for values in ([[float('nan'), 0.]], [[0., float('inf')]], [[0.]], [[0., 0., 0.]], [[True, 0.]]):
            for seconds in (0., 1.):
                with self.subTest(values=values, seconds=seconds), self.assertRaises(ValueError):
                    render.smooth([0.], values, seconds)

    def test_csv_preserves_each_native_clock_and_raw_values(self):
        value = {'systems': {
            'b01': {'times_seconds': [0., .5], 'raw_va': [[-.25, .75], [.25, -.75]]},
            'v2': {'times_seconds': [0., .1, .2], 'raw_va': [[0., 0.], [.1, -.1], [.2, -.2]]},
        }}
        before = copy.deepcopy(value)
        rows = list(csv.DictReader(io.StringIO(render.csv_text(value).lstrip('\ufeff'))))
        self.assertEqual([float(r['time_seconds']) for r in rows if r['system'] == 'b01'], [0., .5])
        self.assertEqual([float(r['time_seconds']) for r in rows if r['system'] == 'v2'], [0., .1, .2])
        self.assertEqual(float(rows[0]['valence_raw']), -.25)
        self.assertTrue(all(r['display_window_seconds'] == '' for r in rows))
        display = list(csv.DictReader(io.StringIO(render.csv_text(value, 1.).lstrip('\ufeff'))))
        self.assertIn('valence_display', display[0])
        self.assertNotIn('valence_raw', display[0])
        self.assertEqual(value, before)


if __name__ == '__main__':
    unittest.main()
