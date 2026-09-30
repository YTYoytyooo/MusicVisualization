import unittest
import numpy as np
from studio.timeline import validate_edits, emotion_at, apply_edits, VISUAL_BOUNDS, frame_times


def point(**overrides):
    e = dict(id='e1', layer='emotion', field='arousal', operation='point_target',
             time_us=2000000, value=.6, transition_in_us=1000000,
             transition_out_us=1000000)
    e.update(overrides)
    return e


class TimelineTests(unittest.TestCase):
    def test_target_and_boundaries_do_not_mutate_source(self):
        raw = np.zeros((50, 5))
        edits = validate_edits([point()], 5000000)
        values = emotion_at(raw, [0, 1000000, 1500000, 2000000, 2500000, 3000000, 4000000], edits)
        np.testing.assert_allclose(values[:, 1], [0, 0, .3, .6, .3, 0, 0])
        np.testing.assert_array_equal(values[:, 0], 0)
        np.testing.assert_array_equal(raw, 0)

    def test_non_grid_keyframe_is_exact(self):
        edits = validate_edits([point(time_us=1234567, transition_in_us=500000,
                                      transition_out_us=500000)], 5000000)
        self.assertAlmostEqual(emotion_at(np.zeros((50, 5)), [1234567], edits)[0, 1], .6)

    def test_conflict_and_disabled_edits(self):
        with self.assertRaises(ValueError):
            validate_edits([point(), point(id='e2')], 5000000)
        self.assertEqual(len(validate_edits([point(), point(id='e2', enabled=False)], 5000000)), 2)

    def test_invalid_values_and_times(self):
        for change in ({'value': float('nan')}, {'value': 1.1}, {'time_us': -1},
                       {'field': 'energy'}, {'time_us': 4999999}, {'value': True},
                       {'interpolation': 'magic'}, {'enabled': 'yes'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_edits([point(**change)], 5000000)

    def test_touching_closed_intervals_conflict_in_either_order(self):
        first = dict(id='first', layer='emotion', field='arousal',
                     operation='interval_set', start_us=0, end_us=1000000, value=.2)
        second = dict(first, id='second', start_us=1000000, end_us=2000000, value=.8)
        for edits in ([first, second], [second, first]):
            with self.subTest(order=[e['id'] for e in edits]), self.assertRaises(ValueError):
                validate_edits(edits, 5000000)
        self.assertEqual(len(validate_edits([first, dict(second, enabled=False)], 5000000)), 2)

    def test_touching_zero_weight_transition_does_not_conflict(self):
        first = dict(id='first', layer='emotion', field='arousal',
                     operation='interval_set', start_us=0, end_us=1000000,
                     transition_out_us=1000000, value=.2)
        second = dict(id='second', layer='emotion', field='arousal',
                      operation='interval_set', start_us=2000000, end_us=3000000,
                      value=.8)
        for edits in ([first, second], [second, first]):
            with self.subTest(order=[e['id'] for e in edits]):
                checked = validate_edits(edits, 5000000)
                self.assertAlmostEqual(emotion_at(np.zeros((50, 5)), [2000000], checked)[0, 1], .8)

    def test_point_transition_supports_can_touch(self):
        edits = validate_edits([point(), point(id='later', time_us=4000000)], 5000000)
        self.assertEqual(emotion_at(np.zeros((50, 5)), [3000000], edits)[0, 1], 0.)

    def test_offset_preserves_changes_and_rejects_overflow(self):
        e = dict(id='offset', layer='emotion', field='valence', operation='interval_offset',
                 start_us=1000000, end_us=2000000, value=.2)
        edits = validate_edits([e], 5000000)
        raw = np.zeros((50, 5))
        raw[10, 0], raw[11, 0] = .1, .3
        np.testing.assert_allclose(emotion_at(raw, [1000000, 1100000], edits)[:, 0], [.3, .5])
        raw[10, 0] = .9
        with self.assertRaises(ValueError):
            emotion_at(raw, [1000000], edits)

    def test_hue_uses_short_arc(self):
        e = point(layer='visual', field='hue_base', value=10)
        edits = validate_edits([e], 5000000)
        fields = tuple(VISUAL_BOUNDS)
        base = np.array([[350, 60, .8, .5, 200, 2, .3, 20]] * 3, dtype=float)
        values = apply_edits(base, [1000000, 1500000, 2000000], edits, 'visual', fields)
        np.testing.assert_allclose(values[:, 0], [350, 0, 10])

    def test_video_clock_no_accumulated_drift(self):
        ts = frame_times(180000000, 30)
        self.assertEqual(len(ts), 5400)
        self.assertEqual(ts[30], 1000000)
        self.assertEqual(ts[-1], 179966667)


if __name__ == '__main__':
    unittest.main()
