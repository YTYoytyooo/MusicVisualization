"""Shared known-answer vectors for Python and browser transition evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest

import numpy as np

from studio.timeline import (EMOTIONS, apply_edits, emotion_at, preview_times,
                             validate_edits, validation_times, visual_validation_times, weight_at)

VECTORS = json.loads((Path(__file__).with_name('transition_vectors.json')).read_text(encoding='utf-8'))


class TransitionMathTests(unittest.TestCase):
    def edit(self, patch):
        return validate_edits([{**VECTORS['base_edit'], **patch}], VECTORS['duration_us'])[0]

    def test_shared_weight_vectors(self):
        for case in VECTORS['weight_cases']:
            with self.subTest(case=case['name']):
                np.testing.assert_allclose(weight_at(self.edit(case['patch']), case['times_us']),
                                           case['expected'], rtol=0, atol=1e-12)

    def test_shared_apply_vectors_preserve_source_and_other_fields(self):
        for case in VECTORS['apply_cases']:
            with self.subTest(case=case['name']):
                base = np.zeros((len(case['times_us']), 5))
                base[:, 0] = .123
                base[:, 1] = case['base_arousal']
                snapshot = base.copy()
                result = apply_edits(base, case['times_us'], [self.edit(case['patch'])], 'emotion', EMOTIONS)
                np.testing.assert_array_equal(base, snapshot)
                np.testing.assert_allclose(result[:, 1], case['expected'], atol=1e-12)
                np.testing.assert_array_equal(result[:, 0], snapshot[:, 0])

    def test_old_records_default_to_linear_without_mutation(self):
        original = deepcopy(VECTORS['base_edit'])
        self.assertEqual(validate_edits([original], 1000000)[0]['interpolation'], 'linear')
        self.assertNotIn('interpolation', original)

    def test_invalid_shared_vectors(self):
        for patch in VECTORS['invalid_patches']:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                self.edit(patch)

    def test_overlaps_and_touching_endpoint_remain_rejected(self):
        a = {**VECTORS['base_edit'], 'interpolation': 'smootherstep'}
        b = {**a, 'id': 'other'}
        with self.assertRaises(ValueError):
            validate_edits([a, b], 1000000)
        b['enabled'] = False
        self.assertEqual(len(validate_edits([a, b], 1000000)), 2)
        a.update(start_us=0, end_us=500000, transition_in_us=0, transition_out_us=0)
        b = {**a, 'id': 'touch', 'start_us': 500000, 'end_us': 1000000}
        with self.assertRaises(ValueError):
            validate_edits([a, b], 1000000)

    def test_preview_preserves_jumps_non_grid_target_and_local_curve_samples(self):
        e = self.edit(VECTORS['weight_cases'][-1]['patch'])
        times = preview_times(10, 1000000, [e])
        for point in (0, 99999, 100000, 365123, 352779, 377467, 1000000):
            self.assertIn(point, times)
        self.assertGreaterEqual(np.count_nonzero((times > 352779) & (times < 365123)), 31)
        raw = np.zeros((10, 5))
        raw[1, 0] = .9
        result = emotion_at(raw, [99999, 100000], [e])
        np.testing.assert_allclose(result[:, 0], [0, .9])

    def test_validation_does_not_skip_an_offset_overflow_between_display_samples(self):
        raw = np.zeros((10, 5))
        raw[3, 1] = .95
        e = self.edit({'operation': 'interval_offset', 'value': .4, 'interpolation': 'smootherstep'})
        with self.assertRaises(ValueError):
            emotion_at(raw, validation_times(len(raw), 1000000, [e]), [e])

    def test_shared_visual_rounding_and_short_hue_arc(self):
        from legacy_pipeline import _interpolate_visual_state
        from mcts import VisualState
        from studio.timeline import VISUAL_BOUNDS
        fields = tuple(VISUAL_BOUNDS)
        sample = VECTORS['visual_samples']
        plan = VECTORS['visual_plan']
        states = [VisualState(**row) for row in plan['states']]
        rows = [_interpolate_visual_state(states, np.asarray(plan['times_us']) / 1e6, t / 1e6)
                for t in sample['times_us']]
        for key in ('hue_base', 'particle_count', 'trail_length'):
            np.testing.assert_allclose([getattr(row, key) for row in rows], sample[key], atol=1e-12)
        edits = validate_edits([VECTORS['visual_edit']], 1000000)
        base = [[getattr(row, field) for field in fields] for row in rows]
        edited = apply_edits(base, sample['times_us'], edits, 'visual', fields)
        np.testing.assert_array_equal(np.rint(edited[:, fields.index('particle_count')]), VECTORS['visual_edited_count'])
        for value, expected in VECTORS['round_cases']:
            self.assertEqual(np.rint(value), expected)

    def test_visual_validation_includes_integer_switches_and_offset_extrema(self):
        from legacy_pipeline import _interpolate_visual_state
        from mcts import VisualState
        from studio.timeline import VISUAL_BOUNDS
        plan = deepcopy(VECTORS['visual_plan'])
        checks = visual_validation_times(plan, 1000000, [])
        for point in VECTORS['visual_validation_required_times']:
            self.assertIn(point, checks)
        case = VECTORS['offset_extrema']
        plan['states'][0]['particle_speed'] = case['start_speed']
        plan['states'][1]['particle_speed'] = case['end_speed']
        edits = validate_edits([case['edit']], 1000000)
        checks = visual_validation_times(plan, 1000000, edits)
        states = [VisualState(**state) for state in plan['states']]
        rows = [_interpolate_visual_state(states, np.asarray(plan['times_us']) / 1e6, t / 1e6)
                for t in checks]
        fields = tuple(VISUAL_BOUNDS)
        base = [[getattr(row, field) for field in fields] for row in rows]
        with self.assertRaisesRegex(ValueError, '越界'):
            apply_edits(base, checks, edits, 'visual', fields)
        # Independent known-answer witness: both ends of the entering ramp are
        # valid, but an interior value exceeds the particle-speed maximum.
        t = case['witness_us']
        baseline = case['start_speed'] + (case['end_speed'] - case['start_speed']) * t / 1000000
        actual = baseline + case['edit']['value'] * weight_at(edits[0], [t])[0]
        self.assertAlmostEqual(actual, case['expected_witness_speed'])
        self.assertGreater(actual, 8.)


if __name__ == '__main__':
    unittest.main()
