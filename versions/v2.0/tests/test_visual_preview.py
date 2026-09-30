import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import json
import io
import sys
from contextlib import redirect_stderr, redirect_stdout

from studio.demo import create_demo
from studio.pipeline import cached_visual_plan, prepare_visual_plan, validate_preview, planning_key, plan
from studio.jobs import JobManager


class VisualPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = create_demo(self.root, 1.)

    def tearDown(self):
        self.temp.cleanup()

    def edit(self, **changes):
        value = dict(id='speed', layer='visual', field='particle_speed', operation='point_target',
                     time_us=500000, value=3., transition_in_us=250000,
                     transition_out_us=250000, enabled=True, interpolation='linear')
        value.update(changes)
        return value

    def test_real_plan_is_reused_for_visual_only_changes(self):
        self.assertIsNone(cached_visual_plan(self.project, []))
        result = prepare_visual_plan(self.project, 'r000000', [])
        self.assertEqual(result['project_id'], self.project.name)
        original = cached_visual_plan(self.project, [])
        self.assertTrue(original['states'])
        with patch('mcts.MCTS.search', side_effect=AssertionError('search repeated')):
            self.assertEqual(cached_visual_plan(self.project, [self.edit()])['key'], original['key'])
            payload = validate_preview(self.project, [self.edit()])
        self.assertIn('source_raw', payload)
        self.assertEqual(payload['step_us'], 100000)

    def test_emotion_change_does_not_use_old_visual_plan(self):
        prepare_visual_plan(self.project, 'r000000', [])
        emotion = self.edit(layer='emotion', field='arousal', value=.8)
        self.assertIsNone(cached_visual_plan(self.project, [emotion]))

    def test_offset_out_of_bounds_rejected_after_real_baseline(self):
        prepare_visual_plan(self.project, 'r000000', [])
        edit = self.edit(operation='interval_offset', start_us=250000, end_us=750000, value=100.)
        with self.assertRaisesRegex(ValueError, '越界'):
            validate_preview(self.project, [edit])

    def test_visual_preview_requires_real_baseline(self):
        with self.assertRaisesRegex(ValueError, '规划'):
            validate_preview(self.project, [self.edit()])

    def test_plan_requests_deduplicated_but_failure_can_retry(self):
        with patch.object(JobManager, '_loop', return_value=None):
            manager = JobManager(self.root)
        try:
            payload = {'project_id': self.project.name, 'base_revision': 'r000000', 'edits': []}
            a = manager.submit('plan', payload, dedupe_key='same')
            b = manager.submit('plan', payload, dedupe_key='same')
            self.assertEqual(a['id'], b['id'])
            manager.cancel(a['id'])
            c = manager.submit('plan', payload, dedupe_key='same')
            self.assertNotEqual(a['id'], c['id'])
        finally:
            manager.close()

    def test_old_missing_interpolation_has_same_plan_key_without_mutation(self):
        edit = self.edit(layer='emotion', field='arousal', value=.8)
        old = dict(edit)
        del old['interpolation']
        self.assertEqual(planning_key(self.project, [old]), planning_key(self.project, [edit]))
        self.assertNotIn('interpolation', old)

    def test_short_emotion_transition_changes_actual_planning_inputs(self):
        from mcts import VisualState
        values = {}
        grids = []
        for mode in ('linear', 'smoothstep', 'smootherstep'):
            edit = self.edit(layer='emotion', field='arousal', value=.8, time_us=250000, interpolation=mode)
            captured = []
            def search(v, a, **kwargs):
                captured.append(a)
                return VisualState()
            with patch('mcts.MCTS.search', side_effect=search):
                ts, _, _ = plan(self.project, {'edits':[edit]}, iterations=1)
            grids.append(ts)
            values[mode] = captured[list(ts).index(62500)]
        np.testing.assert_array_equal(grids[0], grids[1])
        np.testing.assert_array_equal(grids[1], grids[2])
        self.assertGreater(values['linear'], values['smoothstep'])
        self.assertGreater(values['smoothstep'], values['smootherstep'])

    def test_interior_visual_overflow_rejected_by_preview_cli_and_render(self):
        from studio.store import atomic_json, object_digest, read_json, load_project, save_revision
        from studio.pipeline import render
        from main import main
        case = read_json(Path(__file__).with_name('transition_vectors.json'))['offset_extrema']
        prepare_visual_plan(self.project, 'r000000', [])
        original = cached_visual_plan(self.project, [])
        left, right = dict(original['states'][0]), dict(original['states'][-1])
        left['particle_speed'], right['particle_speed'] = case['start_speed'], case['end_speed']
        data = dict(key=original['key'], times_us=[0, 1000000], states=[left, right])
        data['sha256'] = object_digest(data)
        atomic_json(self.project / 'plans' / data['key'] / 'plan.json', data)
        edit = case['edit']
        with self.assertRaisesRegex(ValueError, '越界'):
            validate_preview(self.project, [edit])

        # Exercise the actual CLI parser/service, not only the helper function.
        edit_file = self.root / 'overflow.json'
        atomic_json(edit_file, [edit])
        error = io.StringIO()
        with (patch.object(sys, 'argv', ['main.py', 'edit', str(self.project),
                                        '--base', 'r000000', '--edits', str(edit_file)]),
              redirect_stderr(error), redirect_stdout(io.StringIO())):
            self.assertEqual(main(), 1)
        self.assertIn('越界', error.getvalue())
        self.assertEqual(load_project(self.project)['current_revision'], 'r000000')

        # Imported/legacy revisions can bypass the UI, but rendering must still
        # reject the internal extremum before any frame is written.
        imported = save_revision(self.project, 'r000000', [edit])
        with self.assertRaisesRegex(ValueError, '越界'):
            render(self.project, imported['id'], dict(fps=30, width=320, height=180), allow_silent=True)


if __name__ == '__main__':
    unittest.main()
