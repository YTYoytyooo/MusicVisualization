"""Persisted preview/revision contracts on nonuniform clocks and independent gaps.

Fixtures are synthetic engineering data, never training labels. These tests do
not infer, train, render pixels or invoke an encoder.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from tools.train_next_v2.v3_workbench.studio import pipeline, store
from tools.train_next_v2.v3_workbench.studio.smoothing import baseline_at
from tools.train_next_v2.v3_workbench.studio.timeline import TimedValues


class SmoothingGapTests(unittest.TestCase):
    def test_expired_native_support_splits_smoothing_runs(self):
        raw = TimedValues([[0, 0], [0, 0], [1, 1]], [0, 500000, 1100000],
                          step_us=500000, duration_us=1600000)
        targets = [0, 500000, 750000, 1000000, 1000001, 1099999, 1100000, 1600000]
        expected = [[0, 0], [0, 0], [0, 0], [0, 0], [np.nan, np.nan],
                    [np.nan, np.nan], [1, 1], [1, 1]]
        actual = baseline_at(raw, targets, 500000, {'enabled': True, 'window_seconds': 2})
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14, equal_nan=True)
        np.testing.assert_array_equal(raw, [[0, 0], [0, 0], [1, 1]])

    def test_regular_clock_interpolation_and_axis_mask_are_unchanged(self):
        raw = TimedValues([[0, 0], [.5, .5], [1, 1]], [0, 500000, 1000000],
                          mask=[[1, 1], [0, 1], [1, 1]],
                          step_us=500000, duration_us=1500000)
        actual = baseline_at(raw, [250000, 500000, 750000, 1000000, 1500000],
                             500000, {'enabled': True, 'window_seconds': .1})
        expected = [[0, .25], [np.nan, .5], [np.nan, .75], [1, 1], [1, 1]]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14, equal_nan=True)


class RevisionParityTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[4] / 'data/v3.0/test-workbench'
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=base)
        self.root = Path(self.temporary.name)
        audio = self.root / 'engineering-fixture.wav'
        with wave.open(str(audio), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(b'\0\0' * 48000)
        self.clock = [0, 400000, 900000, 1600000, 2000000, 2500000,
                      3000000, 3500000, 4000000, 4500000, 5000000, 5500000, 6000000]
        values = np.tile([.2, .4], (len(self.clock), 1))
        values[:5] = [[.1, .2], [.3, .4], [.5, .6], [-.2, -.3], [.7, .8]]
        self.mask = np.ones_like(values, dtype=bool)
        self.mask[1, 0] = False
        self.mask[4, 1] = False
        self.expected_raw = values.astype(object).tolist()
        self.expected_raw[1][0] = None
        self.expected_raw[4][1] = None
        self.work = store.create_project(
            self.root / 'works', 'Synthetic revision parity fixture', audio, None, None,
            {'duration': 6}, source_system='b01', provenance={'kind': 'synthetic'},
            systems={'b01': dict(values=values, times_us=self.clock, mask=self.mask,
                                 step_us=500000, model_sha256='a' * 64)})
        self.raw_hash = store.digest(self.work / 'analysis/b01_raw.npz')

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def target(identity, start, end, value):
        return dict(id=identity, layer='emotion', field='valence',
                    operation='interval_set', start_us=start, end_us=end,
                    transition_in_us=0, transition_out_us=0, value=value)

    def assert_source_unchanged(self, payload):
        self.assertEqual(payload['source_times_us'], self.clock)
        self.assertEqual(payload['source_mask'], self.mask.tolist())
        self.assertEqual(payload['source_raw'], self.expected_raw)
        self.assertEqual(store.digest(self.work / 'analysis/b01_raw.npz'), self.raw_hash)

    def test_preview_and_saved_revision_preserve_native_gaps_with_smoothing(self):
        edits = [self.target('manual-fill', 400000, 900000, .75)]
        smoothing = {'enabled': True, 'window_seconds': .1}
        snapshot = pipeline.create_preview_snapshot(
            self.work, 'r000000', edits, smoothing=smoothing, edit_id='manual-fill')
        saved = store.save_revision(self.work, 'r000000', edits, smoothing=smoothing)
        # The preview's frozen content and the normal saved-revision route must
        # resolve to the same public chart, without rewriting model evidence.
        preview = pipeline.timeline_payload(
            self.work, revision=snapshot['base_revision'], edits=snapshot['edits'],
            smoothing=snapshot['smoothing'], motion=snapshot['motion'],
            source_system=snapshot['source_system'])
        formal = pipeline.timeline_payload(self.work, revision=saved['id'])
        self.assertEqual(preview, formal)
        self.assert_source_unchanged(preview)
        at = {t: i for i, t in enumerate(preview['times_us'])}
        self.assertEqual(preview['raw'][at[400000]], [None, .4])
        np.testing.assert_allclose(preview['effective'][at[400000]], [.75, .4], rtol=0, atol=1e-14)
        self.assertEqual(preview['raw'][at[2000000]], [.7, None])
        self.assertAlmostEqual(preview['effective'][at[2000000]][0], .7, places=14)
        self.assertIsNone(preview['effective'][at[2000000]][1])
        # No observation exists between 1.400001 s and 1.6 s. Neither smoothing
        # nor chart construction may interpolate across that hole.
        self.assertEqual(preview['baseline'][at[1500000]], [None, None])
        self.assertEqual(preview['effective'][at[1500000]], [None, None])
        self.assertEqual(preview['times_us'][-1], 6000000)

    def test_selected_preview_pins_absolute_clip_and_content_after_later_save(self):
        selected = self.target('selected', 3250000, 3750000, .65)
        selected.update(transition_in_us=100000, transition_out_us=100000)
        snapshot = pipeline.create_preview_snapshot(
            self.work, 'r000000', [selected], edit_id='selected')
        # Support [3.15, 3.85] s plus 2 s of context, on the song clock.
        self.assertAlmostEqual(snapshot['options']['start'], 1.15)
        self.assertAlmostEqual(snapshot['options']['end'], 5.85)
        snapshot_path = self.work / 'previews' / snapshot['id'] / 'snapshot.json'
        snapshot_hash = store.digest(snapshot_path)
        later = store.save_revision(self.work, 'r000000', [
            self.target('different-value', 3250000, 3750000, -.6)])
        # Observe the real preview-to-render dispatch boundary. No video work
        # occurs, and this does not claim pixel/audio parity of rendered clips.
        with patch.object(pipeline, 'render', return_value={'status': 'dispatch-only'}) as render:
            self.assertEqual(pipeline.render_preview(self.work, snapshot['id']),
                             {'status': 'dispatch-only'})
        frozen = render.call_args.kwargs['_snapshot']
        self.assertEqual(frozen['edits'][0]['value'], .65)
        self.assertEqual(frozen['source_system'], 'b01')
        self.assertEqual(render.call_args.kwargs['options'], snapshot['options'])
        self.assertEqual(store.digest(snapshot_path), snapshot_hash)
        self.assertEqual(store.load_revision(self.work)['id'], later['id'])
        self.assertEqual(store.digest(self.work / 'analysis/b01_raw.npz'), self.raw_hash)

    def test_incomplete_preview_stays_inspectable_but_planning_rejects_missing_axis(self):
        edits = [self.target('manual-valence-only', 400000, 900000, .75)]
        payload = pipeline.validate_preview(self.work, edits, motion={'engine': 'flow-v1'})
        self.assertEqual(payload['motion']['status'], 'unavailable')
        self.assert_source_unchanged(payload)
        # Filling one axis does not authorize synthetic values in the other
        # missing axis or in a temporal gap. Reject before MCTS/audio features.
        with (patch('tools.train_next_v2.v3_workbench.mcts.MCTS.search',
                    side_effect=AssertionError('planning consumed incomplete evidence')),
              patch('tools.train_next_v2.v3_workbench.studio.motion_features.load_motion_features',
                    side_effect=AssertionError('decoded audio before rejecting gaps'))):
            with self.assertRaises(pipeline.MissingObservations):
                pipeline.plan(self.work, {'edits': edits, 'source_system': 'b01'})
            again = pipeline.validate_preview(self.work, edits, motion={'engine': 'flow-v1'})
        self.assertEqual(again['motion']['status'], 'unavailable')
        self.assertEqual(store.load_project(self.work)['current_revision'], 'r000000')


if __name__ == '__main__':
    unittest.main()
