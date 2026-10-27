"""Workbench render contracts: real clocks, frozen drivers and reusable plans."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from tools.train_next_v2.v3_workbench.studio import pipeline, store, motion
from tools.train_next_v2.v3_workbench.studio.timeline import TimedValues
from tools.train_next_v2.v3_workbench.motion_schema import MODE_LABELS, mode_params


class RenderPipelineTests(unittest.TestCase):
    def setUp(self):
        self.base = Path(__file__).resolve().parents[4] / 'data/v3.0/test-workbench'
        self.base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=self.base)
        self.root = Path(self.temporary.name)
        self.audio = self.root / 'source.wav'
        self.samples = np.rint(np.sin(np.arange(8800)*2*np.pi*220/8000)*2000).astype('<i2')
        with wave.open(str(self.audio), 'wb') as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(self.samples.tobytes())

    def tearDown(self):
        self.temporary.cleanup()

    def project(self, gap=False):
        b01 = np.array([[.7, .4], [.6, .3], [.5, .2]])
        mask = np.ones((3, 2), dtype=bool)
        if gap:
            mask[1, 0] = False
        systems = {
            'b01': {'values':b01, 'times_us':[0, 500000, 1000000], 'mask':mask,
                    'step_us':500000, 'model_sha256':'b01-weight', 'label':'B01'},
            'b00': {'values':-b01, 'times_us':[0, 500000, 1000000],
                    'step_us':500000, 'model_sha256':'b00-weight', 'label':'B00'},
        }
        return store.create_project(self.root/'projects', 'render-test', self.audio, None, None,
            {'duration':1.1, 'tempo':120., 'beat_times':[0., .5, 1.]},
            systems=systems, source_system='b01', provenance={'kind':'synthetic'})

    def test_default_full_song_and_high_quality_options(self):
        p = {'duration':401.032}
        result = pipeline.validate_render_options(p, {})
        self.assertEqual((result['width'], result['height'], result['fps']), (1280, 720, 30))
        self.assertEqual(result['end'], 401.032)
        hd = pipeline.validate_render_options(p, {'width':1920, 'height':1080, 'mode':'presentation'})
        self.assertEqual(hd['end'], p['duration'])
        with self.assertRaises(ValueError):
            pipeline.validate_render_options(p, {'width':1921})

    def test_driver_model_identity_separates_plans(self):
        path = self.project()
        a = pipeline.planning_key(path, [], source_system='b01')
        b = pipeline.planning_key(path, [], source_system='b00')
        self.assertNotEqual(a, b)

    def test_visual_only_change_reuses_plan_without_search(self):
        path = self.project()
        revision = store.load_revision(path)
        first = pipeline.plan(path, revision, iterations=4)
        visual = dict(id='color', layer='visual', field='hue_base', operation='interval_set',
            start_us=0, end_us=1100000, value=120, transition_in_us=0, transition_out_us=0)
        updated = {**revision, 'edits':[visual]}
        with patch('tools.train_next_v2.v3_workbench.mcts.MCTS.search', side_effect=AssertionError('replanned')):
            second = pipeline.plan(path, updated, iterations=4)
        self.assertEqual(first[2], second[2])
        np.testing.assert_array_equal(first[0], second[0])

    def test_native_masked_payload_uses_null_and_two_dimensions(self):
        path = self.project(gap=True)
        payload = pipeline.timeline_payload(path)
        self.assertEqual(payload['source_times_us'], [0, 500000, 1000000])
        self.assertEqual(payload['source_raw'][1], [None, .3])
        self.assertEqual(payload['source_mask'][1], [False, True])
        self.assertEqual([row['id'] for row in payload['comparison_systems']], ['b01', 'b00'])
        json.dumps(payload, allow_nan=False)

    def test_historical_v2_extras_keep_verified_native_clock_and_are_read_only(self):
        path = self.project()
        self.assertIsNone(pipeline.timeline_payload(path)['legacy_reference'])
        original_raw = store.load_raw(path).copy()
        archive = path/'archive/v2-original'
        source = archive/'analysis/predictions_raw.npz'
        source.parent.mkdir(parents=True)
        five = np.tile([-.2, .4, .6, -.8, 1.], (11, 1))
        np.savez_compressed(source, values=five)
        store.atomic_json(archive/'project.json', {'step_us':100000, 'raw_sha256':store.digest(source)})
        store.atomic_json(path/'migration.json', {'archive_sha256':{
            'project.json':store.digest(archive/'project.json'),
            'analysis/predictions_raw.npz':store.digest(source)}})
        p = store.read_json(path/'project.json')
        p['provenance'].update(legacy_archive='archive/v2-original', migration_report='migration.json')
        store.atomic_json(path/'project.json', p)
        payload = pipeline.timeline_payload(path)
        reference = payload['legacy_reference']
        self.assertEqual(reference['times_us'], list(range(0, 1100000, 100000)))
        self.assertEqual(reference['fields'], ['energy', 'tension', 'brightness'])
        self.assertEqual(reference['values'], five[:, 2:].tolist())
        self.assertTrue(reference['read_only'])
        self.assertIn('紧张度未获独立验证', reference['source_note'])
        np.testing.assert_array_equal(store.load_raw(path), original_raw)
        self.assertEqual(payload['source_times_us'], [0, 500000, 1000000])
        np.savez_compressed(source, values=five*.5)
        with self.assertRaisesRegex(ValueError, '历史 V2 五维参考校验失败'):
            pipeline.timeline_payload(path)

    def test_standalone_legacy_reference_requires_registered_hash(self):
        path = self.project()
        source = path/'analysis/legacy_five_dimensions.npz'
        five = np.tile([.1, .2, .3, .4, .5], (3, 1))
        np.savez_compressed(source, values=five)
        self.assertIsNone(pipeline.legacy_reference(path))
        p = store.read_json(path/'project.json')
        p['provenance'].update(legacy_reference_file='analysis/legacy_five_dimensions.npz',
            legacy_reference_sha256=store.digest(source), legacy_reference_step_us=500000)
        store.atomic_json(path/'project.json', p)
        reference = pipeline.legacy_reference(path)
        self.assertEqual(reference['times_us'], [0, 500000, 1000000])
        self.assertEqual(reference['values'], five[:, 2:].tolist())

    def test_archived_acoustic_reference_is_verified_before_display(self):
        path = self.project()
        source = path/'analysis/source-v3/analysis/result.json'
        store.atomic_json(source, {'acoustic':{'times_seconds':[0.], 'rms_amplitude':[.2]}})
        p = store.read_json(path/'project.json')
        p['provenance'].update(frozen_result_file=source.relative_to(path).as_posix(),
                               source_result_sha256=store.digest(source))
        store.atomic_json(path/'project.json', p)
        self.assertEqual(pipeline.timeline_payload(path)['acoustic']['rms_amplitude'], [.2])
        store.atomic_json(source, {'acoustic':{'times_seconds':[0.], 'rms_amplitude':[.8]}})
        with self.assertRaisesRegex(ValueError, '声学参考记录校验失败'):
            pipeline.timeline_payload(path)

    def test_missing_dimension_rejected_before_visual_search(self):
        path = self.project(gap=True)
        with patch('tools.train_next_v2.v3_workbench.mcts.MCTS.search', side_effect=AssertionError('searched gaps')):
            with self.assertRaisesRegex(ValueError, 'valence.*缺失'):
                pipeline.plan(path, store.load_revision(path), iterations=4)

    def test_missing_dimension_keeps_flow_timeline_inspectable(self):
        path = self.project(gap=True)
        with patch('tools.train_next_v2.v3_workbench.studio.motion_features.load_motion_features',
                   side_effect=AssertionError('decoded audio despite absent VA')):
            payload = pipeline.timeline_payload(path, motion={'engine':'flow-v1'})
        self.assertEqual(payload['motion']['status'], 'unavailable')
        self.assertIn('valence', payload['motion']['reason'])
        self.assertIsNone(payload['source_raw'][1][0])

    def test_snapshot_pins_driver_after_current_revision_changes(self):
        path = self.project()
        base = store.load_revision(path)
        saved = pipeline.create_preview_snapshot(path, base['id'], [], source_system='b00')
        store.save_revision(path, base['id'], [], source_system='b01')
        with patch.object(pipeline, 'render', return_value={'status':'captured'}) as render:
            result = pipeline.render_preview(path, saved['id'])
        self.assertEqual(result['status'], 'captured')
        self.assertEqual(render.call_args.kwargs['_snapshot']['source_system'], 'b00')
        self.assertEqual(saved['model_sha256'], 'b00-weight')

    def test_snapshot_rejects_changed_model_attribution_with_same_raw(self):
        path = self.project()
        saved = pipeline.create_preview_snapshot(path, store.load_revision(path)['id'], [], source_system='b00')
        p = store.read_json(path/'project.json')
        p['systems']['b00']['model_sha256'] = 'different-weights'
        p['systems_sha256'] = store.object_digest(p['systems'])
        store.atomic_json(path/'project.json', p)
        with self.assertRaisesRegex(ValueError, '驱动模型身份'):
            pipeline.render_preview(path, saved['id'])

    def test_render_uses_saved_driver_and_submits_exact_frame_values(self):
        path = self.project()
        original = store.load_revision(path)
        latest = store.save_revision(path, original['id'], [], source_system='b00')
        self.assertEqual(store.load_project(path)['source_system'], 'b00')
        # Audio decode seam keeps this check small; rendering and file output are real.
        with patch('tools.train_next_v2.v3_workbench.feature_extraction.load_audio',
                   return_value=(self.samples.astype(float)/32768, 8000)):
            result = pipeline.render(path, original['id'],
                {'width':320, 'height':180, 'fps':10, 'mode':'presentation'},
                allow_silent=True, iterations=4)
        self.assertEqual(result['status'], 'diagnostic_only')
        self.assertEqual(result['source_system'], 'b01')
        self.assertEqual(result['model_sha256'], 'b01-weight')
        self.assertEqual(result['frames'], 11)
        with (path/'renders'/result['id']/'frame_values.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 11)
        self.assertEqual(float(rows[0]['valence_effective']), .7)
        self.assertEqual(float(rows[-1]['valence_effective']), .5)
        self.assertNotIn('tension_effective', rows[0])
        self.assertEqual(store.load_revision(path)['id'], latest['id'])

    def test_manual_effective_fill_renders_without_rewriting_missing_raw(self):
        path = self.project(gap=True)
        base = store.load_revision(path)
        revision = store.save_revision(path, base['id'], [dict(id='human-value', layer='emotion',
            field='valence', operation='interval_set', start_us=500000, end_us=1000000,
            transition_in_us=0, transition_out_us=0, value=.25)])
        for mode in ('presentation', 'analysis'):
            with self.subTest(mode=mode):
                with patch('tools.train_next_v2.v3_workbench.feature_extraction.load_audio',
                           return_value=(self.samples.astype(float)/32768, 8000)):
                    result = pipeline.render(path, revision['id'],
                        {'width':320, 'height':180, 'fps':10, 'mode':mode},
                        allow_silent=True, iterations=4)
                self.assertEqual(result['status'], 'diagnostic_only')
                with (path/'renders'/result['id']/'frame_values.csv').open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                missing = rows[6]
                self.assertEqual(missing['valence_raw'], '')
                self.assertEqual(missing['valence_raw_valid'], 'False')
                self.assertEqual(missing['arousal_raw_valid'], 'True')
                self.assertEqual(float(missing['arousal_raw']), .3)
                self.assertEqual(float(missing['valence_effective']), .25)
        self.assertTrue(np.isnan(store.load_raw(path)[1, 0]))

    def test_analysis_overlay_marks_raw_gap_and_does_not_connect_it(self):
        from tools.train_next_v2.v3_workbench.prediction_overlay import PredictionOverlay
        effective = [dict(valence=.1,arousal=.2)]*3
        raw = [dict(valence=-.5,arousal=.2),dict(valence=np.nan,arousal=.2),dict(valence=.5,arousal=.2)]
        hud = PredictionOverlay('gap',1.1,effective,raw_emotion_states=raw,times=[0,.5,1.])
        messages = []
        actual_text = hud._text
        def capture(image, text, *args, **kwargs):
            messages.append(text)
            return actual_text(image, text, *args, **kwargs)
        with patch.object(hud, '_text', side_effect=capture):
            hud.draw(np.zeros((720,1280,3),dtype=np.uint8), .6)
        self.assertTrue(any('RAW GAP: V' in text for text in messages))
        self.assertTrue(np.isnan(hud.raw_values[1, 0]))
        self.assertFalse(hud._raw_history_mask.any())
        hud._advance_history(2)
        # A missing middle observation never creates a line from left to right.
        midpoint = hud.point_for(0., .2)
        self.assertEqual(int(hud._raw_history_mask[midpoint[1],midpoint[0]]), 0)

    def test_motion_only_changes_reuse_auto_plan_without_mcts(self):
        path = self.project()
        raw = store.load_raw(path)
        features = {'key':'fixed-audio', 'kind':'acoustic-test', 'times_us':np.array([0,500000,1000000]),
            'rms':np.array([.1,.1,.1]), 'onset':np.array([0.,0.,0.]),
            'activity':np.array([.4,.4,.4]), 'pulse_strength':np.array([0.,0.,0.])}
        config = {'engine':'flow-v1'}
        with patch('tools.train_next_v2.v3_workbench.studio.motion_features.load_motion_features', return_value=features):
            first = motion.build_motion_plan(path, raw, [], None, config, source_system='b01')
            config['overrides'] = [dict(id='wave',mode='wave',start_us=0,end_us=1000000,
                transition_in_us=0,transition_out_us=0)]
            with (patch.object(motion, 'automatic_plan', side_effect=AssertionError('auto recomputed')),
                  patch('tools.train_next_v2.v3_workbench.mcts.MCTS.search', side_effect=AssertionError('MCTS used'))):
                second = motion.build_motion_plan(path, raw, [], None, config, source_system='b01')
        self.assertEqual(first['automatic_plan_key'], second['automatic_plan_key'])
        self.assertEqual(second['effective'][0]['mode'], 'wave')

    def test_audio_feature_hit_skips_decode_and_extraction(self):
        from tools.train_next_v2.v3_workbench.studio import motion_features
        path = self.project()
        first = motion_features.load_motion_features(path)
        with (patch.object(motion_features, '_motion_audio', side_effect=AssertionError('decoded cache hit')),
              patch.object(motion_features, '_extract', side_effect=AssertionError('extracted cache hit'))):
            second = motion_features.load_motion_features(path)
        self.assertEqual(first['key'], second['key'])
        np.testing.assert_array_equal(first['rms'], second['rms'])

    def test_all_nine_modes_and_checkpoint_resume_are_real_deterministic(self):
        from tools.train_next_v2.v3_workbench.mcts import VisualState
        from tools.train_next_v2.v3_workbench.renderer import VideoRenderer
        from tools.train_next_v2.v3_workbench.studio.checkpoints import save_checkpoint, load_checkpoint
        a = VideoRenderer(str(self.root/'a.avi'), width=320, height=180, rng_seed=7)
        b = VideoRenderer(str(self.root/'b.avi'), width=320, height=180, rng_seed=7)
        visual = VisualState(particle_count=50, trail_length=5)
        try:
            self.assertEqual(len(MODE_LABELS), 9)
            for index, mode in enumerate(MODE_LABELS):
                control = {'mode':mode,'source':'manual','reason':'contract',
                    'components':[{'mode':mode,'weight':1.,'params':mode_params(mode)}]}
                frame = a.render_frame(visual, np.zeros(100), .1, index*.01, .2, motion=control)
                self.assertEqual(frame.shape, (180, 320, 3))
            checkpoint = save_checkpoint(self.root/'checkpoints', 'same-settings', a)
            load_checkpoint(checkpoint, 'same-settings', b)
            for i in range(3):
                left = a.render_frame(visual, np.zeros(100), .1, .1+i*.01, .2, motion=control)
                right = b.render_frame(visual, np.zeros(100), .1, .1+i*.01, .2, motion=control)
                np.testing.assert_array_equal(left, right)
            with self.assertRaises(ValueError):
                load_checkpoint(checkpoint, 'different-settings', b)
        finally:
            a.release()
            b.release()


if __name__ == '__main__':
    unittest.main()
