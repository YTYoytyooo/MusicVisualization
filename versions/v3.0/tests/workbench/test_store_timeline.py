"""Storage/clock/migration behavior, independent of inference or GPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from tools.train_next_v2.v3_workbench.studio import store
from tools.train_next_v2.v3_workbench.studio import bridge
from tools.train_next_v2.v3_workbench.studio.bridge import import_v2
from tools.train_next_v2.v3_workbench.studio.timeline import TimedValues, raw_at, validate_edits
from tools.train_next_v2.v3_workbench.studio.smoothing import baseline_at, emotion_values


def edit(operation='interval_set', value=.7):
    return dict(id='e-one', layer='emotion', field='valence', operation=operation,
                start_us=500000, end_us=1000000, value=value,
                transition_in_us=100000, transition_out_us=100000, interpolation='smootherstep')


class NativeClockTests(unittest.TestCase):
    def test_verified_v2_final_hold_does_not_fill_interior_gaps_or_masks(self):
        raw = TimedValues([[.1, .2], [.3, .4], [.5, .6]], [0, 100000, 900000],
                          step_us=100000, duration_us=1150000, hold_last_to_end=True)
        values = raw_at(raw, [250000, 1000000, 1150000, 1150001])
        self.assertTrue(np.isnan(values[0]).all())
        np.testing.assert_allclose(values[1:3], [[.5, .6], [.5, .6]])
        self.assertTrue(np.isnan(values[3]).all())
        masked = TimedValues([[.1, .2], [.3, .4]], [0, 900000], mask=[[1, 1], [0, 1]],
                             step_us=100000, duration_us=1150000, hold_last_to_end=True)
        self.assertTrue(np.isnan(raw_at(masked, [1100000])[0, 0]))
        self.assertEqual(raw_at(masked, [1100000])[0, 1], .4)
        self.assertEqual(baseline_at(raw, [1150000], 100000, {'enabled': True, 'window_seconds': .5})[0, 1], .6)

    def test_native_offset_clock_is_not_stretched_to_duration(self):
        raw = TimedValues([[.1, .2], [.3, .4], [.5, .6]], [200000, 700000, 2200000],
                          step_us=500000, duration_us=2600000)
        values = raw_at(raw, [0, 200000, 699999, 700000, 1200000, 1200001, 2199999, 2200000])
        self.assertTrue(np.isnan(values[0]).all())
        np.testing.assert_allclose(values[1:4], [[.1, .2], [.1, .2], [.3, .4]])
        np.testing.assert_allclose(values[4], [.3, .4])
        self.assertTrue(np.isnan(values[5:7]).all())
        np.testing.assert_allclose(values[7], [.5, .6])

    def test_independent_masks_and_smoothing_do_not_cross_missing(self):
        raw = TimedValues([[0, 0], [.9, .5], [1, 1], [-1, -1]], [0, 500000, 1000000, 2000000],
                          mask=[[1, 1], [0, 1], [1, 1], [1, 1]], step_us=500000, duration_us=2500000)
        values = baseline_at(raw, [0, 250000, 500000, 1000000, 1600000, 2000000],
                             500000, {'enabled': True, 'window_seconds': 2})
        self.assertEqual(values[0, 0], 0)
        self.assertEqual(values[1, 0], 0)
        self.assertTrue(np.isnan(values[2, 0]))
        self.assertEqual(values[3, 0], 1)
        np.testing.assert_allclose(values[:4, 1], [.5, .5, .5, .5])
        self.assertTrue(np.isnan(values[4]).all())
        np.testing.assert_allclose(values[5], [-1, -1])

    def test_human_target_can_fill_effective_only_and_raw_remains_missing(self):
        raw = TimedValues([[.1, .2], [.3, .4], [.5, .6]], [0, 500000, 1000000],
                          mask=[[1, 1], [0, 1], [1, 1]], step_us=500000, duration_us=1500000)
        edits = validate_edits([edit()], 1500000)
        result = emotion_values(raw, [500000], edits, 500000)
        np.testing.assert_allclose(result, [[.7, .4]])
        self.assertTrue(np.isnan(raw[1, 0]))
        self.assertFalse(raw.mask[1, 0])

    def test_clock_and_valid_value_corruption_fail_closed(self):
        for clock in ([0, 0], [0, -1], [0, .5]):
            with self.assertRaises(ValueError):
                TimedValues([[0, 0], [0, 0]], clock)
        with self.assertRaises(ValueError):
            TimedValues([[np.nan, 0]], [0], mask=[[1, 1]])
        with self.assertRaises(ValueError):
            TimedValues([[2, 0]], [0])


class StoreAndMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.audio = self.root/'two seconds.wav'
        with wave.open(str(self.audio), 'wb') as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(8000)
            stream.writeframes(b'\0\0'*16000)
        self.systems = {
            'b01': dict(values=np.full((5, 2), .2), times_us=np.arange(5)*500000, step_us=500000, model_sha256='a'*64),
            'v2': dict(values=np.full((20, 2), .8), times_us=np.arange(20)*100000, step_us=100000, model_sha256='b'*64),
        }
        self.work = store.create_project(self.root/'works', 'native work', self.audio, None, None,
                    dict(duration=2), systems=self.systems, source_system='b01')

    def tearDown(self):
        self.temp.cleanup()

    def test_driver_switch_pins_each_revision_and_preserves_both_native_arrays(self):
        before = {p.name: store.digest(p) for p in (self.work/'analysis').glob('*raw.npz')}
        r = store.save_revision(self.work, 'r000000', [], source_system='v2')
        self.assertEqual(r['source_system'], 'v2')
        self.assertEqual(store.load_project(self.work)['step_us'], 100000)
        self.assertEqual(store.load_project(self.work, 'b01')['step_us'], 500000)
        self.assertEqual(store.load_revision(self.work, 'r000000')['source_system'], 'b01')
        np.testing.assert_allclose(store.load_raw(self.work), np.full((20, 2), .8))
        self.assertEqual(before, {p.name: store.digest(p) for p in (self.work/'analysis').glob('*raw.npz')})
        self.assertEqual(store.systems_metadata(self.work, True)[0]['times_us'], [0, 500000, 1000000, 1500000, 2000000])

    def test_stale_save_and_out_of_range_offset_do_not_publish_revision(self):
        store.save_revision(self.work, 'r000000', [])
        with self.assertRaisesRegex(ValueError, '版本已变化'):
            store.save_revision(self.work, 'r000000', [])
        with self.assertRaisesRegex(ValueError, '越界'):
            store.save_revision(self.work, 'r000001', [edit('interval_offset', .3)], source_system='v2')
        self.assertEqual(store.load_project(self.work)['current_revision'], 'r000001')
        self.assertFalse((self.work/'revisions/r000002').exists())

    def test_unknown_driver_and_tampered_raw_rejected(self):
        with self.assertRaises(ValueError):
            store.save_revision(self.work, 'r000000', [], source_system='fake')
        (self.work/'analysis/b01_raw.npz').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, '校验失败'):
            store.load_raw(self.work)

    def test_revision_chain_detects_changed_parent_even_when_parent_rehashed(self):
        store.save_revision(self.work, 'r000000', [])
        path = self.work/'revisions/r000000/revision.json'
        value = store.read_json(path)
        value['created_at'] = 'changed'
        value['sha256'] = store.object_digest({k: v for k, v in value.items() if k != 'sha256'})
        store.atomic_json(path, value)
        with self.assertRaisesRegex(ValueError, '父版本'):
            store.load_revision(self.work)

    def test_audio_is_portable_and_duration_mismatch_rejected(self):
        self.assertEqual(store.digest(self.audio), store.digest(self.work/store.load_project(self.work)['audio_file']))
        with self.assertRaisesRegex(ValueError, '时长'):
            store.create_project(self.root/'works', 'bad geometry', self.audio, None, None,
                                 dict(duration=3), systems=self.systems)
        self.assertEqual(len(store.list_projects(self.root/'works')), 1)

    def test_visual_seed_is_an_integer_and_not_a_boolean(self):
        for seed in (True, -1, 1.5, 2**32):
            with self.assertRaisesRegex(ValueError, '随机种子'):
                store.create_project(self.root/'works', 'badseed', self.audio, None, None,
                                     dict(duration=2), systems=self.systems, seed=seed)

    def test_frozen_import_adds_cached_acoustic_beats_without_changing_source_or_duration(self):
        source = self.root/'frozen'
        (source/'analysis').mkdir(parents=True)
        p = {'id': 'frozen'}
        result = {'audio': {'name': self.audio.name, 'path': str(self.audio),
                           'sha256': store.digest(self.audio), 'duration_seconds': 2},
                  'manifest_sha256': 'f'*64,
                  'systems': {'b01': {'raw_va': np.full((5, 2), .2).tolist(),
                                      'times_seconds': [0, .5, 1, 1.5, 2]}}}
        store.atomic_json(source/'project.json', p)
        store.atomic_json(source/'analysis/result.json', result)
        receipt = {'project_sha256': store.digest(source/'project.json'),
                   'result_sha256': store.digest(source/'analysis/result.json')}
        store.atomic_json(source/'analysis/receipt.json', receipt)
        before = {file.relative_to(source).as_posix(): store.digest(file) for file in source.rglob('*') if file.is_file()}
        manifest = {'systems': {'b01': {'members': []}}}
        module = 'tools.train_next_v2.v3_workbench.feature_extraction'
        with patch.object(bridge, '_verified_analysis', return_value=(p, result, receipt, manifest, self.audio)), \
                patch.object(bridge, 'DATA_ROOT', self.root/'data'), \
                patch(module+'.load_audio', return_value=(np.zeros(44100), 22050)), \
                patch(module+'.extract_global_info', return_value=dict(tempo=120, beat_times=np.array([.25, .75, 1.25, 1.75]), duration=2.00001)) as extract:
            first = bridge.import_analysis(source, self.root/'works', seed=123)
            second = bridge.import_analysis(source, self.root/'works', seed=321)
            self.assertEqual(extract.call_count, 1)
        self.assertEqual(before, {file.relative_to(source).as_posix(): store.digest(file) for file in source.rglob('*') if file.is_file()})
        imported = store.load_project(first)
        self.assertEqual(imported['tempo'], 120)
        self.assertEqual(imported['beat_times'], [.25, .75, 1.25, 1.75])
        self.assertEqual(imported['duration_us'], 2000000)
        self.assertEqual(imported['seed'], 123)
        self.assertEqual(store.load_project(second)['seed'], 321)
        self.assertEqual(imported['provenance']['source_result_sha256'], receipt['result_sha256'])
        self.assertEqual(store.digest(first/'analysis/source-v3/analysis/result.json'), receipt['result_sha256'])

    def _legacy(self, unknown=False):
        source = self.root/'v2-old'
        (source/'analysis').mkdir(parents=True)
        (source/'revisions/r000000').mkdir(parents=True)
        (source/'audio.wav').write_bytes(self.audio.read_bytes())
        five = np.column_stack((np.full((20, 2), .2), np.full((20, 3), .73)))
        np.savez_compressed(source/'analysis/predictions_raw.npz', values=five)
        raw_hash = store.digest(source/'analysis/predictions_raw.npz')
        p = dict(schema_version=1, id='p-old', name='old', duration=2, duration_us=2000000,
                 step_us=100000, current_revision='r000000', audio_file='audio.wav',
                 audio_sha256=store.digest(source/'audio.wav'), raw_sha256=raw_hash,
                 model_sha256='c'*64, seed=42, tempo=120, beat_times=[0, .5, 1, 1.5])
        r = dict(schema_version=1, id='r000000', parent=None, created_at='2026-01-01T00:00:00Z',
                 analysis_sha256=raw_hash, edits=validate_edits([edit()], 2000000))
        if unknown:
            r['new_unrecognised_mode'] = True
        r['sha256'] = store.object_digest(r)
        store.atomic_json(source/'project.json', p)
        store.atomic_json(source/'revisions/r000000/revision.json', r)
        return source

    def test_v2_import_archives_five_dimensions_without_changing_source(self):
        old = self._legacy()
        before = {p.relative_to(old).as_posix(): store.digest(p) for p in old.rglob('*') if p.is_file()}
        imported = import_v2(old, self.root/'works')
        self.assertEqual(before, {p.relative_to(old).as_posix(): store.digest(p) for p in old.rglob('*') if p.is_file()})
        for relative, expected in before.items():
            self.assertEqual(store.digest(imported/'archive/v2-original'/relative), expected)
        self.assertEqual(store.load_raw(imported).shape, (20, 2))
        with np.load(imported/'archive/v2-original/analysis/predictions_raw.npz') as data:
            np.testing.assert_allclose(data['values'][:, 2:], .73)
        r = store.load_revision(imported)
        self.assertEqual(r['edits'][0]['value'], .7)
        self.assertEqual(r['source_system'], 'v2')
        self.assertEqual(r['motion']['engine'], 'legacy')
        self.assertEqual(store.read_json(imported/'migration.json')['revision_mapping'], {'r000000': 'r000001'})

    def test_v2_unknown_fields_do_not_silently_publish_partial_import(self):
        old = self._legacy(unknown=True)
        with self.assertRaisesRegex(ValueError, '未支持字段'):
            import_v2(old, self.root/'works')
        self.assertEqual(len(store.list_projects(self.root/'works')), 1)

    def test_v2_import_preserves_synthetic_provenance_without_labelling_real_as_synthetic(self):
        old = self._legacy()
        real = import_v2(old, self.root/'works')
        self.assertFalse(store.load_project(real)['provenance']['is_synthetic'])
        p = store.read_json(old/'project.json')
        p['provenance'] = {'kind': 'synthetic', 'note': 'hand-designed fixture values'}
        store.atomic_json(old/'project.json', p)
        synthetic = import_v2(old, self.root/'works')
        provenance = store.load_project(synthetic)['provenance']
        self.assertTrue(provenance['is_synthetic'])
        self.assertEqual(provenance['source_provenance'], p['provenance'])


if __name__ == '__main__':
    unittest.main()
