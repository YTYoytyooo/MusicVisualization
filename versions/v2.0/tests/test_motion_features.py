import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from studio import motion_features as motion
from studio.store import atomic_json, digest, project_lock, read_json


class MotionFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'project'
        self.path.mkdir()

    def project(self, samples, rate=8000):
        audio = self.path / 'audio.wav'
        sf.write(audio, samples, rate, subtype='DOUBLE')
        atomic_json(self.path / 'project.json', {
            'schema_version': 1, 'audio_file': audio.name,
            'audio_sha256': digest(audio), 'current_revision': 'r000004',
            'duration_us': int(round(len(samples) / rate * 1000000)),
        })
        (self.path / 'analysis').mkdir(exist_ok=True)
        (self.path / 'analysis' / 'predictions_raw.npz').write_bytes(b'untouched prediction fixture')
        return audio

    def test_silence_and_existing_project_data_are_unchanged(self):
        audio = self.project(np.zeros(8200))
        protected = [audio, self.path / 'project.json', self.path / 'analysis/predictions_raw.npz']
        before = [digest(path) for path in protected]
        result = motion.load_motion_features(self.path)
        self.assertEqual(result['kind'], 'audio-rms-spectral-flux-v1')
        self.assertEqual(result['step_us'], 100000)
        np.testing.assert_array_equal(result['times_us'], np.arange(11, dtype=np.int64) * 100000)
        self.assertEqual(result['times_us'].dtype, np.dtype('int64'))
        for name in ('rms', 'onset', 'activity', 'pulse_strength'):
            self.assertEqual(result[name].dtype, np.dtype('float64'))
            np.testing.assert_array_equal(result[name], np.zeros(11))
        self.assertEqual(before, [digest(path) for path in protected])

    def test_quiet_constant_audio_does_not_normalize_to_high_activity(self):
        self.project(np.full(8200, .001))
        result = motion.load_motion_features(self.path)
        np.testing.assert_allclose(result['rms'], .001, atol=1e-12)
        self.assertLess(float(np.max(result['activity'])), .02)
        np.testing.assert_array_equal(result['pulse_strength'], np.zeros(11))

    def test_constant_tone_has_no_fake_beats_including_partial_eof(self):
        rate = 8000
        t = np.arange(8247) / rate
        self.project(.2 * np.sin(2 * np.pi * 440 * t), rate)
        result = motion.load_motion_features(self.path)
        self.assertLess(float(np.max(result['pulse_strength'])), .01)

    def test_rhythm_onsets_follow_real_loud_blocks_and_stereo_is_not_cancelled(self):
        rate = 8000
        samples = np.zeros(rate * 2)
        tone = .4 * np.sin(2 * np.pi * 440 * np.arange(800) / rate)
        beats = [2, 6, 10, 14, 18]
        for index in beats:
            samples[index * 800:(index + 1) * 800] = tone
        self.project(np.column_stack((samples, -samples)), rate)
        result = motion.load_motion_features(self.path)
        np.testing.assert_allclose(result['rms'][beats], .4 / np.sqrt(2), atol=1e-12)
        self.assertTrue(np.all(result['onset'][beats] > .1))
        self.assertTrue(np.all(result['pulse_strength'][beats] > .8))
        self.assertTrue(np.all(result['activity'][beats] > .99))
        silent = np.setdiff1d(np.arange(20), beats)
        np.testing.assert_array_equal(result['pulse_strength'][silent], np.zeros(len(silent)))

    def test_noninteger_sample_hops_do_not_drift_and_rms_is_direct(self):
        rate = 11025
        samples = np.linspace(-.2, .2, 11701)
        self.project(samples, rate)
        result = motion.load_motion_features(self.path)
        np.testing.assert_array_equal(result['times_us'], np.arange(11) * 100000)
        expected = []
        for index in range(11):
            block = samples[index * rate // 10:min(len(samples), (index + 1) * rate // 10)]
            expected.append(np.sqrt(np.mean(block ** 2)))
        np.testing.assert_allclose(result['rms'], expected, atol=1e-12)

    def test_cache_reused_without_extraction_and_returned_arrays_are_independent(self):
        self.project(np.full(8000, .1))
        first = motion.load_motion_features(self.path)
        first['rms'][0] = 99
        with patch.object(motion, '_extract', side_effect=AssertionError('cache miss')):
            second = motion.load_motion_features(self.path)
        self.assertEqual(first['key'], second['key'])
        self.assertAlmostEqual(second['rms'][0], .1)

    def test_damaged_cache_is_preserved_then_rebuilt(self):
        self.project(np.full(8000, .1))
        first = motion.load_motion_features(self.path)
        folder = self.path / 'motion_features'
        cached = folder / (first['key'] + '.npz')
        cached.write_bytes(b'broken cache retained for diagnosis')
        second = motion.load_motion_features(self.path)
        np.testing.assert_array_equal(first['rms'], second['rms'])
        invalid = list(folder.glob('invalid-*'))
        self.assertEqual(len(invalid), 1)
        self.assertEqual((invalid[0] / cached.name).read_bytes(), b'broken cache retained for diagnosis')

    def test_matching_checksum_does_not_bypass_time_shape_and_bound_validation(self):
        self.project(np.full(8000, .1))
        reference = motion.load_motion_features(self.path)
        folder = self.path / 'motion_features'
        data = folder / (reference['key'] + '.npz')
        meta_path = folder / (reference['key'] + '.json')
        for field, bad in [('times_us', np.ones(10, dtype=np.int64)),
                           ('rms', np.zeros((10, 1))),
                           ('activity', np.full(10, 1.1)),
                           ('onset', np.full(10, np.nan)),
                           ('pulse_strength', np.zeros(10, dtype=np.float32))]:
            with self.subTest(field=field):
                values = {name: reference[name].copy() for name in motion.FIELDS}
                values[field] = bad
                np.savez_compressed(data, **values)
                meta = read_json(meta_path)
                meta['sha256'] = digest(data)
                atomic_json(meta_path, meta)
                rebuilt = motion.load_motion_features(self.path)
                np.testing.assert_array_equal(rebuilt[field], reference[field])
        self.assertEqual(len(list(folder.glob('invalid-*'))), 5)

    def test_cancellation_does_not_publish_and_can_retry(self):
        self.project(np.ones(8000) * .1)
        calls = []
        def progress(stage, value):
            calls.append((stage, value))
        with self.assertRaises(motion.MotionFeatureCancelled):
            motion.load_motion_features(self.path, progress=progress, cancel=lambda: bool(calls))
        self.assertEqual(list((self.path / 'motion_features').glob('*.json')), [])
        result = motion.load_motion_features(self.path)
        self.assertEqual(len(result['rms']), 10)

    def test_audio_changed_during_extraction_is_not_published(self):
        audio = self.project(np.ones(8000) * .1)
        original = motion._extract
        def change(*args):
            values = original(*args)
            sf.write(audio, np.ones(8000) * .2, 8000, subtype='DOUBLE')
            return values
        with patch.object(motion, '_extract', side_effect=change):
            with self.assertRaisesRegex(ValueError, '音频在特征提取过程中变化'):
                motion.load_motion_features(self.path)
        self.assertEqual(list((self.path / 'motion_features').glob('*.json')), [])
        self.assertEqual(list((self.path / 'motion_features').glob('*.npz')), [])

    def test_code_change_during_extraction_is_not_published(self):
        self.project(np.ones(8000) * .1)
        with patch.object(motion, '_fingerprint', side_effect=['before', 'after']):
            with self.assertRaisesRegex(ValueError, '代码在提取过程中变化'):
                motion.load_motion_features(self.path)
        self.assertEqual(list((self.path / 'motion_features').glob('*.json')), [])

    def test_cache_configuration_version_change_creates_new_key(self):
        self.project(np.ones(8000) * .1)
        first = motion.load_motion_features(self.path)
        with patch.object(motion, 'RMS_FLOOR', .2):
            second = motion.load_motion_features(self.path)
        self.assertNotEqual(first['key'], second['key'])
        np.testing.assert_allclose(second['activity'], .5)
        self.assertEqual(len(list((self.path / 'motion_features').glob('*.npz'))), 2)

    def test_nonobject_metadata_is_quarantined_and_progress_details_are_optional(self):
        self.project(np.zeros(8000))
        reports = []
        def report(stage, fraction, details):
            reports.append((stage, fraction, details))
        report.accepts_details = True
        first = motion.load_motion_features(self.path, progress=report)
        self.assertEqual(reports[-1][1], 1)
        self.assertEqual(reports[-1][2]['done'], 10)
        self.assertEqual(reports[-1][2]['total'], 10)
        folder = self.path / 'motion_features'
        atomic_json(folder / (first['key'] + '.json'), [])
        rebuilt = motion.load_motion_features(self.path)
        self.assertEqual(first['key'], rebuilt['key'])
        self.assertEqual(len(list(folder.glob('invalid-*'))), 1)

    def test_cache_lock_does_not_write_over_another_owner(self):
        self.project(np.zeros(8000))
        folder = self.path / 'motion_features'
        folder.mkdir()
        with project_lock(folder):
            with self.assertRaisesRegex(ValueError, '项目正在写入'):
                motion.load_motion_features(self.path)
        self.assertEqual(list(folder.glob('*.json')), [])


if __name__ == '__main__':
    unittest.main()
